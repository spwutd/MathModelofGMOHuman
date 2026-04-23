#!/usr/bin/env python3
"""
pipeline/codon_optimizer.py
Human codon optimization pipeline:
  1. Input: amino acid sequence
  2. Replace each codon with optimal human codon
  3. Fix forbidden restriction sites (EcoRI, NotI, XhoI, BamHI, HindIII)
  4. Remove cryptic splice sites (GT-AG rule, donor/acceptor consensus)
  5. Calculate Codon Adaptation Index (CAI)
  6. Output: optimized CDS + QC metrics
"""

import re, json, os, random, hashlib

# ── Human codon usage table (Kazusa 2023, H.sapiens) ─────────────────────────
# Format: codon → (AA, relative_frequency_0_to_1)
# Frequency 1.0 = most used for that AA
HUMAN_CODON_FREQ = {
    # Phe F
    'TTT': ('F', 0.54), 'TTC': ('F', 1.00),
    # Leu L
    'TTA': ('L', 0.10), 'TTG': ('L', 0.27), 'CTT': ('L', 0.38),
    'CTC': ('L', 0.68), 'CTA': ('L', 0.17), 'CTG': ('L', 1.00),
    # Ile I
    'ATT': ('I', 0.72), 'ATC': ('I', 1.00), 'ATA': ('I', 0.33),
    # Met M
    'ATG': ('M', 1.00),
    # Val V
    'GTT': ('V', 0.38), 'GTC': ('V', 0.58), 'GTA': ('V', 0.23), 'GTG': ('V', 1.00),
    # Ser S
    'TCT': ('S', 0.63), 'TCC': ('S', 0.85), 'TCA': ('S', 0.49),
    'TCG': ('S', 0.18), 'AGT': ('S', 0.53), 'AGC': ('S', 1.00),
    # Pro P
    'CCT': ('P', 0.72), 'CCC': ('P', 1.00), 'CCA': ('P', 0.76), 'CCG': ('P', 0.20),
    # Thr T
    'ACT': ('T', 0.59), 'ACC': ('T', 1.00), 'ACA': ('T', 0.70), 'ACG': ('T', 0.23),
    # Ala A
    'GCT': ('A', 0.72), 'GCC': ('A', 1.00), 'GCA': ('A', 0.55), 'GCG': ('A', 0.18),
    # Tyr Y
    'TAT': ('Y', 0.53), 'TAC': ('Y', 1.00),
    # Stop *
    'TAA': ('*', 0.47), 'TAG': ('*', 0.20), 'TGA': ('*', 1.00),
    # His H
    'CAT': ('H', 0.57), 'CAC': ('H', 1.00),
    # Gln Q
    'CAA': ('Q', 0.35), 'CAG': ('Q', 1.00),
    # Asn N
    'AAT': ('N', 0.53), 'AAC': ('N', 1.00),
    # Lys K
    'AAA': ('K', 0.58), 'AAG': ('K', 1.00),
    # Asp D
    'GAT': ('D', 0.63), 'GAC': ('D', 1.00),
    # Glu E
    'GAA': ('E', 0.58), 'GAG': ('E', 1.00),
    # Cys C
    'TGT': ('C', 0.52), 'TGC': ('C', 1.00),
    # Trp W
    'TGG': ('W', 1.00),
    # Arg R
    'CGT': ('R', 0.23), 'CGC': ('R', 0.55), 'CGA': ('R', 0.21),
    'CGG': ('R', 0.49), 'AGA': ('R', 0.40), 'AGG': ('R', 1.00),
    # Ser S (continued above)
    # Gly G
    'GGT': ('G', 0.40), 'GGC': ('G', 0.76), 'GGA': ('G', 0.47), 'GGG': ('G', 1.00),
}

# Best codon per amino acid (highest frequency)
_AA_CODONS: dict[str, list] = {}
for codon, (aa, freq) in HUMAN_CODON_FREQ.items():
    _AA_CODONS.setdefault(aa, []).append((freq, codon))
for aa in _AA_CODONS:
    _AA_CODONS[aa].sort(reverse=True)

BEST_CODON = {aa: codons[0][1] for aa, codons in _AA_CODONS.items()}

# ── Forbidden restriction sites ───────────────────────────────────────────────
RESTRICTION_SITES = {
    'EcoRI':  'GAATTC',
    'NotI':   'GCGGCCGC',
    'XhoI':   'CTCGAG',
    'BamHI':  'GGATCC',
    'HindIII':'AAGCTT',
    'NheI':   'GCTAGC',
    'SalI':   'GTCGAC',
    'XbaI':   'TCTAGA',
    'SpeI':   'ACTAGT',
}

# ── Cryptic splice site patterns (GT donor, AG acceptor) ─────────────────────
DONOR_PATTERN   = re.compile(r'[AG]GTAAG|[AG]GTGAG|[AG]GTATG')  # GT-AG rule donors
ACCEPTOR_PATTERN= re.compile(r'[CT]{8,}NYYAG')  # simplified polypyrimidine tract

# ── Core functions ─────────────────────────────────────────────────────────────

def back_translate_optimal(protein_seq: str) -> str:
    """Translate AA sequence to optimal human CDS."""
    codons = []
    for aa in protein_seq:
        if aa == '*':
            codons.append('TGA')  # preferred stop
        elif aa in BEST_CODON:
            codons.append(BEST_CODON[aa])
        else:
            codons.append('NNN')   # unknown AA
    return ''.join(codons)


def calculate_cai(cds: str) -> float:
    """
    Codon Adaptation Index — geometric mean of relative codon frequencies.
    CAI = 1.0 means all codons are optimal. CAI < 0.7 is poor.
    """
    if len(cds) % 3 != 0:
        return 0.0
    freqs = []
    for i in range(0, len(cds)-3, 3):
        codon = cds[i:i+3].upper()
        if codon in HUMAN_CODON_FREQ:
            aa, freq = HUMAN_CODON_FREQ[codon]
            if aa != '*':  # exclude stop codons from CAI
                # Normalize to best codon for this AA
                best_freq = _AA_CODONS[aa][0][0]
                freqs.append(freq / best_freq)
    if not freqs:
        return 0.0
    import math
    return math.exp(sum(math.log(f) for f in freqs) / len(freqs))


def find_restriction_sites(cds: str) -> list[dict]:
    """Find all restriction sites in CDS."""
    hits = []
    cds_upper = cds.upper()
    for enzyme, site in RESTRICTION_SITES.items():
        pos = 0
        while True:
            idx = cds_upper.find(site, pos)
            if idx == -1:
                break
            hits.append({'enzyme': enzyme, 'site': site, 'position': idx,
                         'in_codon': idx // 3})
            pos = idx + 1
    return hits


def remove_restriction_site(cds: str, protein_seq: str, site_pos: int,
                             site_len: int) -> str:
    """
    Remove restriction site at site_pos by substituting synonymous codons
    within the affected codon range.
    """
    # Which codons overlap the restriction site?
    codon_start = (site_pos // 3)
    codon_end   = ((site_pos + site_len - 1) // 3) + 1
    cds_list = list(cds)
    rng = random.Random(site_pos)  # deterministic

    for c_idx in range(codon_start, min(codon_end, len(protein_seq))):
        aa = protein_seq[c_idx]
        if aa not in _AA_CODONS:
            continue
        # Try each synonymous codon until restriction site gone
        for freq, codon in sorted(_AA_CODONS[aa], key=lambda x: -x[0]):
            old_codon = cds[c_idx*3:c_idx*3+3]
            if codon == old_codon:
                continue
            # Apply substitution
            test_cds = cds[:c_idx*3] + codon + cds[c_idx*3+3:]
            if RESTRICTION_SITES.get(
                    next((e for e,s in RESTRICTION_SITES.items()
                          if s == cds[site_pos:site_pos+site_len]), ''), '') \
                    not in test_cds[max(0,site_pos-6):site_pos+site_len+6]:
                cds_list[c_idx*3:c_idx*3+3] = list(codon)
                break
    return ''.join(cds_list)


def remove_cryptic_donors(cds: str, protein_seq: str) -> tuple[str, int]:
    """Remove GT-AG cryptic splice donor sequences by synonymous substitution."""
    n_fixed = 0
    for match in list(DONOR_PATTERN.finditer(cds)):
        pos = match.start() + 1  # GT starts at +1
        c_idx = pos // 3
        if c_idx >= len(protein_seq):
            continue
        aa = protein_seq[c_idx]
        if aa not in _AA_CODONS:
            continue
        for freq, codon in _AA_CODONS[aa]:
            test = cds[:c_idx*3] + codon + cds[c_idx*3+3:]
            if not DONOR_PATTERN.search(test[max(0,pos-3):pos+9]):
                cds = test
                n_fixed += 1
                break
    return cds, n_fixed


def optimize(protein_seq: str, mod_name: str = '',
             remove_sites: bool = True,
             fix_splice: bool = True) -> dict:
    """
    Full codon optimization pipeline.
    Returns dict with optimized CDS and QC metrics.
    """
    # Step 1: back-translate with optimal codons
    cds = back_translate_optimal(protein_seq)
    cai_initial = calculate_cai(cds)

    sites_removed = []
    splice_fixed  = 0

    # Step 2: remove restriction sites
    if remove_sites:
        hits = find_restriction_sites(cds)
        for hit in hits:
            cds = remove_restriction_site(cds, protein_seq,
                                           hit['position'], len(hit['site']))
            sites_removed.append(hit['enzyme'])

    # Step 3: remove cryptic splice donors
    if fix_splice:
        cds, splice_fixed = remove_cryptic_donors(cds, protein_seq)

    cai_final = calculate_cai(cds)

    # Step 4: verify back-translation matches protein
    decoded = ''
    for i in range(0, len(cds)-2, 3):
        codon = cds[i:i+3]
        if codon in HUMAN_CODON_FREQ:
            aa, _ = HUMAN_CODON_FREQ[codon]
            if aa != '*':
                decoded += aa
    match = (decoded == protein_seq.replace('*',''))

    return {
        'mod_name':        mod_name,
        'protein_length':  len(protein_seq),
        'cds_length_bp':   len(cds),
        'cai_initial':     round(cai_initial, 4),
        'cai_final':       round(cai_final, 4),
        'cai_ok':          cai_final >= 0.80,
        'restriction_sites_removed': list(set(sites_removed)),
        'splice_donors_fixed': splice_fixed,
        'translation_verified': match,
        'optimized_cds':   cds,
        'first_60nt':      cds[:60],
    }


def optimize_all(seq_results: dict, verbose: bool = True) -> dict:
    """Run optimization for all sequences."""
    results = {}
    if verbose:
        print(f'\nCodon optimization for {len(seq_results)} mods...')
        print(f'  {"Mod":30} {"AA":>5} {"CAI":>6}  {"Sites":15} {"Splice":>7} {"OK":>4}')
        print('  ' + '-'*72)

    for name, seq_data in seq_results.items():
        aa_seq = seq_data.get('protein_aa', '')
        if not aa_seq:
            results[name] = {'error': 'No protein sequence'}
            continue
        r = optimize(aa_seq, mod_name=name)
        results[name] = r
        if verbose:
            sites = ','.join(r['restriction_sites_removed']) or 'none'
            ok = '✓' if r['cai_ok'] and r['translation_verified'] else '✗'
            print(f'  {name:30} {r["protein_length"]:>5} {r["cai_final"]:>6.3f}  {sites:15} '
                  f'{r["splice_donors_fixed"]:>7}  {ok}')
    return results


def export_optimized_fasta(opt_results: dict, output_dir: str) -> str:
    """Export all optimized CDS as FASTA."""
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    for name, r in opt_results.items():
        if 'error' in r or not r.get('optimized_cds'):
            continue
        cai = r['cai_final']
        ok  = 'VERIFIED' if r['translation_verified'] else 'UNVERIFIED'
        lines.append(f'>{name} | CAI={cai:.3f} | {ok} | {r["cds_length_bp"]}bp')
        cds = r['optimized_cds']
        for i in range(0, len(cds), 80):
            lines.append(cds[i:i+80])
        lines.append('')
    path = os.path.join(output_dir, 'optimized_cds.fasta')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


def export_qc_json(opt_results: dict, output_dir: str) -> str:
    """Export QC metrics as JSON (without full CDS sequences)."""
    slim = {}
    for name, r in opt_results.items():
        s = {k: v for k, v in r.items() if k != 'optimized_cds'}
        slim[name] = s
    path = os.path.join(output_dir, 'codon_optimization_qc.json')
    with open(path, 'w') as f:
        json.dump(slim, f, indent=2)
    return path


if __name__ == '__main__':
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from pipeline.sequence_fetcher import ACCESSIONS, get_sequence

    OUT = os.path.join(os.path.dirname(__file__), '..', 'output_final', 'pipeline')

    print('=== Codon Optimizer Test ===')
    seq_results = {n: get_sequence(n, verbose=False) for n in ACCESSIONS}
    opt_results = optimize_all(seq_results, verbose=True)

    # Summary
    ok     = sum(1 for r in opt_results.values() if r.get('cai_ok') and r.get('translation_verified'))
    errors = sum(1 for r in opt_results.values() if 'error' in r)
    print(f'\n  Passed QC: {ok}/{len(opt_results)-errors}  Errors: {errors}')
    print(f'  Mean CAI: {sum(r.get("cai_final",0) for r in opt_results.values())/max(1,len(opt_results)-errors):.3f}')

    fasta_path = export_optimized_fasta(opt_results, OUT)
    qc_path    = export_qc_json(opt_results, OUT)
    print(f'\n  FASTA: {fasta_path}')
    print(f'  QC:    {qc_path}')
