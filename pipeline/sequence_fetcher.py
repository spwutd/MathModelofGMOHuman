#!/usr/bin/env python3
"""
pipeline/sequence_fetcher.py
Fetches real protein + CDS sequences from NCBI E-utils API.
Falls back to synthetic placeholder when offline.
Cache: .pipeline_seq_cache.json next to this file.
"""

import os, json, time, re, hashlib
from typing import Optional

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR   = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_DIR, '.pipeline_seq_cache.json')

# ── NCBI accessions for all 29 mods ──────────────────────────────────────────
# Format: name → (protein_acc, cds_acc_or_None, expected_aa_len, organism)
ACCESSIONS = {
    # v1-v4 mods — human/whale/NMR duplications
    'TP53_human':          ('NP_000537',  'NM_000546',  393, 'Homo sapiens'),
    'ERCC1_human':         ('NP_001974',  'NM_001983',  297, 'Homo sapiens'),
    'RAD51_human':         ('NP_002866',  'NM_002875',  339, 'Homo sapiens'),
    'FEN1_human':          ('NP_004102',  'NM_004111',  380, 'Homo sapiens'),
    'ATM_human':           ('NP_000042',  'NM_000051', 3056, 'Homo sapiens'),
    'AR_human':            ('NP_000035',  'NM_000044',  919, 'Homo sapiens'),
    'AIRE_human':          ('NP_055310',  'NM_014363',  545, 'Homo sapiens'),
    'CCND1_human':         ('NP_444284',  'NM_053056',  295, 'Homo sapiens'),
    'TERT_human':          ('NP_937983',  'NM_198253', 1132, 'Homo sapiens'),
    # Foreign genes
    'PIWI_Tdohrnii':       ('XP_046451122', None,       861, 'Turritopsis dohrnii'),
    'LAMP2A_NMR':          ('XP_004840812', None,       424, 'Heterocephalus glaber'),
    'GLO1_NMR':            ('XP_004840812', None,       184, 'Heterocephalus glaber'),  # approx
    'ADAR_Cephalopod':     ('XP_014787312', None,      1071, 'Octopus bimaculoides'),
    'Myotis_MITO_CI_ND5':  ('YP_003398498', None,       538, 'Myotis lucifugus'),
    'LIF6_elephant':       ('XP_023410761', None,       212, 'Loxodonta africana'),
    'HAS2_NMR':            ('XP_021082893', None,       552, 'Heterocephalus glaber'),
    'CD44_NMR':            ('XP_004839000', None,       742, 'Heterocephalus glaber'),
    'FOXO3_Hydra':         ('XP_012557498', None,       568, 'Hydra vulgaris'),
    'GATA4_zebrafish':     ('NP_571471',    None,       441, 'Danio rerio'),
    'HAND2_zebrafish':     ('NP_571483',    None,       217, 'Danio rerio'),
    'NRF2_NMR':            ('XP_004889397', None,       614, 'Heterocephalus glaber'),
    'TBX5_zebrafish':      ('NP_571501',    None,       518, 'Danio rerio'),
    'RELA_shark':          ('XP_041052389', None,       551, 'Somniosus microcephalus'),
    'TFEB_human':          ('NP_006606',    'NM_006702', 476, 'Homo sapiens'),
    'NEURO_REGEN_FGF8b':   ('NP_571519',    None,       233, 'Danio rerio'),
    # Synthetic — no NCBI accession, will generate placeholder
    'SENOLYSIN_circuit':   (None, None,  198, 'SYNTHETIC'),
    'OSKM_cyclic':         (None, None, 1204, 'SYNTHETIC'),
    'GLUCOSPANASE_bact':   ('WP_003232589', None, 312,  'Bacillus subtilis'),
    'DDCBE_mito':          (None, None, 1083, 'SYNTHETIC'),
    'LIPOFUSCINASE':       (None, None,  447, 'SYNTHETIC'),
    'NEURO_OSKM_SK':       (None, None,  989, 'SYNTHETIC'),
    'MITOSOD':             (None, None,  387, 'SYNTHETIC'),
    'INFLAMMABREAK':       (None, None,  312, 'SYNTHETIC'),
}

# ── Codon table (human optimized) ─────────────────────────────────────────────
AA_TO_BEST_CODON = {
    'A':'GCC','R':'CGG','N':'AAC','D':'GAC','C':'TGC','Q':'CAG','E':'GAG',
    'G':'GGC','H':'CAC','I':'ATC','L':'CTG','K':'AAG','M':'ATG','F':'TTC',
    'P':'CCC','S':'AGC','T':'ACC','W':'TGG','Y':'TAC','V':'GTG','*':'TGA',
}

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {}

def _load_cache():
    global _cache
    if os.path.exists(_CACHE):
        try:
            with open(_CACHE) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}

def _save_cache():
    try:
        with open(_CACHE, 'w') as f:
            json.dump(_cache, f, indent=2)
    except Exception:
        pass

_load_cache()

# ── NCBI fetch ─────────────────────────────────────────────────────────────────
def _ncbi_fetch(accession: str, db: str = 'protein') -> Optional[str]:
    """Fetch FASTA from NCBI E-utils. Returns sequence string or None."""
    try:
        import requests
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        params = {
            'db': db,
            'id': accession,
            'rettype': 'fasta',
            'retmode': 'text',
            'tool': 'HomoPerpetuu',
            'email': 'pipeline@homoperpetuus.sim',
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200 and r.text.startswith('>'):
            # Parse FASTA
            lines = r.text.strip().split('\n')
            seq = ''.join(l.strip() for l in lines[1:] if not l.startswith('>'))
            return seq
    except Exception:
        pass
    return None

def _ncbi_fetch_cds(accession: str) -> Optional[str]:
    """Fetch nucleotide CDS from NCBI."""
    return _ncbi_fetch(accession, db='nuccore')

# ── Synthetic sequence generation ─────────────────────────────────────────────
# Human proteome amino acid frequencies (Uniprot 2023)
_AA_FREQ = list('ACDEFGHIKLMNPQRSTVWY')

def _synthetic_protein(length: int, name: str) -> str:
    """Generate deterministic synthetic AA sequence as placeholder."""
    import random
    rng = random.Random(hashlib.md5(name.encode()).hexdigest())
    # Bias toward common human protein composition
    weights = [7,2,5,5,4,7,2,6,5,9,2,4,5,4,6,7,6,7,3,3]  # approx ACDEFGHIKLMNPQRSTVWY
    seq = ''
    for _ in range(length):
        seq += rng.choices(_AA_FREQ, weights=weights)[0]
    return 'M' + seq[1:]  # always start with Met

def _back_translate(protein_seq: str) -> str:
    """Back-translate AA → human-optimized codon DNA."""
    codons = [AA_TO_BEST_CODON.get(aa, 'NNN') for aa in protein_seq]
    return ''.join(codons)

# ── Public API ─────────────────────────────────────────────────────────────────
def get_sequence(mod_name: str, verbose: bool = True) -> dict:
    """
    Fetch or generate sequence data for a modification.
    Returns dict with keys: name, protein_aa, cds_nt, length_aa, length_bp,
                             source, accession, online, from_cache
    """
    if mod_name in _cache:
        entry = _cache[mod_name]
        entry['from_cache'] = True
        return entry

    if mod_name not in ACCESSIONS:
        # Unknown mod — generate minimal placeholder
        return {
            'name': mod_name, 'protein_aa': '', 'cds_nt': '',
            'length_aa': 0, 'length_bp': 0, 'source': 'UNKNOWN',
            'accession': None, 'online': False, 'from_cache': False,
            'error': f'No accession defined for {mod_name}',
        }

    prot_acc, cds_acc, expected_len, organism = ACCESSIONS[mod_name]
    result = {
        'name': mod_name,
        'source': organism,
        'accession': prot_acc,
        'expected_aa': expected_len,
        'from_cache': False,
        'online': False,
    }

    if prot_acc is None:
        # Synthetic — generate deterministic placeholder
        aa_seq  = _synthetic_protein(expected_len, mod_name)
        cds     = _back_translate(aa_seq)
        result.update({
            'protein_aa': aa_seq,
            'cds_nt': cds,
            'length_aa': len(aa_seq),
            'length_bp': len(cds),
            'note': 'SYNTHETIC placeholder — replace with designed sequence',
        })
        if verbose:
            print(f'  [SEQ] {mod_name:30} → SYNTHETIC ({expected_len} aa, {len(cds)} bp)')
        _cache[mod_name] = result
        _save_cache()
        return result

    # Try NCBI online
    if verbose:
        print(f'  [SEQ] {mod_name:30} → fetching {prot_acc} ... ', end='', flush=True)

    aa_seq = _ncbi_fetch(prot_acc, db='protein')
    if aa_seq:
        result['online'] = True
        result['protein_aa'] = aa_seq
        result['length_aa'] = len(aa_seq)
        # Try CDS
        if cds_acc:
            cds = _ncbi_fetch_cds(cds_acc)
            if cds:
                result['cds_nt'] = cds
                result['length_bp'] = len(cds)
            else:
                result['cds_nt'] = _back_translate(aa_seq)
                result['length_bp'] = len(result['cds_nt'])
                result['note'] = 'CDS back-translated (human codon optimized)'
        else:
            result['cds_nt'] = _back_translate(aa_seq)
            result['length_bp'] = len(result['cds_nt'])
            result['note'] = 'CDS back-translated (human codon optimized)'
        if verbose:
            print(f'OK  ({len(aa_seq)} aa, {result["length_bp"]} bp)')
        _save_cache()
    else:
        # Offline fallback — synthetic with correct length
        aa_seq = _synthetic_protein(expected_len, mod_name)
        cds    = _back_translate(aa_seq)
        result.update({
            'protein_aa': aa_seq,
            'cds_nt': cds,
            'length_aa': len(aa_seq),
            'length_bp': len(cds),
            'note': f'OFFLINE placeholder (expected {expected_len} aa). '
                    f'Replace with real {prot_acc} sequence when online.',
        })
        if verbose:
            print(f'OFFLINE (placeholder {expected_len} aa)')

    _cache[mod_name] = result
    _save_cache()
    return result


def fetch_all(verbose: bool = True) -> dict:
    """Fetch sequences for all 29 mods. Returns {name: seq_dict}."""
    results = {}
    if verbose:
        print(f'\nFetching sequences for {len(ACCESSIONS)} mods...')
    for name in ACCESSIONS:
        results[name] = get_sequence(name, verbose=verbose)
        time.sleep(0.35)  # NCBI rate limit: 3 req/sec
    if verbose:
        online  = sum(1 for v in results.values() if v.get('online'))
        synth   = sum(1 for v in results.values() if v.get('source') == 'SYNTHETIC')
        offline = len(results) - online - synth
        print(f'\n  Online: {online}  Synthetic: {synth}  Offline placeholder: {offline}')
    return results


def clear_cache():
    global _cache
    _cache = {}
    if os.path.exists(_CACHE):
        os.remove(_CACHE)
    print('Cache cleared.')


if __name__ == '__main__':
    print('=== Sequence Fetcher Test ===')
    # Test single
    result = get_sequence('TFEB_human', verbose=True)
    print(f'  AA[:20]: {result["protein_aa"][:20]}...')
    print(f'  CDS[:30]: {result["cds_nt"][:30]}...')
    print()
    # Test synthetic
    result2 = get_sequence('SENOLYSIN_circuit', verbose=True)
    print(f'  AA[:20]: {result2["protein_aa"][:20]}...')
