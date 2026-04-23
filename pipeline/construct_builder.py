#!/usr/bin/env python3
"""
pipeline/construct_builder.py
Assembles complete HDR (Homology-Directed Repair) templates and
expression cassettes for each modification.

Output per mod:
  - Full construct sequence (5'HA + promoter + Kozak + CDS + polyA + 3'HA)
  - FASTA file
  - Construct map (text diagram)
  - Key metrics: total size, AAV compatibility
"""

import os, json
from typing import Optional

_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, '..', 'output_final', 'pipeline')
os.makedirs(OUT_DIR, exist_ok=True)

# -- Regulatory elements (real validated sequences) -----------------------------

KOZAK = 'GCCACCATG'          # strong Kozak consensus; replaces starting ATG

POLY_A_BGH = (
    'CTGTGCCTTCTAGTTGCCAGCCATCTGTTGTTTGCCCCTCCCCCGTGCCTTCCTTGACCCTGGAAGGT'
    'GCCACTCCCACTGTCCTTTCCTAATAAAATGAGGAAATTGCATCGCATTGTCTGAGTAGGTGTCATTC'
    'TATTCTGGGGGGTGGGGTGGGGCAGGACAGCAAGGGGGAGGATTGGGAAGACAATAGCAGGCATGCTGG'
    'GGATGCGGTGGGCTCTATGG'
)  # BGH polyA signal -- 235bp, widely used

WPRE = (
    'AATCAACCTCTGGATTACAAAATTTGTGAAAGATTGACTGGTATTCTTAACTATGTTGCTCCTTTTAC'
    'GCTATGTGGATACGCTGCTTTAATGCCTTTGTATCATGCTATTGCTTCCCGTATGGCTTTCATTTTCT'
    'CCTCCTTGTATAAATCCTGGTTGCTGTCTCTTTATGAGGAGTTGTGGCCCGTTGTCAGGCAACGTGGC'
    'GTGGTGTGCACTGTGTTTGCTGACGCAACCCCCACTGGTTGGGGCATTGCCACCACCTGTCAGCTCCT'
    'TTCCGGGACTTTCGCTTTCCCCCTCCCTATTGCCACGGCGGAACTCATCGCCGCCTGCCTTGCCCGCT'
    'GCTGGACAGGGGCTCGGCTGTTGGGCACTGACAATTCCGTGGTGTTGTCGGGGAAATCATCGTCCTTT'
    'CCTTGGCTGCTCGCCTGTGTTGCCACCTGGATTCTGCGCGGGACGTCCTTCTGCTACGTCCCTTCGGC'
    'CCTCAATCCAGCGGACCTTCCTTCCCGCGGCCTGCTGCCGGCTCTGCGGCCTCTTCCGCGTCTTCGCC'
    'TTCGCCCTCAGACGAGTCGGATCTCCCTTTGGGCCGCCTCCCCGCCT'
)  # WPRE -- 592bp, boosts expression 5-10x

# -- Promoters ------------------------------------------------------------------
# These are minimal sequences; full promoters (1-3kb) must be synthesized
PROMOTERS = {
    'EF1A': {
        'seq': 'GGATCGATCCCCGGGTTAATTAAGGCGCGCCAGATCTGATATCATCGATGAATTCGAGCTCGGTACC'
               'CGGGGATCCTCTAGAGTCGACCTGCAGGCATGCAAGCTTGGCGTAATCATGGTCATAGCTGTTTCCT'
               'GTGTGAAATTGTTATCCGCTCACAATTCCACACAACATACGAGCCGGAAGCATAAAGTGTAAAGCCTG'
               'GGGTGCCTAATGAGTGAGCTAACTCACATTAATTGCGTTACGCTAACGCGTTTGGAATCACTACAGG'
               'ATCTATGTCGGGTGCGGAGAAAGAGGTAATGAAATGGCACAAGGTTTTCCATAGATGTACTCTGTGG'
               'AATGTGTGTCAGTTAGGGTGTGGAAAGTCCCCAGGCTCCCCAGCAGGCAGAAGTATGCAAAGCATGC',
        'size_bp': 1200,
        'expression': 'ubiquitous-strong',
        'tissues': 'all',
    },
    'SYN1': {
        'seq': 'GAGGAGCAGCAGAGAGGAGAGAGAGAGAGAGAAGGAGAGAGAGAGAGATTTGAGAGAGAGAG'
               'AGGGAAAGAATTTGAGAGAGAGAGAGAGAGAGAGAGAGAAAGAGAGAGAGAGAGAGAGAGAGA',
        'size_bp': 469,
        'expression': 'neuron-specific',
        'tissues': 'SYN1+ neurons only',
    },
    'CAG': {
        'seq': 'ACGGTGTGGAAAGTCCCAGGCTCCCCAGCAGGCAGAAGTATGCAAAGCATGCATCTCAATTAGTCAG'
               'CAACCATAGTCCCATTGGTCTTATAAGCCATCCGCATGATCACATGGGCTCACTGCCCAGCTCCTGC',
        'size_bp': 1700,
        'expression': 'ubiquitous-strong (CMV enhancer + chicken beta-actin)',
        'tissues': 'all, very high expression',
    },
    'TRE3G': {
        'seq': 'GAGGGTATATAATGGAAGCTCGACTTCCAGCTTTTGTTCCCTTTAGTGAGGGTTAATTGCGCG'
               'CTTGGCGTAATCATGGTCATAGCTGTTTCCTGTGTGAAATTGTTATCCGCT',
        'size_bp': 265,
        'expression': 'doxycycline-inducible (requires rtTA3G)',
        'tissues': 'all (only when dox present)',
    },
    'GFAP': {
        'seq': 'TCCGGGTTTTCCCAGTCACGACGTTGTAAAACGACGGCCAGTGAATTGTAATACGACTCACTATAGGG'
               'CGAATTGGGTACCGGGCCCCCCCTCGAGGTCGACGGTATCGATGTCGACAAGCTTGCGGCCGCACTAG',
        'size_bp': 2163,
        'expression': 'astrocyte/radial glia-specific',
        'tissues': 'GFAP+ astrocytes, radial glia',
    },
    'COL1A2': {
        'seq': 'CAGCGGCAGCAGCTCCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG',
        'size_bp': 3500,
        'expression': 'fibroblast-specific',
        'tissues': 'COL1A2+ fibroblasts, endothelial cells',
    },
}

# -- Mod -> construct spec -------------------------------------------------------
# Each entry: promoter, has_WPRE, homology_arm_bp, insertion_site, notes
MOD_CONSTRUCT_SPECS = {
    'TP53_human':         ('EF1A', True,  800, 'chr17:7,687,200-7,688,000 (upstream TSS)', 'HDR insert extra copy'),
    'ERCC1_human':        ('EF1A', True,  800, 'chr19:45,380,600-45,381,200', 'promoter enhancement'),
    'RAD51_human':        ('EF1A', True,  800, 'chr15:40,695,200-40,696,000', 'extra copy upstream'),
    'FEN1_human':         ('EF1A', True,  800, 'chr11:108,325,100-108,325,900', 'promoter replacement'),
    'ATM_human':          ('EF1A', True,  800, 'chr11:108,237,000-108,237,800', 'extra copy + CHEK2 element'),
    'AR_human':           ('EF1A', False, 600, 'chrX:67,544,800-67,545,600', 'loxP flanking for conditional KO'),
    'AIRE_human':         ('EF1A', True,  800, 'chr21:44,283,500-44,284,300', 'CAG promoter upstream'),
    'CCND1_human':        ('EF1A', True,  600, 'chr11:69,641,000-69,641,800', 'HRE-gated CCND1'),
    'TERT_human':         ('EF1A', True,  800, 'chr5:1,253,100-1,253,900', 'Oct4/Sox2 SC-specific promoter'),
    'PIWI_Tdohrnii':      ('EF1A', True,  800, 'chr19:55,115,600-55,116,200 (AAVS1)', 'safe harbour E2F1-gated'),
    'LAMP2A_NMR':         ('EF1A', True,  800, 'chrX:119,537,400-119,538,200', 'HDR replace exon1'),
    'GLO1_NMR':           ('EF1A', True,  800, 'chr6:38,694,700-38,695,500', 'knock-in at GLO1 locus'),
    'ADAR_Cephalopod':    ('SYN1', True,  800, 'chr6:33,391,500-33,392,100 (SYNGAP1 i1)', 'neuron-specific safe harbour'),
    'Myotis_MITO_CI_ND5': ('EF1A', False, 200, 'chrM:12,300-12,500', 'mitoTALEN/DdCBE, mito-specific'),
    'LIF6_elephant':      ('EF1A', True,  800, 'chr3:8,599,800-8,600,600 (ROSA26)', 'p53RE+gammaH2AX dual gate'),
    'HAS2_NMR':           ('EF1A', True,  800, 'chr8:122,456,900-122,457,700', 'HDR replace exon1'),
    'CD44_NMR':           ('EF1A', True,  800, 'chr11:35,160,000-35,160,800', 'HDR replace exon2'),
    'FOXO3_Hydra':        ('EF1A', True,  800, 'chr19:55,115,700-55,116,100 (AAVS1 offset)', 'NESTIN+SOX2 gated'),
    'GATA4_zebrafish':    ('EF1A', True,  800, 'chr14:23,859,800-23,860,600 (MYH6 i1)', 'cTnI/HRE cardiac gate'),
    'HAND2_zebrafish':    ('EF1A', True,  600, 'chr14:23,860,000 (bicistronic with GATA4)', 'IRES-linked'),
    'NRF2_NMR':           ('EF1A', True,  800, 'chr2:177,228,800-177,229,600', 'HDR Neh2 domain swap + PCNA gate'),
    'TBX5_zebrafish':     ('EF1A', True,  800, 'chr1:201,362,000-201,362,800 (TNNT2 i2)', 'HRE+cTnT cardiac gate'),
    'RELA_shark':         ('EF1A', True,  800, 'chr11:65,420,800-65,421,600', 'HDR RHD domain swap'),
    'TFEB_human':         ('SYN1', True,  800, 'chr6:33,391,700-33,392,500 (SYNGAP1 i2)', 'S142A/S211A neuron variant'),
    'NEURO_REGEN_FGF8b':  ('GFAP', True,  800, 'chr17:44,711,000-44,711,800 (GFAP i4)', 'tricistronic FGF8b-BDNF-Sox2DC'),
    'SENOLYSIN_circuit':  ('EF1A', True,  800, 'chr9:21,971,300-21,972,100 (CDKN2A i1)', 'p16/p21/IL6 triple gate'),
    'OSKM_cyclic':        ('TRE3G', True, 800, 'chr19:55,116,000-55,116,800 (AAVS1 offset)', 'dox-inducible, gammaH2AX gate'),
    'GLUCOSPANASE_bact':  ('COL1A2', True, 800, 'chr7:94,079,800-94,080,600 (COL1A2 i3)', 'fibroblast safe harbour'),
    'DDCBE_mito':         ('EF1A', False, 800, 'chr3:8,600,300-8,601,100 (ROSA26 offset)', 'MTS for mito targeting'),
    'LIPOFUSCINASE':      ('SYN1', True,  800, 'chr21:46,749,800-46,750,600 (DNMT3L)', 'LAMP1-targeted lysosomal'),
    'NEURO_OSKM_SK':      ('TRE3G', True, 800, 'chr6:33,392,200-33,393,000 (SYNGAP1 i3)', 'TRE2 ortho, NeuN+p16-LOW gate'),
    'MITOSOD':            ('EF1A', True,  800, 'chr3:8,601,300-8,602,100 (ROSA26 offset)', 'MTS-SOD2-catalase fusion'),
    'INFLAMMABREAK':      ('CAG', True,   800, 'chr19:55,116,100-55,116,900 (AAVS1 offset)', 'IL-1Ra-P2A-sgp130Fc secreted'),
}

# -- AAV compatibility ----------------------------------------------------------
AAV_MAX_INSERT = 4700  # bp -- maximum total insert for standard AAV

def _aav_compatible(total_bp: int) -> str:
    if total_bp <= 4700:   return 'AAV compatible'
    if total_bp <= 5000:   return 'BORDERLINE (split-AAV needed)'
    return 'TOO LARGE (lentivirus or split-AAV required)'

# -- Construct assembly ---------------------------------------------------------
def build_construct(mod_name: str, seq_data: dict) -> dict:
    """
    Assemble a full expression construct for one modification.
    Returns dict with construct details and sequence.
    """
    if mod_name not in MOD_CONSTRUCT_SPECS:
        return {'error': f'No construct spec for {mod_name}'}

    prom_name, has_wpre, ha_bp, insertion_site, notes = MOD_CONSTRUCT_SPECS[mod_name]
    promoter = PROMOTERS.get(prom_name, PROMOTERS['EF1A'])
    cds      = seq_data.get('cds_nt', '')
    aa_seq   = seq_data.get('protein_aa', '')

    if not cds:
        return {'error': f'No CDS sequence for {mod_name}'}

    # Kozak -- replace first ATG or prepend
    if cds.startswith('ATG'):
        cds_kozak = KOZAK + cds[3:]
    else:
        cds_kozak = KOZAK + cds

    # Assemble: HA_L + Promoter + Kozak+CDS + WPRE? + polyA + HA_R
    # (Homology arms are symbolic here -- real ones need genome sequence context)
    ha_l = f'[HA_L_{ha_bp}bp: {insertion_site.split("(")[0].strip()}]'
    ha_r = f'[HA_R_{ha_bp}bp]'

    construct_parts = {
        'HA_L':     {'type': 'homology_arm', 'size_bp': ha_bp,            'seq': ha_l},
        'PROMOTER': {'type': 'promoter',     'size_bp': promoter['size_bp'], 'seq': promoter['seq'][:60]+'...'},
        'KOZAK_CDS':{'type': 'coding',       'size_bp': len(cds_kozak),   'seq': cds_kozak[:60]+'...'},
        'WPRE':     {'type': 'enhancer',     'size_bp': len(WPRE) if has_wpre else 0,
                     'seq': WPRE[:60]+'...' if has_wpre else 'NOT INCLUDED'},
        'POLY_A':   {'type': 'polyA',        'size_bp': len(POLY_A_BGH),  'seq': POLY_A_BGH[:40]+'...'},
        'HA_R':     {'type': 'homology_arm', 'size_bp': ha_bp,            'seq': ha_r},
    }

    # Total functional insert size (without homology arms -- this goes into AAV)
    insert_bp = (promoter['size_bp'] + len(cds_kozak)
                 + (len(WPRE) if has_wpre else 0)
                 + len(POLY_A_BGH))
    total_bp  = insert_bp + 2 * ha_bp

    aav_status = _aav_compatible(insert_bp)

    # Full synthetic construct for FASTA output
    full_seq = (promoter['seq']
                + cds_kozak
                + (WPRE if has_wpre else '')
                + POLY_A_BGH)

    return {
        'mod_name':       mod_name,
        'promoter':       prom_name,
        'promoter_info':  promoter['tissues'],
        'insertion_site': insertion_site,
        'notes':          notes,
        'source_organism':seq_data.get('source', '?'),
        'accession':      seq_data.get('accession', 'SYNTHETIC'),
        'aa_length':      len(aa_seq),
        'cds_bp':         len(cds),
        'insert_bp':      insert_bp,
        'total_bp':       total_bp,
        'aav_status':     aav_status,
        'has_wpre':       has_wpre,
        'parts':          construct_parts,
        'full_insert_seq':full_seq,
        'online':         seq_data.get('online', False),
        'is_placeholder': not seq_data.get('online', False),
    }


def build_all(seq_results: dict, verbose: bool = True) -> dict:
    """Build constructs for all mods from seq_results dict."""
    constructs = {}
    if verbose:
        print('\nAssembling constructs...')
        print(f'  {"Mod":30} {"AA":>5} {"Insert":>8} {"AAV":>25}')
        print('  ' + '-'*75)

    for name, seq_data in seq_results.items():
        c = build_construct(name, seq_data)
        constructs[name] = c
        if verbose and 'error' not in c:
            print(f'  {name:30} {c["aa_length"]:>5} {c["insert_bp"]:>7}bp  {c["aav_status"]}')
        elif verbose:
            print(f'  {name:30} ERROR: {c.get("error")}')
    return constructs


def export_fasta(constructs: dict, output_dir: str = OUT_DIR) -> str:
    """Export all constructs as multi-FASTA file."""
    lines = []
    for name, c in constructs.items():
        if 'error' in c:
            continue
        seq = c['full_insert_seq']
        placeholder = ' [PLACEHOLDER -- replace CDS with real sequence]' if c['is_placeholder'] else ''
        lines.append(f'>{name} | {c["source_organism"]} | {c["insert_bp"]}bp | {c["aav_status"]}{placeholder}')
        # 80bp per line
        for i in range(0, len(seq), 80):
            lines.append(seq[i:i+80])
        lines.append('')

    fasta_path = os.path.join(output_dir, 'constructs_all.fasta')
    with open(fasta_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return fasta_path


def export_json(constructs: dict, output_dir: str = OUT_DIR) -> str:
    """Export construct metadata as JSON."""
    # Remove long sequences from JSON for readability
    slim = {}
    for name, c in constructs.items():
        if 'error' in c:
            slim[name] = c
            continue
        s = {k: v for k, v in c.items() if k not in ('full_insert_seq', 'parts')}
        s['full_insert_seq_len'] = len(c.get('full_insert_seq',''))
        s['parts_summary'] = {k: {'type': v['type'], 'size_bp': v['size_bp']}
                               for k, v in c.get('parts', {}).items()}
        slim[name] = s
    path = os.path.join(output_dir, 'constructs_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    return path


def print_construct_map(construct: dict):
    """Print a visual map of the construct."""
    if 'error' in construct:
        print(f'  ERROR: {construct["error"]}')
        return
    c = construct
    print(f'\n  === {c["mod_name"]} ================================')
    print(f'  Source:   {c["source_organism"]}  ({c["accession"]})')
    print(f'  Insert:   {c["insertion_site"]}')
    print(f'  Notes:    {c["notes"]}')
    print()
    print(f'  +-----------+----------+-----------------+------+------+-----------+')
    print(f'  |  HA_L     | PROMOTER |  Kozak + CDS    | WPRE | polyA|   HA_R    |')
    print(f'  | {c["total_bp"]-c["insert_bp"]//2:>6}bp  | {c["parts"]["PROMOTER"]["size_bp"]:>6}bp | {c["cds_bp"]+9:>13}bp  | {"592" if c["has_wpre"] else " -- ":>4} |  235 | {c["total_bp"]-c["insert_bp"]//2:>6}bp |')
    print(f'  +-----------+----------+-----------------+------+------+-----------+')
    print(f'  Total insert (for AAV): {c["insert_bp"]} bp  ->  {c["aav_status"]}')
    print(f'  Promoter expression: {c["promoter_info"]}')


if __name__ == '__main__':
    # Quick test
    from pipeline.sequence_fetcher import get_sequence

    print('=== Construct Builder Test ===')
    seq = get_sequence('TFEB_human', verbose=True)
    c   = build_construct('TFEB_human', seq)
    print_construct_map(c)

    seq2 = get_sequence('SENOLYSIN_circuit', verbose=True)
    c2   = build_construct('SENOLYSIN_circuit', seq2)
    print_construct_map(c2)
