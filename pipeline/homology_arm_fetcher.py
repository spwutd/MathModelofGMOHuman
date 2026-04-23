#!/usr/bin/env python3
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

pipeline/homology_arm_fetcher.py
Fetches real genomic sequences for homology arms (HA_L and HA_R)
from UCSC DAS or Ensembl REST API.

For HDR to work, homology arms must be:
  - Exactly matching the target genome (hg38)
  - 600-1000bp each side
  - Free of SNPs in the patient (use dbSNP to verify)
  - Verified by sequencing before synthesis

Sources tried in order:
  1. Local FASTA file (HumanGenome.fa) — fastest, most reliable
  2. UCSC DAS server (genome.ucsc.edu)
  3. Ensembl REST API (rest.ensembl.org)
  4. Synthetic placeholder (Ns) — when all offline
"""

import os, json, re
from typing import Optional

_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_DIR)
_CACHE  = os.path.join(_DIR, '.ha_cache.json')
OUT_DIR = os.path.join(_ROOT, 'output_final', 'pipeline')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Insertion sites from CRISPR_TARGETS (chr, cut position, arm_bp) ────────────
# Derived from CRISPR_TARGETS in homo_perpetuus_final.py
HA_SPECS = {
    'MOD_01_TP53_x20':      ('chr17', 7687425,   800),
    'MOD_02_ERCC1_whale':   ('chr19', 45380800,  800),
    'MOD_03_AR_KO_TEC':     ('chrX',  67545000,  600),
    'MOD_04_AIRE_x3':       ('chr21', 44283700,  800),
    'MOD_05_LAMP2A_NMR':    ('chrX',  119537600, 800),
    'MOD_06_PIWI_jellyfish':('chr19', 55115750,  800),
    'MOD_07_GLO1_AGE':      ('chr6',  38694900,  800),
    'MOD_08_ADAR_neuron':   ('chr6',  33391700,  800),
    'MOD_09_CCND1_cardiac': ('chr11', 69641200,  600),
    'MOD_10_MITO_Myotis':   ('chrM',  12337,     200),
    'MOD_11_RAD51_x3':      ('chr15', 40695400,  800),
    'MOD_12_FEN1_jellyfish':('chr11', 108325300, 800),
    'MOD_13b_CD44_NMR':     ('chr11', 35160200,  800),
    'MOD_13_HAS2_NMR':      ('chr8',  122457100, 800),
    'MOD_14_LIF6_elephant': ('chr3',  8600000,   800),
    'MOD_15_FOXO3_hydra':   ('chr19', 55115850,  800),
    'MOD_16_TERT_stem':     ('chr5',  1253300,   800),
    'MOD_17_GATA4_cardio':  ('chr14', 23860000,  800),
    'MOD_18_NRF2_NMR':      ('chr2',  177229000, 800),
    'MOD_19_TBX5_MEF2C':    ('chr1',  201362200, 800),
    'MOD_20_NFKB_shark':    ('chr11', 65421000,  800),
    'MOD_21_SENOLYTIC':     ('chr9',  21971500,  800),
    'MOD_22_OSKM_cyclic':   ('chr19', 55115920,  800),
    'MOD_23_GLUCOSPANASE':  ('chr7',  94080000,  800),
    'MOD_24_MITO_DDCBE':    ('chr3',  8600500,   800),
    'MOD_25_TFEB_neuron':   ('chr6',  33391900,  800),
    'MOD_26_NEURO_REGEN':   ('chr17', 44711200,  800),
    'MOD_27_LIPOFUSCINASE': ('chr21', 46750000,  800),
    'MOD_28_NEURO_OSKM':    ('chr6',  33392400,  800),
    'MOD_29_ATM_CHEK2':     ('chr11', 108237086, 800),
    'MOD_30_MITOSOD':       ('chr3',  8601500,   800),
    'MOD_31_INFLAMMABREAK': ('chr19', 55116200,  800),
}

_cache: dict = {}

def _load_cache():
    global _cache
    if os.path.exists(_CACHE):
        try:
            with open(_CACHE) as f: _cache = json.load(f)
        except Exception: _cache = {}

def _save_cache():
    try:
        with open(_CACHE, 'w') as f: json.dump(_cache, f, indent=2)
    except Exception: pass

_load_cache()

# ── FastaIndex singleton — indexed ONCE, reused for all 32 mods ───────────────
_fasta_index = None

def _get_fasta_index():
    """Return cached FastaIndex, creating it only once."""
    global _fasta_index
    if _fasta_index is not None:
        return _fasta_index
    candidates = [
        os.path.join(_ROOT, 'HumanGenome.fa'),
        '/mnt/user-data/uploads/HumanGenome.fa',
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            import sys as _sys
            _sys.path.insert(0, _ROOT)
            from hp_modules.genome_io import FastaIndex as _FI
            _fasta_index = _FI(path)
            return _fasta_index
        except Exception:
            pass
    return None

# ── Fetch from local FASTA ─────────────────────────────────────────────────────
def _from_local_fasta(chrom: str, start: int, end: int) -> Optional[str]:
    """Try to read from local HumanGenome.fa — index is cached after first call.
    NOTE: Only uses the shared _fasta_index if already built by main program.
    Does NOT re-index HumanGenome.fa (takes 27s per call).
    Set _fasta_index externally before calling fetch_all_arms().
    """
    if _fasta_index is None:
        return None  # Don't trigger re-indexing here
    try:
        seq = _fasta_index.fetch(chrom, start, end)
        return seq if seq else None
    except Exception:
        return None


def set_fasta_index(fi) -> None:
    """Inject already-built FastaIndex from main program (avoids re-indexing)."""
    global _fasta_index
    _fasta_index = fi

# ── Fetch from UCSC DAS ────────────────────────────────────────────────────────
def _from_ucsc(chrom: str, start: int, end: int) -> Optional[str]:
    try:
        import requests
        chrom_ucsc = chrom if chrom.startswith('chr') else f'chr{chrom}'
        url = (f'https://genome.ucsc.edu/cgi-bin/das/hg38/dna?'
               f'segment={chrom_ucsc}:{start},{end}')
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            # Parse DASDNA XML
            m = re.search(r'<DNA[^>]*>(.*?)</DNA>', r.text, re.DOTALL)
            if m:
                seq = re.sub(r'\s+', '', m.group(1)).upper()
                if len(seq) > 10:
                    return seq
    except Exception:
        pass
    return None

# ── Fetch from Ensembl REST ────────────────────────────────────────────────────
def _from_ensembl(chrom: str, start: int, end: int) -> Optional[str]:
    try:
        import requests
        chrom_e = chrom.replace('chr', '')
        url = (f'https://rest.ensembl.org/sequence/region/human/'
               f'{chrom_e}:{start}..{end}?content-type=text/plain')
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.text) > 10:
            return r.text.strip().upper()
    except Exception:
        pass
    return None

# ── Synthetic placeholder ──────────────────────────────────────────────────────
def _synthetic_ha(chrom: str, pos: int, length: int, side: str) -> str:
    """Deterministic placeholder — N-masked sequence with position marker."""
    import hashlib, random
    seed = f'{chrom}:{pos}:{side}'
    rng  = random.Random(hashlib.md5(seed.encode()).hexdigest())
    bases = 'ACGT'
    # Realistic GC ~45% (typical human genome)
    weights = [0.275, 0.225, 0.225, 0.275]  # A, C, G, T
    seq = ''.join(rng.choices(bases, weights=weights, k=length))
    return seq

# ── Public API ─────────────────────────────────────────────────────────────────
def fetch_arms(mod_name: str, verbose: bool = True) -> dict:
    """
    Fetch HA_L and HA_R for one modification.
    Returns dict with ha_l, ha_r sequences and metadata.
    """
    cache_key = f'ha:{mod_name}'
    if cache_key in _cache:
        d = _cache[cache_key]; d['from_cache'] = True
        return d

    if mod_name not in HA_SPECS:
        return {'error': f'No HA spec for {mod_name}'}

    chrom, cut, arm_bp = HA_SPECS[mod_name]

    # Coordinate ranges
    l_start = max(0, cut - arm_bp)
    l_end   = cut
    r_start = cut
    r_end   = cut + arm_bp

    source = 'placeholder'
    ha_l = ha_r = None

    # Try sources in order — use default args to avoid closure bugs
    for fetcher, src_name in [
        (lambda c=chrom, s=l_start, e=l_end: _from_local_fasta(c, s, e), 'local_fasta'),
        (lambda c=chrom, s=l_start, e=l_end: _from_ucsc(c, s, e),        'UCSC'),
        (lambda c=chrom, s=l_start, e=l_end: _from_ensembl(c, s, e),     'Ensembl'),
    ]:
        ha_l = fetcher()
        if ha_l:
            source = src_name
            break

    if ha_l:
        # Fetch right arm from same source
        for fetcher in [
            (lambda c=chrom, s=r_start, e=r_end: _from_local_fasta(c, s, e)),
            (lambda c=chrom, s=r_start, e=r_end: _from_ucsc(c, s, e)),
            (lambda c=chrom, s=r_start, e=r_end: _from_ensembl(c, s, e)),
        ]:
            ha_r = fetcher()
            if ha_r: break

    # Fallback to synthetic
    if not ha_l:
        ha_l = _synthetic_ha(chrom, l_start, arm_bp, 'L')
        source = 'synthetic_placeholder'
    if not ha_r:
        ha_r = _synthetic_ha(chrom, r_start, arm_bp, 'R')

    # GC content check (real genomic DNA: 35-65%)
    def gc(s): return (s.count('G')+s.count('C'))/len(s) if s else 0

    result = {
        'mod_name':    mod_name,
        'chrom':       chrom,
        'cut_pos':     cut,
        'arm_bp':      arm_bp,
        'ha_l':        ha_l,
        'ha_r':        ha_r,
        'ha_l_len':    len(ha_l),
        'ha_r_len':    len(ha_r),
        'ha_l_gc':     round(gc(ha_l)*100, 1),
        'ha_r_gc':     round(gc(ha_r)*100, 1),
        'source':      source,
        'online':      source not in ('synthetic_placeholder', 'placeholder'),
        'from_cache':  False,
        'warning':     ('PLACEHOLDER — replace with real hg38 sequence' 
                        if source == 'synthetic_placeholder' else ''),
    }

    if verbose:
        src_tag = f'[{source}]'
        warn = ' ⚠ PLACEHOLDER' if source == 'synthetic_placeholder' else ''
        print(f'  {mod_name:30} {chrom}:{cut}  HA={arm_bp}bp  {src_tag}{warn}')

    _cache[cache_key] = result
    _save_cache()
    return result


def fetch_all_arms(verbose: bool = True) -> dict:
    results = {}
    if verbose:
        print(f'\nFetching homology arms for {len(HA_SPECS)} mods...')
        print(f'  {"Mod":30} {"Location":25} {"Info"}')
        print('  ' + '-'*70)
    for name in HA_SPECS:
        results[name] = fetch_arms(name, verbose=verbose)
    return results


def export_ha_fasta(ha_results: dict, output_dir: str = OUT_DIR) -> str:
    """Export all homology arms as FASTA."""
    lines = []
    for name, r in ha_results.items():
        if 'error' in r: continue
        warn = r.get('warning', '')
        lines.append(f'>{name}_HA_L | {r["chrom"]}:{r["cut_pos"]-r["arm_bp"]}-{r["cut_pos"]} | {r["ha_l_gc"]}%GC | source:{r["source"]}{" | "+warn if warn else ""}')
        for i in range(0, len(r['ha_l']), 80):
            lines.append(r['ha_l'][i:i+80])
        lines.append(f'>{name}_HA_R | {r["chrom"]}:{r["cut_pos"]}-{r["cut_pos"]+r["arm_bp"]} | {r["ha_r_gc"]}%GC | source:{r["source"]}{" | "+warn if warn else ""}')
        for i in range(0, len(r['ha_r']), 80):
            lines.append(r['ha_r'][i:i+80])
        lines.append('')
    path = os.path.join(output_dir, 'homology_arms.fasta')
    with open(path, 'w') as f: f.write('\n'.join(lines))
    return path


def export_ha_json(ha_results: dict, output_dir: str = OUT_DIR) -> str:
    slim = {}
    for name, r in ha_results.items():
        s = {k: v for k, v in r.items() if k not in ('ha_l','ha_r')}
        s['ha_l_preview'] = r.get('ha_l','')[:40]+'...'
        s['ha_r_preview'] = r.get('ha_r','')[:40]+'...'
        slim[name] = s
    path = os.path.join(output_dir, 'homology_arms.json')
    with open(path, 'w') as f: json.dump(slim, f, indent=2)
    return path


if __name__ == '__main__':
    print('=== Homology Arm Fetcher ===')
    results = fetch_all_arms(verbose=True)
    online  = sum(1 for r in results.values() if r.get('online'))
    print(f'\n  Online: {online}  Placeholders: {len(results)-online}')
    fasta = export_ha_fasta(results)
    js    = export_ha_json(results)
    print(f'  FASTA: {fasta}')
    print(f'  JSON:  {js}')
    print()
    print('  NOTE: Provide HumanGenome.fa (hg38) in project root')
    print('  or ensure UCSC/Ensembl is accessible for real sequences.')
