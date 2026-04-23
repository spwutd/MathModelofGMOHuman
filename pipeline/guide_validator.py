#!/usr/bin/env python3
"""
pipeline/guide_validator.py
Validates all CRISPR guide RNAs from CRISPR_TARGETS.
Checks:
  - GC content (optimal 40-70%)
  - Homopolymer runs (bad for sgRNA transcription)
  - PAM presence (NGG for SpCas9)
  - On-target score (Rule Set 2 / Doench 2016 simplified)
  - Predicted secondary structure risk (simple stem-loop check)
  - Seed region analysis (positions 1-12 from PAM — most critical)

For off-target search against full genome: requires FASTA file.
Without FASTA: computes all sequence-based metrics.
"""

import os, json, re, itertools, math
from typing import Optional

_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, '..', 'output_final', 'pipeline')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Doench 2016 Rule Set 2 — position-specific nucleotide weights ─────────────
# Simplified version: nucleotide preference at each position (1-20, 1=PAM-distal)
# Source: Doench et al. 2016 Nature Biotechnology 34:184
_POS_WEIGHTS = {
    # Position: {nuc: weight} — positive = good, negative = bad
    1:  {'G': 0.15, 'C': 0.05, 'T': -0.10, 'A': -0.10},
    2:  {'G': 0.10, 'C': 0.08, 'T': -0.05, 'A': -0.13},
    3:  {'G': 0.05, 'C': 0.03, 'T': 0.02,  'A': -0.10},
    4:  {'G': 0.05, 'C': 0.05, 'T': -0.05, 'A': -0.05},
    5:  {'G': 0.08, 'C': 0.05, 'T': -0.08, 'A': -0.05},
    6:  {'G': 0.10, 'C': 0.05, 'T': -0.05, 'A': -0.10},
    7:  {'G': 0.08, 'C': 0.08, 'T': -0.03, 'A': -0.08},
    8:  {'G': 0.05, 'C': 0.05, 'T': 0.05,  'A': -0.15},
    9:  {'G': 0.10, 'C': 0.05, 'T': -0.05, 'A': -0.10},
    10: {'G': 0.15, 'C': 0.10, 'T': -0.10, 'A': -0.15},  # seed region start
    11: {'G': 0.15, 'C': 0.10, 'T': -0.08, 'A': -0.17},
    12: {'G': 0.20, 'C': 0.10, 'T': -0.10, 'A': -0.20},  # seed core
    13: {'G': 0.20, 'C': 0.15, 'T': -0.10, 'A': -0.25},
    14: {'G': 0.25, 'C': 0.15, 'T': -0.10, 'A': -0.30},
    15: {'G': 0.25, 'C': 0.20, 'T': -0.05, 'A': -0.40},
    16: {'G': 0.30, 'C': 0.20, 'T': -0.05, 'A': -0.45},
    17: {'G': 0.30, 'C': 0.25, 'T': -0.10, 'A': -0.45},
    18: {'G': 0.35, 'C': 0.25, 'T': -0.10, 'A': -0.50},
    19: {'G': 0.30, 'C': 0.25, 'T': -0.05, 'A': -0.50},
    20: {'G': 0.20, 'C': 0.15, 'T': 0.05,  'A': -0.40},  # PAM-proximal
}

# ── Thermodynamic complement ───────────────────────────────────────────────────
_COMP = str.maketrans('ACGTacgt', 'TGCAtgca')

def reverse_complement(seq: str) -> str:
    return seq.translate(_COMP)[::-1]

# ── Guide metrics ──────────────────────────────────────────────────────────────

def gc_content(guide: str) -> float:
    g = guide.upper()
    return (g.count('G') + g.count('C')) / len(g)


def homopolymer_max(guide: str) -> int:
    """Longest run of the same nucleotide."""
    max_run = 1; cur_run = 1
    for i in range(1, len(guide)):
        if guide[i] == guide[i-1]:
            cur_run += 1; max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run


def doench_score(guide: str) -> float:
    """
    Simplified Doench 2016 on-target score (0-1).
    Guide should be 20nt, PAM-proximal at position 20.
    """
    g = guide.upper()
    if len(g) != 20:
        return 0.5
    score = 0.60  # baseline
    for pos, nuc in enumerate(g, 1):
        score += _POS_WEIGHTS.get(pos, {}).get(nuc, 0)
    # GC bonus
    gc = gc_content(g)
    if 0.4 <= gc <= 0.7:
        score += 0.1
    elif gc < 0.3 or gc > 0.8:
        score -= 0.15
    return max(0.0, min(1.0, score))


def stem_loop_risk(guide: str, window: int = 6) -> bool:
    """
    Check if guide has significant internal stem-loop potential.
    Returns True if risk detected (consecutive complementary bases ≥ window).
    """
    g = guide.upper()
    for i in range(len(g) - window*2):
        subseq = g[i:i+window]
        rc     = reverse_complement(subseq)
        # Look for RC match downstream
        if rc in g[i+window:]:
            return True
    return False


def pam_check(guide: str, pam: str) -> dict:
    """Check PAM compatibility for SpCas9 (NGG) or other systems."""
    pam_upper = pam.upper()
    if pam_upper.endswith('GG'):
        return {'compatible': True, 'system': 'SpCas9 (NGG)', 'note': 'standard'}
    elif pam_upper == 'NG':
        return {'compatible': True, 'system': 'SpRY (NG)', 'note': 'relaxed PAM variant'}
    elif pam_upper == 'NNGRRT':
        return {'compatible': True, 'system': 'SaCas9', 'note': 'shorter guide 21nt'}
    elif pam_upper == 'TTN':
        return {'compatible': True, 'system': 'AsCas12a', 'note': 'TTTN PAM, different cut'}
    else:
        return {'compatible': False, 'system': 'unknown', 'note': f'PAM {pam} not recognized'}


def seed_gc(guide: str) -> float:
    """GC content of seed region (positions 9-20 from PAM-distal end = last 12 nt)."""
    seed = guide[-12:].upper()
    return (seed.count('G') + seed.count('C')) / 12


def validate_guide(name: str, guide: str, pam: str, target_chr: str,
                   target_pos: int) -> dict:
    """
    Full validation of one guide RNA.
    Returns dict with all metrics and a pass/fail verdict.
    """
    g = guide.upper()
    gc  = gc_content(g)
    hom = homopolymer_max(g)
    ds  = doench_score(g)
    slr = stem_loop_risk(g)
    pam_ok = pam_check(g, pam)
    s_gc = seed_gc(g)
    length_ok = (18 <= len(g) <= 21)

    issues = []
    if gc < 0.35:   issues.append(f'LOW GC ({gc:.0%})')
    if gc > 0.75:   issues.append(f'HIGH GC ({gc:.0%})')
    if hom >= 4:    issues.append(f'HOMOPOLYMER run={hom}')
    if slr:         issues.append('STEM-LOOP risk')
    if not pam_ok['compatible']: issues.append(f'PAM incompatible ({pam})')
    if not length_ok: issues.append(f'LENGTH {len(g)}nt (expect 20)')
    if g.startswith('TTTT'): issues.append('TTTT start (RNA Pol III terminator)')

    # On-target score thresholds
    if ds >= 0.70:   score_grade = 'EXCELLENT'
    elif ds >= 0.55: score_grade = 'GOOD'
    elif ds >= 0.40: score_grade = 'FAIR'
    else:            score_grade = 'POOR'; issues.append(f'LOW on-target score ({ds:.2f})')

    verdict = 'PASS' if not issues else ('WARN' if len(issues) <= 1 else 'FAIL')

    return {
        'mod_name':      name,
        'guide_seq':     g,
        'pam':           pam,
        'target':        f'{target_chr}:{target_pos}',
        'length':        len(g),
        'gc_pct':        round(gc * 100, 1),
        'seed_gc_pct':   round(s_gc * 100, 1),
        'homopolymer_max': hom,
        'doench_score':  round(ds, 3),
        'score_grade':   score_grade,
        'stem_loop_risk': slr,
        'pam_system':    pam_ok['system'],
        'issues':        issues,
        'verdict':       verdict,
    }


def validate_all(crispr_targets: dict, verbose: bool = True) -> dict:
    """Validate all guides from CRISPR_TARGETS dict."""
    results = {}
    if verbose:
        print(f'\nValidating {len(crispr_targets)} guide RNAs...')
        print(f'  {"Mod":28} {"Guide":22} {"GC%":>5} {"Score":>7} {"Grade":10} {"Verdict"}')
        print('  ' + '-'*85)

    for name, data in crispr_targets.items():
        guide = data.get('guide', '')
        pam   = data.get('pam', 'TGG')
        chrom = data.get('chr', '?')
        pos   = data.get('cut', 0)
        r = validate_guide(name, guide, pam, chrom, pos)
        results[name] = r
        if verbose:
            issues_str = ', '.join(r['issues']) if r['issues'] else 'none'
            print(f'  {name:28} {r["guide_seq"]:22} {r["gc_pct"]:>4.0f}% '
                  f'{r["doench_score"]:>7.3f} {r["score_grade"]:10} '
                  f'{r["verdict"]}  {issues_str}')
    return results


def summary_stats(results: dict) -> dict:
    vals = list(results.values())
    return {
        'total':     len(vals),
        'pass':      sum(1 for r in vals if r['verdict'] == 'PASS'),
        'warn':      sum(1 for r in vals if r['verdict'] == 'WARN'),
        'fail':      sum(1 for r in vals if r['verdict'] == 'FAIL'),
        'mean_gc':   round(sum(r['gc_pct'] for r in vals)/len(vals), 1),
        'mean_doench': round(sum(r['doench_score'] for r in vals)/len(vals), 3),
        'mean_seed_gc': round(sum(r['seed_gc_pct'] for r in vals)/len(vals), 1),
    }


def export_guide_report(results: dict, output_dir: str = OUT_DIR) -> str:
    stats = summary_stats(results)
    lines = ['CRISPR GUIDE RNA VALIDATION REPORT', '='*60, '']
    lines.append(f'Total guides:  {stats["total"]}')
    lines.append(f'PASS:          {stats["pass"]}')
    lines.append(f'WARN:          {stats["warn"]}')
    lines.append(f'FAIL:          {stats["fail"]}')
    lines.append(f'Mean GC:       {stats["mean_gc"]}%')
    lines.append(f'Mean Doench:   {stats["mean_doench"]}')
    lines.append(f'Mean seed GC:  {stats["mean_seed_gc"]}%')
    lines.append('')

    for verdict in ['FAIL', 'WARN', 'PASS']:
        group = [r for r in results.values() if r['verdict'] == verdict]
        if not group:
            continue
        lines.append(f'\n── {verdict} ({len(group)}) ──')
        for r in group:
            lines.append(f'  {r["mod_name"]}')
            lines.append(f'    Guide:  5\'-{r["guide_seq"]}-3\' PAM:{r["pam"]}')
            lines.append(f'    Target: {r["target"]}')
            lines.append(f'    GC: {r["gc_pct"]}%  Seed GC: {r["seed_gc_pct"]}%  '
                         f'Doench: {r["doench_score"]} ({r["score_grade"]})')
            if r['issues']:
                lines.append(f'    Issues: {", ".join(r["issues"])}')

    lines.append('\n' + '='*60)
    lines.append('NOTE: Off-target analysis requires genome FASTA.')
    lines.append('      Run with HumanGenome.fa for full off-target scoring.')
    lines.append('      Online tools: CRISPOR (crispor.tefor.net), Benchling.')

    path = os.path.join(output_dir, 'guide_validation.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    json_path = os.path.join(output_dir, 'guide_validation.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return path
if __name__ == '__main__':
    import sys as _s; _s.path.insert(0, __import__('os').path.dirname(
        __import__('os').path.dirname(__import__('os').path.abspath(__file__))))
    from hp_modules.crispr import CRISPR_TARGETS

    print('=== Guide RNA Validator ===')
    results = validate_all(CRISPR_TARGETS, verbose=True)
    stats   = summary_stats(results)

    print(f'\nSummary: PASS={stats["pass"]}  WARN={stats["warn"]}  FAIL={stats["fail"]}')
    print(f'Mean Doench score: {stats["mean_doench"]}  Mean GC: {stats["mean_gc"]}%')

    path = export_guide_report(results)
    print(f'\nReport: {path}')
