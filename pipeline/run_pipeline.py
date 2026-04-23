#!/usr/bin/env python3
"""
pipeline/run_pipeline.py
Main entry point for the molecular pipeline.
Integrates: sequence_fetcher → codon_optimizer → construct_builder
            → delivery_planner → guide_validator → homology_arms → pipeline_report

Usage:
    python3 pipeline/run_pipeline.py           # full run (offline mode)
    python3 pipeline/run_pipeline.py --online  # try NCBI for real sequences
    python3 pipeline/run_pipeline.py --clear-cache  # reset sequence cache
"""

import os, sys, json, argparse, time
from datetime import datetime

# Allow running from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

OUT_DIR = os.path.join(_ROOT, 'output_final', 'pipeline')
os.makedirs(OUT_DIR, exist_ok=True)

from pipeline.sequence_fetcher     import ACCESSIONS, get_sequence, fetch_all, clear_cache
from pipeline.codon_optimizer      import optimize_all, export_optimized_fasta, export_qc_json
from pipeline.construct_builder    import build_all, export_fasta, export_json
from pipeline.delivery_planner     import full_delivery_schedule, export_schedule
from pipeline.guide_validator      import validate_all, export_guide_report, summary_stats
from pipeline.pipeline_report      import generate_report
from pipeline.homology_arm_fetcher import fetch_all_arms, export_ha_fasta, export_ha_json


def banner():
    print('\033[36m')
    print('+==========================================================+')
    print('|   HOMO PERPETUUS — Molecular Pipeline  v1.0             |')
    print('|   Sequence -> Optimize -> Construct -> Deliver -> Validate  |')
    print('+==========================================================+')
    print('\033[0m')


def run_step(n: int, total: int, name: str):
    print(f'\n[{n}/{total}] {name}')
    print('  ' + '-' * 60)


def run(online: bool = False, verbose: bool = True) -> dict:
    banner()
    t0 = time.time()
    STEPS = 7

    # ── Step 1: Load CRISPR targets ───────────────────────────────────────────
    run_step(1, STEPS, 'Loading CRISPR targets from hp_modules')
    try:
        from hp_modules.crispr import CRISPR_TARGETS as crispr_targets
        from hp_modules.modifications import MODIFICATIONS
        n_mods = len(MODIFICATIONS)
        print(f'  OK Loaded {len(crispr_targets)} CRISPR targets from {n_mods} modifications')
    except Exception as e:
        print(f'  FAIL Could not load hp_modules: {e}')
        print('    Run from project root: python3 pipeline/run_pipeline.py')
        return {}

    # ── Step 2: Fetch sequences ───────────────────────────────────────────────
    run_step(2, STEPS, f'Fetching sequences ({"NCBI online" if online else "offline/cache mode"})')
    if online:
        seq_results = fetch_all(verbose=verbose)
    else:
        seq_results = {n: get_sequence(n, verbose=verbose) for n in ACCESSIONS}

    n_online  = sum(1 for v in seq_results.values() if v.get('online'))
    n_synth   = sum(1 for v in seq_results.values() if v.get('source') == 'SYNTHETIC')
    n_cache   = sum(1 for v in seq_results.values() if v.get('from_cache') and not v.get('online'))
    n_offline = sum(1 for v in seq_results.values()
                    if not v.get('online') and not v.get('from_cache')
                    and v.get('source') != 'SYNTHETIC')
    print(f'\n  Online: {n_online}  Synthetic: {n_synth}  '
          f'Cache: {n_cache}  Offline: {n_offline}')

    # ── Step 3: Codon optimisation ────────────────────────────────────────────
    run_step(3, STEPS, 'Codon optimization (human CAI, restriction sites, splice donors)')
    opt_results = optimize_all(seq_results, verbose=verbose)
    n_pass = sum(1 for r in opt_results.values()
                 if r.get('cai_ok') and r.get('translation_verified'))
    vals = [r.get('cai_final', 0) for r in opt_results.values() if 'cai_final' in r]
    mean_cai = sum(vals) / len(vals) if vals else 0
    print(f'\n  QC passed: {n_pass}/{len(opt_results)}  Mean CAI: {mean_cai:.3f}')

    # update seq_results with optimised CDS
    for name, opt in opt_results.items():
        if name in seq_results and 'optimized_cds' in opt:
            seq_results[name]['cds_nt'] = opt['optimized_cds']

    # ── Step 4: Build constructs ──────────────────────────────────────────────
    run_step(4, STEPS, 'Assembling HDR templates and expression cassettes')
    constructs = build_all(seq_results, verbose=verbose)
    n_aav = sum(1 for c in constructs.values() if '✓' in c.get('aav_status', ''))
    n_big = sum(1 for c in constructs.values() if 'too large' in c.get('aav_status', ''))
    print(f'\n  AAV compatible: {n_aav}  Require split-AAV/lentivirus: {n_big}')

    # ── Step 5: Delivery planning ─────────────────────────────────────────────
    run_step(5, STEPS, 'Generating 7-phase delivery schedule')
    schedule = full_delivery_schedule(constructs, verbose=verbose)
    total_scheduled = sum(len(p['mods']) for p in schedule)
    print(f'\n  Scheduled: {total_scheduled} mods across {len(schedule)} phases')

    # ── Step 6: Guide RNA validation ──────────────────────────────────────────
    run_step(6, STEPS, f'Validating {len(crispr_targets)} guide RNAs')
    guide_results = validate_all(crispr_targets, verbose=verbose)
    stats = summary_stats(guide_results)
    print(f'\n  PASS: {stats["pass"]}  WARN: {stats["warn"]}  FAIL: {stats["fail"]}')
    print(f'  Mean Doench: {stats["mean_doench"]}  Mean GC: {stats["mean_gc"]}%')

    # ── Step 7: Homology arms ─────────────────────────────────────────────────
    run_step(7, STEPS, 'Fetching homology arms (UCSC/Ensembl/local FASTA)')
    # Inject the already-built FastaIndex to avoid re-indexing HumanGenome.fa
    try:
        from pipeline.homology_arm_fetcher import set_fasta_index
        from hp_modules.config import FASTA_CANDIDATES
        import os
        for _fp in FASTA_CANDIDATES:
            if os.path.exists(_fp):
                from hp_modules.genome_io import FastaIndex as _FI
                set_fasta_index(_FI(_fp))
                print(f'  [HA] Using local FASTA: {os.path.basename(_fp)}')
                break
    except Exception as _e:
        print(f'  [HA] Local FASTA unavailable ({_e}), using UCSC/Ensembl')
    ha_results = fetch_all_arms(verbose=verbose)
    n_ha_online = sum(1 for r in ha_results.values() if r.get('online'))
    print(f'\n  Online: {n_ha_online}  Placeholders: {len(ha_results)-n_ha_online}')
    if n_ha_online == 0:
        print('  Provide HumanGenome.fa (hg38) in project root for real sequences.')

    # ── Export all outputs ────────────────────────────────────────────────────
    print('\n  Exporting outputs...')
    paths = {}
    paths['optimized_fasta']   = export_optimized_fasta(opt_results, OUT_DIR)
    paths['cai_qc_json']       = export_qc_json(opt_results, OUT_DIR)
    paths['construct_fasta']   = export_fasta(constructs, OUT_DIR)
    paths['construct_json']    = export_json(constructs, OUT_DIR)
    paths['delivery_schedule'] = export_schedule(schedule, OUT_DIR)
    paths['guide_report']      = export_guide_report(guide_results, OUT_DIR)
    paths['ha_fasta']          = export_ha_fasta(ha_results, OUT_DIR)
    paths['ha_json']           = export_ha_json(ha_results, OUT_DIR)
    paths['pipeline_report']   = generate_report(seq_results, constructs, schedule, OUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f'\n{"=" * 62}')
    print(f'  Pipeline complete in {elapsed:.1f}s')
    print(f'{"=" * 62}')
    print(f'  Mods processed:      {len(seq_results)}')
    print(f'  Sequences online:    {n_online}')
    print(f'  Codon CAI ≥ 0.80:    {n_pass}/{len(opt_results)}')
    print(f'  AAV-ready:           {n_aav}/{len(constructs)}')
    print(f'  Guides PASS:         {stats["pass"]}/{len(guide_results)}')
    print(f'\n  Output directory: {OUT_DIR}')
    for label, path in paths.items():
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f'  OK {label:<22} {os.path.basename(path)}  ({size//1024}kb)')

    if not online and n_online == 0:
        print(f'\n  WARN  All sequences are PLACEHOLDERS.')
        print(f'     Run with --online when NCBI is accessible')
        print(f'     to replace with real sequences.')

    return {
        'seq_results':  seq_results,
        'opt_results':  opt_results,
        'constructs':   constructs,
        'schedule':     schedule,
        'guides':       guide_results,
        'ha_results':   ha_results,
        'paths':        paths,
        'stats': {
            'n_mods':        len(seq_results),
            'n_online':      n_online,
            'n_aav_ready':   n_aav,
            'mean_cai':      round(mean_cai, 3),
            'guides_pass':   stats['pass'],
            'guides_warn':   stats['warn'],
            'guides_fail':   stats['fail'],
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homo Perpetuus Molecular Pipeline')
    parser.add_argument('--online',      action='store_true',
                        help='Try NCBI for real sequences')
    parser.add_argument('--quiet',       action='store_true',
                        help='Suppress per-mod output')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear sequence cache and re-fetch')
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()

    run(online=args.online, verbose=not args.quiet)
