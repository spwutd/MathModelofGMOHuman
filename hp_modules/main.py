def _import_crispr():
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "crispr_offtarget",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "crispr_offtarget.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # BUG FIX v7: crispr_offtarget.py hardcodes 'output_v2' — override with current OUTPUT_DIR
        m.OUTPUT_DIR = OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return m
    except Exception as e:
        print(f"  [CRISPR] Module load failed: {e}")
        return None
from datetime import datetime


"""hp_modules/main.py — entry point, banner, main()."""
import os, sys, json, re, math, time
import ssl, urllib.request, urllib.parse
import numpy as np
import matplotlib
from hp_modules.ncbi_api import get_gtex_data
from hp_modules.plots import (plot_genome, plot_mods_overview,
                               plot_protein_validation, plot_promoter_cpg,
                               plot_simulations, plot_protein_summary,
                               plot_module_interactions, plot_v5_mechanisms,
                               plot_v6_mechanisms, plot_v7_mechanisms,
                               plot_gtex_expression, plot_module_crosstalk,
                               plot_ai_risk_dashboard)
from hp_modules.crispr import (CRISPR_TARGETS, run_crispr_offtarget,
                                plot_crispr_offtarget, generate_crispr_report)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hp_modules.config          import BASE_DIR, OUTPUT_DIR, FASTA_CANDIDATES, GTF_CANDIDATES
from hp_modules.modifications   import MODIFICATIONS, FOREIGN_GENES, GENE_DB
from hp_modules.genome_io       import FastaIndex, GtfAnnotation
from hp_modules.ncbi_api        import (fetch_uniprot_sequence, get_protein_sequence,
                                         fetch_gtex_for_all_genes, run_esm2_all,
                                         get_alphafold_all, get_opentargets_all)
from hp_modules.modification_engine import ModificationEngine
from hp_modules.simulation_models   import SimulationModels
from hp_modules.ode_engine          import ModuleCrosstalk
from hp_modules.plots               import (save_fig, style_ax, plot_mods_overview,
                                             plot_protein_validation, plot_promoter_cpg,
                                             plot_simulations, plot_protein_summary,
                                             plot_module_interactions, plot_v5_mechanisms,
                                             plot_v6_mechanisms, plot_v7_mechanisms,
                                             plot_gtex_expression, plot_module_crosstalk,
                                             plot_ai_risk_dashboard)
from hp_modules.report              import generate_report
from hp_modules.crispr              import CRISPR_TARGETS, run_crispr_offtarget, plot_crispr_offtarget

def banner():
    print('\033[36m')
    print('+==================================================================+')
    print('|     HOMO PERPETUUS — Genome Simulation Engine  v8.0 FINAL      |')
    print('|     29 mods · 11 modules · 9 organisms · 10 000y Q≥65%          |')
    print('+==================================================================+')
    print('\033[0m')

def find_file(candidates):
    for c in candidates:
        if os.path.exists(c): return c
    return None

def _check_apis():
    """Quick connectivity check — shows which data sources are available."""
    print("\n  [API status]")
    checks = [
        ("UniProt",  "https://rest.uniprot.org/uniprotkb/P04637.json"),
        ("NCBI",     "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=protein&retmode=json"),
        ("Ensembl",  "https://rest.ensembl.org/info/ping"),
        ("GTEx",     "https://gtexportal.org/rest/v1/dataset/datasetInfo?datasetId=gtex_v8"),
    ]
    available = []
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    for name, url in checks:
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'HomoPerpetuum/3.0'})
            with urllib.request.urlopen(req, timeout=4, context=ctx):
                pass
            print(f"  OK {name} — online")
            available.append(name)
        except:
            print(f"  FAIL {name} — offline (using cache/synthetic)")
    if not available:
        print("  -> Running in full offline mode")
    elif "UniProt" in available:
        print("  -> Real protein sequences available")
    print()
    return available


def main():
    banner()
    _check_apis()

    # 1. FASTA
    fasta_path = find_file(FASTA_CANDIDATES)
    fasta = None; genome_info = {}
    if fasta_path:
        print(f'\n[1/6] FASTA: {fasta_path}')
        fasta = FastaIndex(fasta_path)
        chroms = [c for c in fasta.chromosomes() if re.match(r'chr(\d+|X|Y)$',c)]
        total  = sum(fasta.seq_length(c) for c in fasta.chromosomes())
        genome_info = {
            'File': os.path.basename(fasta_path),
            'File size': f'{os.path.getsize(fasta_path)/1e9:.2f} GB',
            'Total sequences': len(fasta.chromosomes()),
            'Standard chromosomes': len(chroms),
            'Total base pairs': f'{total:,}',
        }
        print('\n  Generating genome overview plot...')
        plot_genome(fasta)
    else:
        print('\n[1/6] No FASTA found -> synthetic mode')
        print('      Tip: place HumanGenome.fa next to this script')
        genome_info = {'Mode': 'SYNTHETIC — place HumanGenome.fa to enable real analysis'}

    # 2. GTF
    gtf_path = find_file(GTF_CANDIDATES)
    gtf_path_used = gtf_path
    gtf = None
    if gtf_path:
        print(f'\n[2/6] GTF: {gtf_path}')
        gtf = GtfAnnotation(gtf_path)
    else:
        print('\n[2/6] No GTF found -> genomic sequence mode (intron-aware fallback)')
        print('      Tip: download gencode.v38.annotation.gtf.gz and place next to script')
        print('           wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz')

    # 3. Modifications
    print('\n[3/6] Running modifications (29 mods — v8 FINAL)...')
    engine = ModificationEngine(fasta, gtf)
    results = engine.run()
    print(f'  OK {len(results)} modifications processed')

    # 4. Plots
    print('\n[4/6] Generating plots...')
    sim = SimulationModels()
    plot_mods_overview(results, sim)
    plot_promoter_cpg(engine.promoter_data)
    plot_simulations(sim)
    plot_protein_summary(results, engine.promoter_data)

    # GTEx expression heatmap
    print('\n  Fetching GTEx tissue expression data...')
    gtex_genes = list(GENE_DB.keys())
    gtex_data  = get_gtex_data(gtex_genes)
    plot_gtex_expression(gtex_data, gtex_genes)

    # Module crosstalk coupled ODE
    print('\n  Running module crosstalk simulation...')
    plot_module_crosstalk(years=500)

    # NEW v5: mechanisms panel
    print('\n  Running v5 mechanism simulations...')
    plot_v5_mechanisms(sim)

    # NEW v6: senescence/inflammaging/cardiac quartet panel
    print('\n  Running v6 mechanism simulations...')
    plot_v6_mechanisms(sim)

    # NEW v7: epigenetics/glucosspan/heteroplasmy/Monte Carlo/ODE-Gompertz panel
    print('\n  Running v7 mechanism simulations (Monte Carlo n=150, ~20s)...')
    plot_v7_mechanisms(sim)

    # CRISPR off-target analysis — now in hp_modules/crispr.py
    print('\n  Running CRISPR off-target analysis...')
    crispr_results = []
    try:
        crispr_results = run_crispr_offtarget(
            fasta=fasta, gene_db=GENE_DB,
            targets=CRISPR_TARGETS,
            max_mm=3, verbose=False)
        plot_crispr_offtarget(crispr_results)
        print(f'  OK CRISPR: {len(crispr_results)} guides analysed')
    except Exception as e:
        print(f'  [CRISPR] Analysis failed: {e}')
    print('  OK All plots saved')

    # 5. Report
    print('\n[5/6] Generating report...')
    report_txt, report_path = generate_report(results, genome_info,
                                               engine.promoter_data, sim,
                                               crispr_results=crispr_results)
    print(report_txt)

    # 6. JSON
    print('\n[6/6] Saving JSON...')
    json_path = os.path.join(OUTPUT_DIR, 'modifications_v8_final.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'genome': genome_info,
                   'modifications': results,
                   'promoters': engine.promoter_data},
                  f, indent=2, default=str)
    print(f'  OK {json_path}')

    # ── [7/7] Molecular pipeline ───────────────────────────────────────────────
    # Sequence fetch → codon optimize → construct assembly → delivery plan
    # → guide validation → homology arms
    print('\n[7/7] Running molecular pipeline...')
    try:
        import sys as _sys
        _sys.path.insert(0, BASE_DIR)
        from pipeline.run_pipeline import run as _run_pipeline
        from pipeline.homology_arm_fetcher import set_fasta_index as _set_fa
        if fasta is not None:
            _set_fa(fasta)  # pass already-indexed FASTA — avoids 32× re-indexing
        _pipeline_result = _run_pipeline(online=False, verbose=False)
        if _pipeline_result:
            s = _pipeline_result.get('stats', {})
            print(f'  OK Pipeline: {s.get("n_mods","?")} mods  '
                  f'CAI={s.get("mean_cai","?")}  '
                  f'AAV-ready={s.get("n_aav_ready","?")}  '
                  f'Guides PASS={s.get("guides_pass","?")}')
            pdir = os.path.join(OUTPUT_DIR, 'pipeline')
            os.makedirs(pdir, exist_ok=True)
            for fname in ['constructs_all.fasta', 'optimized_cds.fasta',
                          'homology_arms.fasta', 'delivery_schedule.json',
                          'guide_validation.txt', 'pipeline_report.txt',
                          'constructs_metadata.json', 'codon_optimization_qc.json']:
                src = os.path.join(BASE_DIR, 'output_final', 'pipeline', fname)
                dst = os.path.join(pdir, fname)
                if os.path.exists(src) and src != dst:
                    import shutil; shutil.copy2(src, dst)
            print(f'  OK Pipeline outputs in: {pdir}')
    except ImportError:
        print('  ℹ  pipeline/ not found — skipping molecular pipeline')
        print('     Place pipeline/ folder next to homo_perpetuus_final.py to enable')
    except Exception as _e:
        print(f'  WARN  Pipeline error: {_e}')

    print(f'\n  OK All outputs in: {OUTPUT_DIR}')

