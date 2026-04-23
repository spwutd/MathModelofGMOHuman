"""hp_modules/report.py — generate_report: text report writer."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
from datetime import datetime
from hp_modules.modifications import GENE_DB
from hp_modules.ncbi_api import get_gtex_data

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from hp_modules.config import BASE_DIR, OUTPUT_DIR
from hp_modules.modifications import MODIFICATIONS

def generate_report(results, genome_info, promoter_data, sim, crispr_results=None):
    sv = sim.survival_extended()
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    W = 74
    lines += ['='*W,
              '  HOMO PERPETUUS v8.0 FINAL — GENOME MODIFICATION SIMULATION REPORT',
              f'  Generated : {ts}',
              '='*W]

    lines += ['\n[GENOME]\n']
    for k,v in genome_info.items():
        lines.append(f'  {k:<35} {v}')

    lines += ['\n[MODIFICATIONS — DETAILED]\n']
    for r in results:
        mid = r.get('mod_id','?')
        lines.append(f'  ▶  {mid}')
        lines.append(f'     Type          : {r.get("type","")}')
        lines.append(f'     Module        : {r.get("module","")}')
        lines.append(f'     Risk          : {r.get("risk","")}')
        if r.get('gene'):
            gd = GENE_DB.get(r['gene'],{})
            lines.append(f'     Gene          : {r["gene"]}  ({gd.get("chr","?")}:{gd.get("start","?")}–{gd.get("end","?")}  {gd.get("strand","")})')
            lines.append(f'     Description   : {gd.get("desc","")}')
        if r.get('source_organism'):
            lines.append(f'     Source        : {r["source_organism"]}')
            lines.append(f'     Function      : {r.get("function","")}')
            lines.append(f'     Insertion     : {r.get("insertion_site","")}')
        prot = r.get('protein',{})
        if prot.get('length'):
            lines.append(f'     Protein       : {prot["length"]} aa  |  {prot["MW_kDa"]} kDa  |  '
                         f'charge {prot.get("charge","?")}  |  '
                         f'stable: {prot.get("stable","?")}')
            lines.append(f'     Hydrophobicity: {prot.get("avg_hydrophobicity","?")}  |  '
                         f'Instability idx: {prot.get("instability_index","?")}')
        vs = r.get('validation_status','')
        if vs:
            lines.append(f'     Validation    : {vs}  ({r.get("validation_ratio",""):.1%} of expected length)'
                         if isinstance(r.get("validation_ratio"),float) else
                         f'     Validation    : {vs}')
        src = r.get('source','')
        if src: lines.append(f'     Sequence src  : {src}')
        lines.append(f'     Effect        : {r.get("effect","")}')
        lines.append('')

    lines += ['\n[PROMOTER CpG ANALYSIS]\n',
              f'  {"Gene":<12} {"GC%":>6}  {"CpG Islands":>11}  {"Obs/Exp":>8}  {"Status":<10}',
              '  ' + '-'*55]
    for gn, d in sorted(promoter_data.items()):
        lines.append(f'  {gn:<12} {d.get("gc_content_pct",0):>5.1f}%  '
                     f'{d.get("cpg_islands",0):>11}  '
                     f'{d.get("avg_cpg_obs_exp",0):>8.3f}  '
                     f'{d.get("promoter_status","?"):<10}')

    lines += ['\n[SURVIVAL PROJECTIONS]\n',
              f'  {"Scenario":<42} {"Median lifespan":>16}',
              '  ' + '-'*60,
              f'  {"Normal human (Gompertz)":<42} {sv["med_normal"]:>14} years']
    for label, med in sv['hp_medians']:
        lines.append(f'  {"HP — "+label:<42} {med:>14,} years')

    lines += ['\n  Dominant mortality (HP) : Accidental death from external causes',
              '  Biological ceiling      : ~20,000–50,000 years (neuronal accumulation)',
              '  Key insight             : Biological aging eliminated; survival = safety problem']

    lines += ['\n[RISK SUMMARY]\n']
    for r in results:
        risk = r.get('risk','?').split(' —')[0].split(' —')[0].split()[0] if r.get('risk') else '?'
        gene = r.get('gene', r.get('foreign_gene',''))
        lines.append(f'  {r["mod_id"]:<28}  {risk:<10}  {gene}')

    # GTEx expression summary
    lines += ['\n[TISSUE EXPRESSION (GTEx v8)]\n']
    lines += [f'  {"Gene":<10} {"Thymus":>8} {"Liver":>8} {"Heart":>8} {"Brain":>8}  {"Whole Blood":>12}  {"Source"}']
    lines += ['  '+'-'*72]
    try:
        gd = get_gtex_data(list(GENE_DB.keys()))
        for g in ['TP53','AR','AIRE','FOXN1','LAMP2','GLO1','ERCC1','RAD51','FEN1','CCND1','TERT']:
            expr = gd.get(g,{})
            src_tag = '(GTEx v8)' if expr.get('_source') != 'literature_fallback' else '(lit. est.)'
            lines.append(f'  {g:<10} '
                         f'{expr.get("Thymus",0):>8.1f} '
                         f'{expr.get("Liver",0):>8.1f} '
                         f'{expr.get("Heart LV",0):>8.1f} '
                         f'{expr.get("Brain Cortex",0):>8.1f} '
                         f'{expr.get("Whole Blood",0):>12.1f}  '
                         f'{src_tag}')
    except Exception:
        lines.append('  (GTEx data unavailable)')

    lines += ['', '='*W]
    txt = '\n'.join(lines)
    path = os.path.join(OUTPUT_DIR, 'report_v8_final.txt')
    with open(path, 'w', encoding='utf-8') as f: f.write(txt)
    return txt, path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════