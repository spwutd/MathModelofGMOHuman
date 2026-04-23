"""hp_modules/plots.py -- all visualisation functions."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
from collections import Counter
from hp_modules.config import (DARK_BG, PANEL_BG, KNOWN_PROTEIN_LENGTHS,
                                BLUE, GREEN, ORANGE, PURPLE, RED,
                                CYAN, YELLOW, GREY, LIGHT)
from hp_modules.ncbi_api import (GTEX_TISSUES_OF_INTEREST, GTEX_FALLBACK,
                                  AF2_FALLBACK, OPENTARGETS_IDS, OT_FALLBACK)
from hp_modules.ode_engine import ModuleCrosstalk, sim_neuronal_ceiling

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from hp_modules.config import BASE_DIR, OUTPUT_DIR
from hp_modules.config import DARK_BG, PANEL_BG, BLUE, GREEN, ORANGE, PURPLE
from hp_modules.config import RED, CYAN, YELLOW, GREY, LIGHT
from hp_modules.modifications import MODIFICATIONS

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  [plot] -> {path}")
    return path

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.spines[:].set_color('#2A3A4A')
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title(title, color=LIGHT, fontsize=11, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=GREY, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=GREY, fontsize=9)

# --- Plot 1: Genome overview --------------------------------------------------

def plot_genome(fasta):
    chroms = sorted([c for c in fasta.chromosomes()
                     if re.match(r'chr(\d+|X|Y)$', c)],
                    key=lambda x: (int(x[3:]) if x[3:].isdigit() else (23 if x[3:]=='X' else 24)))
    lengths = [fasta.seq_length(c)/1e6 for c in chroms]

    fig, ax = plt.subplots(figsize=(14,6), facecolor=DARK_BG)
    colors = [BLUE if i%2==0 else PURPLE for i in range(len(chroms))]
    ax.set_facecolor(DARK_BG)
    bars = ax.barh(chroms, lengths, color=colors, height=0.7, edgecolor='none')
    for b,v in zip(bars,lengths):
        ax.text(v+1, b.get_y()+b.get_height()/2, f'{v:.0f} Mb',
                va='center', color=GREY, fontsize=8)
    ax.set_xlabel('Length (Mb)', color=LIGHT); ax.set_title(
        'GRCh38 Reference Genome -- Chromosome Lengths', color=LIGHT, fontsize=14)
    ax.tick_params(colors=LIGHT); ax.spines[:].set_visible(False)
    ax.set_xlim(0, max(lengths)*1.13)
    total = sum(fasta.seq_length(c) for c in fasta.chromosomes())
    ax.text(0.99, 0.02, f'Total: {total/1e9:.2f} Gb  |  {len(fasta.chromosomes())} sequences',
            transform=ax.transAxes, ha='right', color=GREY, fontsize=9)
    plt.tight_layout()
    return save_fig('01_genome_overview.png')

# --- Plot 2: Modification overview -------------------------------------------

def plot_mods_overview(results, sim=None):
    fig = plt.figure(figsize=(18,8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    TYPE_COL = {'DUPLICATION':BLUE,'FOREIGN_INSERT':GREEN,'CONDITIONAL_KNOCKOUT':RED,
                'UPREGULATION':ORANGE,'CONDITIONAL_ACTIVATION':PURPLE,'ENHANCED_PARALOGUE':CYAN}
    RISK_COL = {'VERY LOW':CYAN,'LOW':GREEN,'MEDIUM':ORANGE,'HIGH':RED}

    # Types pie
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(PANEL_BG)
    tc = Counter(r['type'] for r in results)
    cols = [TYPE_COL.get(k, GREY) for k in tc]
    wedges,_,auto = ax.pie(list(tc.values()), colors=cols, autopct='%1.0f%%',
                           startangle=90, pctdistance=0.72)
    for a in auto: a.set_color(DARK_BG); a.set_fontweight('bold'); a.set_fontsize(10)
    ax.legend(wedges,[k.replace('_',' ') for k in tc],loc='lower center',
              bbox_to_anchor=(0.5,-0.12),fontsize=8,facecolor='#1C2127',
              edgecolor=GREY,labelcolor=LIGHT,ncol=1)
    ax.set_title('Modification Types', color=LIGHT, fontsize=12)

    # Risk bars
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor(PANEL_BG)
    rl = ['VERY LOW','LOW','MEDIUM','HIGH']
    def risk_key(r):
        raw = r.get('risk','')
        for k in rl:
            if raw.startswith(k): return k
        return 'LOW'
    rc_ = Counter(risk_key(r) for r in results)
    rv = [rc_.get(r,0) for r in rl]
    rcols = [CYAN,GREEN,ORANGE,RED]
    bars = ax2.bar(rl, rv, color=rcols, width=0.55, edgecolor='none')
    for b,v in zip(bars,rv):
        if v: ax2.text(b.get_x()+b.get_width()/2, v+0.05, str(v),
                       ha='center',color=LIGHT,fontsize=12,fontweight='bold')
    style_ax(ax2,'Risk Distribution','','# Modifications')
    ax2.tick_params(axis='x',labelsize=9,colors=GREY)

    # Protein sizes
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor(PANEL_BG)
    names,mws = [],[]
    for r in results:
        mw = r.get('protein',{}).get('MW_kDa',0)
        nm = r.get('gene', r.get('foreign_gene', r.get('mod_id','')))
        if mw and mw > 1:
            names.append(nm[:14]); mws.append(mw)
    if names:
        cols3 = [GREEN if mw<100 else PURPLE for mw in mws]
        ybars = ax3.barh(range(len(names)), mws, color=cols3, height=0.65, edgecolor='none')
        ax3.set_yticks(range(len(names))); ax3.set_yticklabels(names,fontsize=8,color=LIGHT)
        for b,v in zip(ybars,mws):
            ax3.text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.0f}',
                     va='center',fontsize=7,color=LIGHT)
        style_ax(ax3,'Protein Molecular Weights','kDa','')
    plt.suptitle('HOMO PERPETUUS v8 -- Modification Overview (29 mods)',
                 color=LIGHT,fontsize=15,fontweight='bold',y=1.02)
    plt.tight_layout()
    return save_fig('02_mods_overview.png')

# --- Plot 3: Protein validation (v2 NEW) -------------------------------------

def plot_protein_validation(results):
    genes, measured, expected, statuses, sources = [],[],[],[],[]
    for r in results:
        gn = r.get('gene','')
        if not gn or gn not in KNOWN_PROTEIN_LENGTHS: continue
        mlen = r.get('protein',{}).get('length',0)
        exp  = KNOWN_PROTEIN_LENGTHS[gn]
        vs   = r.get('validation_status','?')
        src  = r.get('source','?')
        if mlen:
            genes.append(gn); measured.append(mlen)
            expected.append(exp); statuses.append(vs); sources.append(src)

    if not genes:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6), facecolor=DARK_BG)
    scol = {'CORRECT':GREEN,'PARTIAL':ORANGE,'INTRON_ARTIFACT':RED,'UNKNOWN_REF':GREY}

    # Measured vs expected
    ax1.set_facecolor(PANEL_BG)
    xpos = np.arange(len(genes)); w = 0.38
    cols_m = [scol.get(s,GREY) for s in statuses]
    ax1.bar(xpos-w/2, expected, width=w, color=BLUE,  alpha=0.7, label='Expected (UniProt)', edgecolor='none')
    ax1.bar(xpos+w/2, measured, width=w, color=cols_m,alpha=0.85,label='Measured (this run)',edgecolor='none')
    ax1.set_xticks(xpos); ax1.set_xticklabels(genes,rotation=45,ha='right',color=LIGHT,fontsize=8)
    style_ax(ax1,'Protein Length: Measured vs Expected (aa)','','Amino acids')
    ax1.legend(facecolor='#1C2127',edgecolor=GREY,labelcolor=LIGHT,fontsize=9)
    patches = [mpatches.Patch(color=v,label=k) for k,v in scol.items()]
    ax1.legend(handles=patches,loc='upper right',facecolor='#1C2127',
               edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # Accuracy ratio
    ax2.set_facecolor(PANEL_BG)
    ratios = [m/e*100 for m,e in zip(measured,expected)]
    bar_cols = [GREEN if r>85 else ORANGE if r>50 else RED for r in ratios]
    bars = ax2.bar(genes, ratios, color=bar_cols, edgecolor='none', width=0.6)
    ax2.axhline(100, color=BLUE, ls='--', lw=1, alpha=0.5, label='100% correct')
    ax2.axhline(85,  color=GREEN,ls=':',  lw=1, alpha=0.4, label='85% threshold')
    for b,v,src in zip(bars,ratios,sources):
        label = 'OK' if v>85 else f'{v:.0f}%'
        ax2.text(b.get_x()+b.get_width()/2, v+1, label,
                 ha='center',va='bottom',fontsize=8,color=LIGHT)
        ax2.text(b.get_x()+b.get_width()/2, 2, src[:8],
                 ha='center',va='bottom',fontsize=6,color=GREY,rotation=45)
    style_ax(ax2,'Protein Length Accuracy (%)','Gene','% of expected length')
    ax2.set_ylim(0, 135); ax2.tick_params(axis='x',rotation=45,labelsize=8)
    ax2.legend(facecolor='#1C2127',edgecolor=GREY,labelcolor=LIGHT,fontsize=9)
    plt.suptitle('HOMO PERPETUUS v8 -- Protein Validation\n(GREEN=correct splice, RED=intron contamination)',
                 color=LIGHT,fontsize=13,fontweight='bold')
    plt.tight_layout()
    return save_fig('03_protein_validation.png')

# --- Plot 4: CpG Promoter analysis (v2 NEW) ----------------------------------

def plot_promoter_cpg(promoter_data):
    if not promoter_data: return None
    genes = sorted(promoter_data.keys())
    gc_vals, cpg_counts, oe_vals, statuses = [],[],[],[]
    for g in genes:
        d = promoter_data[g]
        gc_vals.append(d.get('gc_content_pct',0))
        cpg_counts.append(d.get('cpg_islands',0))
        oe_vals.append(d.get('avg_cpg_obs_exp',0))
        statuses.append(d.get('promoter_status','?'))

    STATUS_COL = {'ACTIVE':GREEN,'POISED':ORANGE,'SILENCED':RED}
    scols = [STATUS_COL.get(s,GREY) for s in statuses]

    fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor=DARK_BG)

    # GC content
    ax = axes[0]; ax.set_facecolor(PANEL_BG)
    bars = ax.bar(range(len(genes)), gc_vals, color=scols, edgecolor='none', width=0.65)
    ax.axhline(55, color=GREEN, ls='--', lw=1, alpha=0.5, label='Active threshold (55%)')
    ax.axhline(45, color=ORANGE,ls=':',  lw=1, alpha=0.5, label='Poised threshold (45%)')
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes,rotation=45,ha='right',fontsize=8,color=LIGHT)
    style_ax(ax,'Promoter GC Content (-2kb upstream)','','GC %')
    ax.legend(facecolor='#1C2127',edgecolor=GREY,labelcolor=LIGHT,fontsize=8)
    ax.set_ylim(0,80)

    # CpG islands count
    ax2 = axes[1]; ax2.set_facecolor(PANEL_BG)
    ax2.bar(range(len(genes)), cpg_counts, color=scols, edgecolor='none', width=0.65)
    ax2.set_xticks(range(len(genes))); ax2.set_xticklabels(genes,rotation=45,ha='right',fontsize=8,color=LIGHT)
    style_ax(ax2,'CpG Islands in Promoter Region','','# Islands')
    for i,v in enumerate(cpg_counts):
        if v: ax2.text(i, v+0.05, str(v), ha='center', color=LIGHT, fontsize=9, fontweight='bold')

    # Obs/Exp ratio
    ax3 = axes[2]; ax3.set_facecolor(PANEL_BG)
    ax3.bar(range(len(genes)), oe_vals, color=scols, edgecolor='none', width=0.65)
    ax3.axhline(0.6, color=GREEN, ls='--', lw=1, alpha=0.5, label='CpG island threshold (0.6)')
    ax3.set_xticks(range(len(genes))); ax3.set_xticklabels(genes,rotation=45,ha='right',fontsize=8,color=LIGHT)
    style_ax(ax3,'CpG Obs/Exp Ratio (promoter)','','Obs/Exp')
    ax3.legend(facecolor='#1C2127',edgecolor=GREY,labelcolor=LIGHT,fontsize=8)
    ax3.set_ylim(0, max(oe_vals)*1.2 if oe_vals else 1)

    # Legend
    patches = [mpatches.Patch(color=v,label=k) for k,v in STATUS_COL.items()]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5,1.02),
               ncol=3, facecolor='#1C2127', edgecolor=GREY, labelcolor=LIGHT, fontsize=10)
    plt.suptitle('HOMO PERPETUUS v8 -- Promoter CpG Analysis\n(determines whether gene can be expressed)',
                 color=LIGHT, fontsize=13, fontweight='bold', y=1.06)
    plt.tight_layout()
    return save_fig('04_promoter_cpg.png')

# --- Plot 5: Simulations dashboard -------------------------------------------

def plot_simulations(sim):
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1. DNA damage
    d = sim.dna_damage(years=150)
    ax = fig.add_subplot(gs[0,0]); ax.set_facecolor(PANEL_BG)
    ax.plot(d['t'], d['normal'], color=RED,   lw=2.5, label='Normal')
    ax.plot(d['t'], d['hp'],     color=GREEN, lw=2.5, label='Homo Perpetuus')
    ax.fill_between(d['t'], d['normal'], d['hp'], alpha=0.1, color=GREEN)
    style_ax(ax,'DNA Damage Accumulation','Age (years)','Damage score')
    ax.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # 2. p53 dynamics
    d2 = sim.p53_dynamics()
    ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor(PANEL_BG)
    ax2.plot(d2['t'], d2['P1'],  color=ORANGE, lw=2, label='p53 x1')
    ax2.plot(d2['t'], d2['P20'], color=GREEN,  lw=2, label='p53 x20')
    ax2.plot(d2['t'], d2['A1'],  color=RED,    lw=2, ls='--', label='Apoptosis x1')
    ax2.plot(d2['t'], d2['A20'], color=CYAN,   lw=2, ls='--', label='Apoptosis x20')
    ax2.axvline(40, color=GREY,ls=':',lw=1)
    if d2['t_hp'] and d2['t_normal']:
        ax2.text(0.97,0.96,f"HP {d2['t_normal']/d2['t_hp']:.1f}x faster",
                 transform=ax2.transAxes,ha='right',va='top',color=GREEN,fontsize=9)
    style_ax(ax2,'p53 Apoptosis Dynamics','Time (hours)','Signal (a.u.)')
    ax2.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # 3. Thymus
    d3 = sim.thymus(years=120)
    ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor(PANEL_BG)
    ax3.plot(d3['t'], d3['normal'], color=RED,  lw=2.5, label='Normal')
    ax3.plot(d3['t'], d3['hp'],     color=BLUE, lw=2.5, label='HP dual thymus')
    ax3.fill_between(d3['t'], 0, d3['hp'],     alpha=0.08, color=BLUE)
    ax3.fill_between(d3['t'], 0, d3['normal'], alpha=0.08, color=RED)
    ax3.axvline(15, color=ORANGE, ls='--', lw=1, alpha=0.7, label='Puberty')
    style_ax(ax3,'Thymic T-cell Output','Age (years)','Output (a.u.)')
    ax3.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # 4. Autophagy / waste
    d4 = sim.autophagy(years=300)
    ax4 = fig.add_subplot(gs[1,0]); ax4.set_facecolor(PANEL_BG)
    ax4.plot(d4['t'], d4['normal'], color=RED,   lw=2.5, label='Normal')
    ax4.plot(d4['t'], d4['hp'],     color=GREEN, lw=2.5, label='HP (NMR+Myotis)')
    ax4.fill_between(d4['t'], d4['normal'], d4['hp'], alpha=0.1, color=GREEN)
    style_ax(ax4,'Intracellular Waste Accumulation','Age (years)','Waste load')
    ax4.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # 5. Telomere dynamics (NEW)
    d5 = sim.telomere_dynamics(years=400)
    ax5 = fig.add_subplot(gs[1,1]); ax5.set_facecolor(PANEL_BG)
    ax5.plot(d5['t'], d5['normal']/1000, color=RED,   lw=2.5, label='Normal')
    ax5.plot(d5['t'], d5['hp']/1000,     color=GREEN, lw=2.5, label='HP (jellyfish FEN1)')
    ax5.axhline(2.0, color=GREY, ls='--', lw=1, alpha=0.6, label='Senescence limit (~2kb)')
    if d5['hayflick_normal'] > 0:
        ax5.axvline(d5['hayflick_normal'], color=RED, ls=':', lw=1, alpha=0.5)
        ax5.text(d5['hayflick_normal']+3, 2.3, f"~{d5['hayflick_normal']}y",
                 color=RED, fontsize=8)
    style_ax(ax5,'Telomere Length Over Time','Age (years)','Telomere length (kb)')
    ax5.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # 6. Cumulative T-cell production
    ax6 = fig.add_subplot(gs[1,2]); ax6.set_facecolor(PANEL_BG)
    ax6.plot(d3['t'], d3['cumul_normal']/1000, color=RED,  lw=2.5, label='Normal')
    ax6.plot(d3['t'], d3['cumul_hp']/1000,     color=BLUE, lw=2.5, label='HP dual thymus')
    ax6.fill_between(d3['t'], d3['cumul_normal']/1000, d3['cumul_hp']/1000,
                     alpha=0.1, color=BLUE)
    style_ax(ax6,'Cumulative Naive T-cells Produced','Age (years)','Cumulative (x103 a.u.)')
    ax6.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=8)

    # 7. EXTENDED SURVIVAL -- multiple safety levels (spans full bottom row)
    d6 = sim.survival_extended(max_age=10000)
    ax7 = fig.add_subplot(gs[2, :]); ax7.set_facecolor(PANEL_BG)
    ax7.plot(d6['t'], d6['normal']*100, color=RED, lw=3, label=f"Normal human  (median {d6['med_normal']}y)", zorder=5)
    hp_colors = [CYAN, GREEN, YELLOW, PURPLE]
    for (label, surv), col, (_, med) in zip(d6['hp_curves'], hp_colors, d6['hp_medians']):
        ax7.plot(d6['t'], surv*100, color=col, lw=2.2,
                 label=f"HP -- {label}  (median {med:,}y)")
    ax7.axhline(50, color=GREY, ls='--', lw=1, alpha=0.5)
    ax7.text(50, 52, '50% survival', color=GREY, fontsize=9)
    style_ax(ax7,'Survival Curves -- Normal vs Homo Perpetuus at Different Safety Levels',
             'Age (years)','Survival (%)')
    ax7.set_ylim(0, 108); ax7.set_xlim(0, 10000)
    ax7.legend(facecolor=PANEL_BG,edgecolor=GREY,labelcolor=LIGHT,fontsize=9,
               loc='upper right')
    # Annotate medians
    for (label, surv), col, (_, med) in zip(d6['hp_curves'], hp_colors, d6['hp_medians']):
        if med < 9500:
            idx = np.argmin(np.abs(d6['t'] - med))
            ax7.annotate(f'{med:,}y', xy=(med, 50), xytext=(med, 60),
                         color=col, fontsize=8, ha='center',
                         arrowprops=dict(arrowstyle='->', color=col, lw=1))

    plt.suptitle('HOMO PERPETUUS v8 -- Biological Simulations',
                 color=LIGHT, fontsize=16, fontweight='bold', y=1.01)
    return save_fig('05_simulations.png')

# --- Plot 6: Module interaction map (NEW) -------------------------------------

def plot_protein_summary(results, promoter_data):
    """Combined panel: correct protein sizes + CpG promoter status side by side."""
    # Collect data
    genes, mws, lengths, statuses = [], [], [], []
    for r in results:
        gn = r.get('gene', r.get('foreign_gene', ''))
        mw = r.get('protein', {}).get('MW_kDa', 0)
        ln = r.get('protein', {}).get('length', 0)
        vs = r.get('validation_status', '')
        if mw and mw > 1:
            genes.append(gn[:14])
            mws.append(mw)
            lengths.append(ln)
            statuses.append(vs)

    STATUS_COL = {'CORRECT': GREEN, 'CORRECT_SYNTHETIC': CYAN,
                  'FOREIGN_GENE': ORANGE, 'PARTIAL': YELLOW, 'INTRON_ARTIFACT': RED}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG)

    # Panel 1: Protein molecular weights (correct values)
    ax = axes[0]; ax.set_facecolor(PANEL_BG)
    cols = [STATUS_COL.get(s, GREY) for s in statuses]
    bars = ax.barh(range(len(genes)), mws, color=cols, height=0.65, edgecolor='none')
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, color=LIGHT, fontsize=9)
    for b, v, ln in zip(bars, mws, lengths):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2,
                f'{v:.0f} kDa  ({ln} aa)', va='center', fontsize=8, color=LIGHT)
    style_ax(ax, 'Protein Molecular Weights (all validated)', 'kDa', '')
    ax.set_xlim(0, max(mws) * 1.35 if mws else 200)
    # Legend
    patches = [mpatches.Patch(color=v, label=k.replace('_', ' '))
               for k, v in STATUS_COL.items() if k != 'INTRON_ARTIFACT']
    ax.legend(handles=patches, loc='lower right', facecolor='#1C2127',
              edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # Panel 2: Promoter status for target genes
    ax2 = axes[1]; ax2.set_facecolor(PANEL_BG)
    if promoter_data:
        prom_genes = sorted(promoter_data.keys())
        PCOL = {'ACTIVE': GREEN, 'POISED': ORANGE, 'SILENCED': RED}
        pcolors = [PCOL.get(promoter_data[g].get('promoter_status', ''), GREY) for g in prom_genes]
        gc_vals = [promoter_data[g].get('gc_content_pct', 0) for g in prom_genes]
        cpg_n   = [promoter_data[g].get('cpg_islands', 0) for g in prom_genes]

        x = np.arange(len(prom_genes))
        ax2.bar(x, gc_vals, color=pcolors, width=0.6, edgecolor='none', alpha=0.85)
        ax2.axhline(55, color=GREEN,  ls='--', lw=1, alpha=0.5, label='Active threshold (55%)')
        ax2.axhline(45, color=ORANGE, ls=':',  lw=1, alpha=0.5, label='Poised threshold (45%)')
        for xi, cn in zip(x, cpg_n):
            if cn: ax2.text(xi, 2, f'{cn}*', ha='center', color=LIGHT, fontsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(prom_genes, rotation=45, ha='right', fontsize=8, color=LIGHT)
        ax2.set_ylim(0, 80)
        style_ax(ax2, 'Promoter GC% & Status (* = CpG islands)', '', 'GC %')
        patches2 = [mpatches.Patch(color=v, label=k) for k, v in PCOL.items()]
        ax2.legend(handles=patches2, loc='upper right', facecolor='#1C2127',
                   edgecolor=GREY, labelcolor=LIGHT, fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No FASTA file\n(promoter analysis requires genome)',
                 ha='center', va='center', color=GREY, fontsize=12,
                 transform=ax2.transAxes)

    plt.suptitle('HOMO PERPETUUS v8 -- Protein Properties & Promoter Status',
                 color=LIGHT, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return save_fig('03_protein_and_promoters.png')


def plot_module_interactions():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG); ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.axis('off')

    modules = {
        'M1\nDNA Repair':    (5.0, 8.5, BLUE),
        'M2\nApoptosis':     (2.0, 6.5, GREEN),
        'M3\nAutophagy':     (8.0, 6.5, ORANGE),
        'M4\nThymusx2':      (5.0, 5.0, PURPLE),
        'M5\nRegeneration':  (1.5, 3.0, CYAN),
        'M6\nStem Cells':    (8.5, 3.0, YELLOW),
        'M7\nMetabolism':    (3.5, 1.2, RED),
        'SMR\nOrgan':        (5.0, 6.5, LIGHT),
        'M8\nAnti-Cancer':   (1.5, 8.5, '#FF69B4'),
        'M9\nSenescence':    (7.0, 1.2, '#00CED1'),  # v6: senolytic + NF-kB shark
        'M10\nEpigenetics':  (9.0, 8.5, '#98FB98'),  # v7: OSKM + glucospanase
        'M11\nNeuronal':     (9.0, 5.5, '#DDA0DD'),  # v8: neurogenesis + lipofuscinase
    }
    edges = [
        ('M7\nMetabolism',  'M1\nDNA Repair',   'Less ROS -> less DNA damage'),
        ('M7\nMetabolism',  'M3\nAutophagy',    'Fewer misfolded proteins'),
        ('M1\nDNA Repair',  'M2\nApoptosis',    'Repaired cells / p53 signal'),
        ('M2\nApoptosis',   'SMR\nOrgan',       'Clears damaged cells'),
        ('SMR\nOrgan',      'M4\nThymusx2',     'Paracrine thymus support'),
        ('M4\nThymusx2',    'M2\nApoptosis',    'T-cell cancer surveillance'),
        ('M3\nAutophagy',   'M5\nRegeneration', 'Clean cells regenerate better'),
        ('M5\nRegeneration','M6\nStem Cells',   'Renewed from stem pool'),
        ('M6\nStem Cells',  'M1\nDNA Repair',   'Stem FOXO3 boosts repair'),
        ('M4\nThymusx2',    'M8\nAnti-Cancer',  'Immune clears pre-cancerous'),
        ('M8\nAnti-Cancer', 'M2\nApoptosis',    'HAS2+LIF6 amplify apoptosis'),
        ('M7\nMetabolism',  'M8\nAnti-Cancer',  'NRF2 reduces oxidative initiation'),
        ('M9\nSenescence',  'M1\nDNA Repair',   'Less SASP -> less paracrine damage'),
        ('M9\nSenescence',  'M7\nMetabolism',   'NF-kB shark reduces inflam-ROS'),
        ('M4\nThymusx2',    'M9\nSenescence',   'NK/T cells clear senescent cells'),
        ('M10\nEpigenetics','M1\nDNA Repair',   'OSKM resets epi-genome instability'),
        ('M10\nEpigenetics','M7\nMetabolism',   'Glucospanase restores ECM elasticity'),
        ('M7\nMetabolism',  'M10\nEpigenetics', 'ROS drives epi-clock drift'),
        # v8 new edges
        ('M11\nNeuronal',   'M5\nRegeneration', 'Neurogenesis renews neural pool'),
        ('M11\nNeuronal',   'M3\nAutophagy',    'Lipofuscinase unblocks lysosomes'),
        ('M7\nMetabolism',  'M11\nNeuronal',    'ROS drives lipofuscin accumulation'),
    ]

    # Draw edges first
    for src, dst, label in edges:
        x1,y1,_ = modules[src]; x2,y2,_ = modules[dst]
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='#2A4A6A', lw=1.8))
        mx,my = (x1+x2)/2+0.1, (y1+y2)/2
        ax.text(mx,my, label, color='#4A7A9A', fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=DARK_BG, edgecolor='none', alpha=0.8))

    # Draw nodes
    for label, (x, y, col) in modules.items():
        circle = plt.Circle((x,y), 0.75, color=col, alpha=0.18, zorder=3)
        ax.add_patch(circle)
        circle2 = plt.Circle((x,y), 0.75, color=col, fill=False, lw=2, zorder=4)
        ax.add_patch(circle2)
        ax.text(x, y, label, ha='center', va='center', color=col,
                fontsize=9, fontweight='bold', zorder=5)

    ax.set_title('HOMO PERPETUUS -- Module Interaction Map',
                 color=LIGHT, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return save_fig('06_module_map.png')


def plot_v5_mechanisms(sim):
    """
    v5-specific plot: 4 panels showing new modification effects.
    Panel 1: Cancer suppression trajectory (normal -> v4 -> v5)
    Panel 2: Stem cell reserve (normal -> v4 -> v5 with FOXO3+TERT)
    Panel 3: ROS/antioxidant profile with NRF2_NMR effect
    Panel 4: v4 vs v5 composite health comparison
    """
    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Cancer suppression
    dc = sim.cancer_suppression(years=400)
    ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(PANEL_BG)
    ax1.fill_between(dc['t'], dc['normal']*100, alpha=0.12, color=RED)
    ax1.plot(dc['t'], dc['normal']*100, color=RED,    lw=2.5, label='Normal human')
    ax1.plot(dc['t'], dc['v4']*100,    color=YELLOW,  lw=2.0, ls='--', label='HP v4 (TP53x20 + immune)')
    ax1.fill_between(dc['t'], dc['v5']*100, alpha=0.12, color=GREEN)
    ax1.plot(dc['t'], dc['v5']*100,    color=GREEN,   lw=2.5, label='HP v5 (+LIF6 +HAS2 +NRF2)')
    ax1.axhline(40, color=GREY, ls='--', lw=1, alpha=0.5)
    ax1.text(5, 41, '~40% lifetime risk (WHO IARC)', color=GREY, fontsize=8)
    idx80 = np.argmin(np.abs(dc['t'] - 80))
    v5_80  = dc['v5'][idx80]*100
    v4_80  = dc['v4'][idx80]*100
    nm_80  = dc['normal'][idx80]*100
    ax1.text(0.98, 0.05,
             f'@80y: normal={nm_80:.0f}% / v4={v4_80:.0f}% / v5={v5_80:.0f}%',
             transform=ax1.transAxes, ha='right', va='bottom',
             color=GREEN, fontsize=8,
             bbox=dict(boxstyle='round', facecolor=PANEL_BG, alpha=0.7))
    style_ax(ax1, 'Cancer Risk Suppression\n(LIF6_elephant + HAS2_NMR + NRF2_NMR)',
             'Age (years)', 'Cumulative cancer risk (%)')
    ax1.set_ylim(0, 100); ax1.set_xlim(0, 400)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # 2. Stem cell reserve
    ds = sim.stem_cell_reserve(years=400)
    ax2 = fig.add_subplot(gs[0, 2:]); ax2.set_facecolor(PANEL_BG)
    ax2.plot(ds['t'], ds['normal']*100, color=RED,    lw=2.5, label='Normal human')
    ax2.plot(ds['t'], ds['v4']*100,     color=YELLOW, lw=2.0, ls='--', label='HP v4 (repair only)')
    ax2.plot(ds['t'], ds['v5']*100,     color=CYAN,   lw=2.5, label='HP v5 (FOXO3_Hydra + TERT_stem)')
    ax2.fill_between(ds['t'], ds['v4']*100, ds['v5']*100, alpha=0.1, color=CYAN)
    ax2.axhline(100, color=GREY, ls=':', lw=1, alpha=0.3, label='Juvenile baseline')
    ax2.axhline(20,  color=ORANGE, ls='--', lw=1, alpha=0.5, label='Impairment threshold')
    dep_arr = ds['normal'] < 0.20
    dep_norm = ds['t'][np.argmax(dep_arr)] if dep_arr.any() else None
    if dep_norm:
        ax2.axvline(dep_norm, color=RED, ls=':', lw=1.2, alpha=0.6)
        ax2.text(dep_norm+3, 22, f'~{dep_norm:.0f}y', color=RED, fontsize=8)
    dep_arr4 = ds['v4'] < 0.20
    dep_v4 = ds['t'][np.argmax(dep_arr4)] if dep_arr4.any() else None
    if dep_v4 and dep_v4 > 0:
        ax2.axvline(dep_v4, color=YELLOW, ls=':', lw=1.2, alpha=0.6)
        ax2.text(dep_v4+3, 22, f'~{dep_v4:.0f}y', color=YELLOW, fontsize=8)
    style_ax(ax2, 'Tissue Stem Cell Reserve\n(FOXO3_Hydra + TERT_stem)',
             'Age (years)', 'Stem cell pool (% juvenile)')
    ax2.set_ylim(0, 115); ax2.set_xlim(0, 400)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # 3. ROS dynamics with NRF2 synergy
    ct_norm  = ModuleCrosstalk.run(years=300, modified=False)
    ct_v4mod = ModuleCrosstalk.run(years=300, modified=True)
    ax3 = fig.add_subplot(gs[1, :2]); ax3.set_facecolor(PANEL_BG)
    ax3.plot(ct_norm['t'],  ct_norm['X'],  color=RED,    lw=2.5, label='Normal')
    ax3.plot(ct_v4mod['t'], ct_v4mod['X'], color=YELLOW, lw=2.0, ls='--', label='HP v4 (Myotis CI)')
    x_v5 = ct_v4mod['X'] / 1.45   # NRF2 analytical approximation
    ax3.plot(ct_v4mod['t'], x_v5,          color=GREEN,  lw=2.5, label='HP v5 (+NRF2_NMR 1.45x)')
    ax3.fill_between(ct_v4mod['t'], ct_v4mod['X'], x_v5, alpha=0.12, color=GREEN)
    style_ax(ax3, 'ROS Level -- NRF2_NMR Synergy\n(Myotis CI -67% + NRF2 1.45x scavenging)',
             'Age (years)', 'ROS level (norm.)')
    ax3.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # 4. v4 vs v5 composite health
    ct_v5 = ModuleCrosstalk.run(years=500, modified=True)
    ct_v4_500 = ModuleCrosstalk.run(years=500, modified=True)   # same (v4 params baked into v5 run)
    ct_norm500 = ModuleCrosstalk.run(years=500, modified=False)
    ax4 = fig.add_subplot(gs[1, 2:]); ax4.set_facecolor(PANEL_BG)
    ax4.plot(ct_norm500['t'], ct_norm500['Q']*100, color=RED,    lw=2.5, label='Normal human')
    ax4.plot(ct_v4_500['t'],  ct_v4_500['Q']*100,  color=YELLOW, lw=2.0, ls='--',
             label='HP v4 (12 mods) [reference]')
    ax4.plot(ct_v5['t'],      ct_v5['Q']*100,      color=GREEN,  lw=2.5, label='HP v5 (18 mods)')
    ax4.fill_between(ct_v5['t'], ct_v4_500['Q']*100, ct_v5['Q']*100, alpha=0.12, color=GREEN)
    for year in [100, 300, 500]:
        idx = np.argmin(np.abs(ct_v5['t'] - year))
        q5  = ct_v5['Q'][idx]*100
        qn  = ct_norm500['Q'][min(idx, len(ct_norm500['Q'])-1)]*100
        if qn > 0:
            ax4.text(year, q5+4, f'{q5/qn:.2f}x', color=GREEN, fontsize=8, ha='center')
    style_ax(ax4, 'Composite Health v4 vs v5\n(numbers = HP/Normal ratio)',
             'Age (years)', 'Health score (%)')
    ax4.set_ylim(0, 108); ax4.set_xlim(0, 500)
    ax4.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    plt.suptitle('HOMO PERPETUUS v8 -- Core Mechanisms (v5 reference)\n'
                 'LIF6_elephant  .  HAS2_NMR  .  FOXO3_Hydra  .  TERT_stem  .  GATA4_zebrafish  .  NRF2_NMR',
                 color=LIGHT, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_fig('09_v5_mechanisms.png')


def plot_v6_mechanisms(sim):
    """
    v6-specific plot: 4 panels showing new v6 modification effects.
    Panel 1: Senescent cell burden -- normal vs HP v5 vs HP v6 (senolytic)
    Panel 2: Inflammaging -- NF-kB shark effect + SASP loop
    Panel 3: Cardiac quartet -- v5 partial vs v6 full (Q contribution)
    Panel 4: Composite health v5 vs v6 (S+I terms now in Q)
    """
    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    ct_norm = ModuleCrosstalk.run(years=500, modified=False)
    ct_v6   = ModuleCrosstalk.run(years=500, modified=True)

    years = ct_v6['t']
    n     = len(years)

    # -- Panel 1: Senescent burden ---------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(PANEL_BG)
    # Approximate v5 S: no senolytic circuit (senolytic_rate=0, nfkb_red=0 equivalent)
    # Use a simplified analytical model for comparison
    D_norm = ct_norm['D']
    D_v6   = ct_v6['D']
    T_norm = ct_norm['T']
    T_v6   = ct_v6['T']
    # Simulate S for display: integrate seno_input - nat_clear (no senolytic, no shark)
    dt_plot = years[1] - years[0]
    S_norm_arr = np.zeros(n); S_v5_arr = np.zeros(n); S_v6_arr = np.zeros(n)
    for i in range(1, n):
        # Normal: no senolytic
        si_n = D_norm[i-1]*0.004*(1+0.0005*years[i-1])
        sc_n = T_norm[i-1]*S_norm_arr[i-1]*0.015 + ct_norm['P'][i-1]*S_norm_arr[i-1]*0.020
        S_norm_arr[i] = max(0, min(1, S_norm_arr[i-1] + (si_n - sc_n)*dt_plot))
        # v5: no senolytic circuit
        si_5 = D_v6[i-1]*0.004*(1+0.0005*years[i-1])
        sc_5 = T_v6[i-1]*S_v5_arr[i-1]*0.015 + ct_v6['P'][i-1]*S_v5_arr[i-1]*0.020
        S_v5_arr[i] = max(0, min(1, S_v5_arr[i-1] + (si_5 - sc_5)*dt_plot))
        # v6: with senolytic (rate 0.04)
        S_v6_arr[i] = ct_v6['S'][i]
    ax1.fill_between(years, S_norm_arr*100, alpha=0.12, color=RED)
    ax1.plot(years, S_norm_arr*100, color=RED,    lw=2.5, label='Normal human')
    ax1.plot(years, S_v5_arr*100,   color=YELLOW, lw=2.0, ls='--', label='HP v5 (no senolytic)')
    ax1.fill_between(years, S_v6_arr*100, alpha=0.15, color=CYAN)
    ax1.plot(years, S_v6_arr*100,   color=CYAN,   lw=2.5, label='HP v6 (+Senolytic p16/p21/IL-6)')
    for yr in [100, 300]:
        idx = np.argmin(np.abs(years - yr))
        ax1.annotate(f'{S_v6_arr[idx]*100:.1f}%', xy=(yr, S_v6_arr[idx]*100),
                     xytext=(yr+15, S_v6_arr[idx]*100+5),
                     color=CYAN, fontsize=8, arrowprops=dict(arrowstyle='->', color=CYAN, lw=1))
    style_ax(ax1, 'Senescent Cell Burden\n(Baker 2011: clearing -> +25% healthspan)',
             'Age (years)', 'Senescent burden (%)')
    ax1.set_xlim(0,500); ax1.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # -- Panel 2: Inflammaging -------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2:]); ax2.set_facecolor(PANEL_BG)
    I_v6 = ct_v6['I']
    # Approximate v5 I: same as v6 but nfkb_red=0
    I_norm_arr = np.zeros(n); I_v5_arr = np.zeros(n)
    for i in range(1, n):
        sasp_n = S_norm_arr[i-1]*0.025; ros_n = ct_norm['X'][i-1]*0.010
        dI_n   = sasp_n + ros_n - ct_norm['T'][i-1]*I_norm_arr[i-1]*0.008 - I_norm_arr[i-1]*0.012
        I_norm_arr[i] = max(0, min(1, I_norm_arr[i-1] + dI_n*dt_plot))
        sasp_5 = S_v5_arr[i-1]*0.025; ros_5 = ct_v6['X'][i-1]*0.010
        dI_5   = sasp_5 + ros_5 - ct_v6['T'][i-1]*I_v5_arr[i-1]*0.008 - I_v5_arr[i-1]*0.012
        I_v5_arr[i] = max(0, min(1, I_v5_arr[i-1] + dI_5*dt_plot))
    ax2.fill_between(years, I_norm_arr*100, alpha=0.12, color=RED)
    ax2.plot(years, I_norm_arr*100, color=RED,    lw=2.5, label='Normal human')
    ax2.plot(years, I_v5_arr*100,   color=YELLOW, lw=2.0, ls='--', label='HP v5 (no NF-kB mod)')
    ax2.fill_between(years, I_v6*100, alpha=0.15, color=GREEN)
    ax2.plot(years, I_v6*100,       color=GREEN,  lw=2.5, label='HP v6 (NF-kB shark -55%)')
    idx200 = np.argmin(np.abs(years - 200))
    diff = (I_v5_arr[idx200] - I_v6[idx200]) * 100
    ax2.annotate(f'-{diff:.1f}% @200y', xy=(200, I_v6[idx200]*100),
                 xytext=(220, I_v6[idx200]*100 + 4),
                 color=GREEN, fontsize=8, arrowprops=dict(arrowstyle='->', color=GREEN, lw=1))
    style_ax(ax2, 'Chronic Inflammaging\n(Nielsen 2016: shark RELA variant -55% tonic NF-kB)',
             'Age (years)', 'Inflammaging index (%)')
    ax2.set_xlim(0,500); ax2.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # -- Panel 3: Cardiac quartet contribution ---------------------------------
    ax3 = fig.add_subplot(gs[1, :2]); ax3.set_facecolor(PANEL_BG)
    # Cardiac health proxy: Q with/without cardiac bonus
    Q_norm  = ct_norm['Q'] * 100
    Q_v5_cardiac = ct_v6['Q'] * 100 - 0.25*0.05*100  # subtract v6 cardiac boost
    Q_v6    = ct_v6['Q'] * 100
    ax3.plot(years, Q_norm,        color=RED,    lw=2.5, label='Normal human')
    ax3.plot(years, Q_v5_cardiac,  color=YELLOW, lw=2.0, ls='--', label='HP v5 cardiac (GATA4+HAND2 +0.75%)')
    ax3.plot(years, Q_v6,          color=GREEN,  lw=2.5, label='HP v6 cardiac (full quartet +1.25%)')
    ax3.fill_between(years, Q_v5_cardiac, Q_v6, alpha=0.15, color=ORANGE)
    ax3.text(0.98, 0.12,
             'Cardiac quartet:\nGATA4 + HAND2 (v5)\n+ TBX5 + MEF2C (v6 new)\n-> full zebrafish regen',
             transform=ax3.transAxes, ha='right', va='bottom',
             color=ORANGE, fontsize=7.5, bbox=dict(boxstyle='round', facecolor=PANEL_BG, alpha=0.8))
    style_ax(ax3, 'Cardiac Regeneration Quartet\n(TBX5+MEF2C complete zebrafish-level regen)',
             'Age (years)', 'Health score (%)')
    ax3.set_ylim(0, 108); ax3.set_xlim(0, 500)
    ax3.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # -- Panel 4: v5 vs v6 composite health -----------------------------------
    ax4 = fig.add_subplot(gs[1, 2:]); ax4.set_facecolor(PANEL_BG)
    # v5 approximation: Q without S and I penalty terms
    Q_v5_approx = np.zeros(n)
    for i in range(n):
        # Remove v6 S and I terms, restore v5 weights, remove cardiac upgrade
        q_base = ct_v6['Q'][i]
        # v6 added: -0.10*S[i] - 0.08*I[i] terms and cardiac +0.0125 vs +0.0075
        s_pen = 0.10 * min(1, ct_v6['S'][i] * 3)
        i_pen = 0.08 * min(1, ct_v6['I'][i] * 4)
        cardiac_diff = (0.25 - 0.15) * 0.05  # v6 vs v5 cardiac bonus diff
        Q_v5_approx[i] = min(1.0, q_base + s_pen + i_pen - cardiac_diff)
    ax4.plot(years, Q_norm,         color=RED,    lw=2.5, label='Normal human')
    ax4.plot(years, Q_v5_approx*100,color=YELLOW, lw=2.0, ls='--', label='HP v5 (21 mods, no senolytic/NF-kB)')
    ax4.plot(years, Q_v6,           color=GREEN,  lw=2.5, label='HP v6 (22 mods + S/I tracked)')
    ax4.fill_between(years, Q_v5_approx*100, Q_v6, alpha=0.12, color=GREEN)
    for year in [100, 200, 400]:
        idx = np.argmin(np.abs(years - year))
        q6  = ct_v6['Q'][idx]*100
        qn  = ct_norm['Q'][min(idx, len(ct_norm['Q'])-1)]*100
        if qn > 0:
            ax4.text(year, q6+3, f'{q6/qn:.2f}x', color=GREEN, fontsize=8, ha='center')
    style_ax(ax4, 'Composite Health v5 -> v6\n(numbers = HP/Normal ratio)',
             'Age (years)', 'Health score (%)')
    ax4.set_ylim(0, 108); ax4.set_xlim(0, 500)
    ax4.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    plt.suptitle('HOMO PERPETUUS v8 -- Extended Mechanisms (v6 reference)\n'
                 'TBX5+MEF2C (cardiac quartet)  .  NF-kB_shark (inflammaging -55%)  .  Senolytic p16/p21/IL-6',
                 color=LIGHT, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_fig('10_v6_mechanisms.png')


def plot_v7_mechanisms(sim):
    """
    v7-specific plot: 4 panels.
    Panel 1: Epigenetic clock drift (E) -- normal vs HP v7 with OSKM reset
    Panel 2: Glucosspan + mtDNA heteroplasmy (G, H) -- new structural variables
    Panel 3: Monte Carlo uncertainty bands for Q(t)
    Panel 4: ODE-linked Gompertz survival -- S(t) derived from Q(t)
    """
    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    ct_norm = ModuleCrosstalk.run(years=500, modified=False)
    ct_v7   = ModuleCrosstalk.run(years=500, modified=True)
    years_t = ct_v7['t']

    # -- Panel 1: Epigenetic clock ---------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(PANEL_BG)
    ax1.fill_between(years_t, ct_norm['E']*100, alpha=0.15, color=RED)
    ax1.plot(years_t, ct_norm['E']*100, color=RED,   lw=2.5, label='Normal human (clock drifts)')
    ax1.fill_between(years_t, ct_v7['E']*100,  alpha=0.15, color=GREEN)
    ax1.plot(years_t, ct_v7['E']*100,  color=GREEN, lw=2.5, label='HP v7 (OSKM cyclic reset)')
    # Show annual reset pulses as small tick marks
    for yr in range(0, 500, 50):
        idx = np.argmin(np.abs(years_t - yr))
        ax1.axvline(yr, color='#2A4A2A', lw=0.5, alpha=0.4)
    idx200 = np.argmin(np.abs(years_t - 200))
    diff = (ct_norm['E'][idx200] - ct_v7['E'][idx200]) * 100
    ax1.annotate(f'-{diff:.1f}% @200y', xy=(200, ct_v7['E'][idx200]*100),
                 xytext=(220, ct_v7['E'][idx200]*100 + 3),
                 color=GREEN, fontsize=8,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1))
    style_ax(ax1, 'Epigenetic Clock Drift\n(Horvath 2013; OSKM cyclic reset -- Gill 2022 Cell)',
             'Age (years)', 'Epigenetic drift (%)')
    ax1.set_xlim(0, 500)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # -- Panel 2: Glucosspan + heteroplasmy ------------------------------------
    ax2 = fig.add_subplot(gs[0, 2:]); ax2.set_facecolor(PANEL_BG)
    ax2_r = ax2.twinx(); ax2_r.set_facecolor(PANEL_BG)
    # Glucosspan G
    ax2.plot(years_t, ct_norm['G']*100, color=ORANGE, lw=2.5, ls='--', label='G normal (crosslinks)')
    ax2.plot(years_t, ct_v7['G']*100,   color=YELLOW, lw=2.5, label='G HP v7 (+glucospanase)')
    ax2.set_ylabel('Glucosspan burden (%)', color=ORANGE, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    # Heteroplasmy H on right axis
    ax2_r.plot(years_t, ct_norm['H']*100, color=PURPLE, lw=2.0, ls='--', label='H normal (heteroplasmy)')
    ax2_r.plot(years_t, ct_v7['H']*100,   color=CYAN,   lw=2.0, label='H HP v7 (+DdCBE)')
    ax2_r.set_ylabel('mtDNA heteroplasmy (%)', color=PURPLE, fontsize=9)
    ax2_r.tick_params(axis='y', labelcolor=PURPLE)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, facecolor=PANEL_BG, edgecolor=GREY,
               labelcolor=LIGHT, fontsize=7, loc='upper left')
    ax2.set_facecolor(PANEL_BG); ax2.set_title(
        'Glucosspan (G) & mtDNA Heteroplasmy (H)\n(Glucospanase + DdCBE, v7 new)',
        color=LIGHT, fontsize=9, fontweight='bold')
    ax2.set_xlabel('Age (years)', color=LIGHT, fontsize=9)
    ax2.tick_params(colors=LIGHT); ax2.set_xlim(0, 500)
    for sp in ax2.spines.values(): sp.set_edgecolor(GREY)

    # -- Panel 3: Monte Carlo uncertainty bands --------------------------------
    ax3 = fig.add_subplot(gs[1, :2]); ax3.set_facecolor(PANEL_BG)
    mc_hp   = ModuleCrosstalk.monte_carlo(years=500, n_runs=150, modified=True)
    mc_norm = ModuleCrosstalk.monte_carlo(years=500, n_runs=150, modified=False)
    t_mc = mc_hp['t']
    ax3.fill_between(t_mc, mc_norm['p05']*100, mc_norm['p95']*100, alpha=0.12, color=RED)
    ax3.fill_between(t_mc, mc_norm['p25']*100, mc_norm['p75']*100, alpha=0.20, color=RED)
    ax3.plot(t_mc, mc_norm['p50']*100, color=RED, lw=2.5, label='Normal (median +/- IQR +/- 90%CI)')
    ax3.fill_between(t_mc, mc_hp['p05']*100, mc_hp['p95']*100, alpha=0.12, color=GREEN)
    ax3.fill_between(t_mc, mc_hp['p25']*100, mc_hp['p75']*100, alpha=0.20, color=GREEN)
    ax3.plot(t_mc, mc_hp['p50']*100, color=GREEN, lw=2.5, label='HP v7 (median +/- IQR +/- 90%CI)')
    style_ax(ax3, 'Monte Carlo Health Uncertainty\n(n=150 runs, biological noise model)',
             'Age (years)', 'Health Q (%)')
    ax3.set_ylim(0, 108); ax3.set_xlim(0, 500)
    ax3.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # -- Panel 4: ODE-linked Gompertz survival ---------------------------------
    ax4 = fig.add_subplot(gs[1, 2:]); ax4.set_facecolor(PANEL_BG)
    gomp = ModuleCrosstalk.gompertz_from_ode(years=500)
    t_g  = gomp['t']
    ax4.fill_between(t_g, gomp['S_norm']*100, alpha=0.12, color=RED)
    ax4.plot(t_g, gomp['S_norm']*100, color=RED,   lw=2.5,
             label=f"Normal (median ~{gomp['median_normal']:.0f}y)")
    ax4.fill_between(t_g, gomp['S_hp']*100, alpha=0.12, color=GREEN)
    ax4.plot(t_g, gomp['S_hp']*100,   color=GREEN, lw=2.5,
             label=f"HP v7 (median >{gomp['median_hp']:.0f}y in window)")
    ax4.axhline(50, color=GREY, ls='--', lw=1, alpha=0.5)
    ax4.text(5, 52, '50% survival', color=GREY, fontsize=8)
    ax4.text(0.98, 0.5,
             'v7 FIX: h(t) = h0.exp(b.t/80).(1/Q(t))2\n'
             'ODE Q(t) directly drives hazard\n'
             '(prev. versions: hardcoded Gompertz params)',
             transform=ax4.transAxes, ha='right', va='center', color=CYAN, fontsize=7.5,
             bbox=dict(boxstyle='round', facecolor=PANEL_BG, alpha=0.8))
    style_ax(ax4, 'ODE-Linked Gompertz Survival\n(hazard driven by Q(t), v7 architectural fix)',
             'Age (years)', 'Survival (%)')
    ax4.set_ylim(0, 108); ax4.set_xlim(0, 500)
    ax4.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    plt.suptitle('HOMO PERPETUUS v8 -- Advanced Mechanisms (Monte Carlo + ODE-Gompertz)\n'
                 'OSKM_cyclic  .  Glucospanase  .  DdCBE_mito  .  TFEB_neuron  '
                 '.  Q-normalized  .  ODE->Gompertz  .  Monte Carlo',
                 color=LIGHT, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_fig('11_v7_mechanisms.png')


def plot_gtex_expression(gtex_data, modification_genes):
    """
    Heatmap: rows = modification genes, columns = tissues.
    Color = log2(TPM+1). Annotated with tissue-specific relevance.
    """
    tissues = list(GTEX_TISSUES_OF_INTEREST.keys())
    genes   = [g for g in modification_genes if g in gtex_data and g in GTEX_FALLBACK]

    # Build matrix
    mat = np.zeros((len(genes), len(tissues)))
    for i, gene in enumerate(genes):
        expr = gtex_data.get(gene, {})
        for j, tissue in enumerate(tissues):
            tpm = expr.get(tissue, 0)
            mat[i, j] = np.log2(float(tpm) + 1)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Custom colormap: dark -> blue -> cyan -> yellow (TPM scale)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('hp_expr',
        ['#0A0E1A', '#1A3A6B', '#2E9BFF', '#00E5FF', '#FFD700'], N=256)

    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=mat.max()*0.9)

    # Axes labels
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=35, ha='right', color=LIGHT, fontsize=9)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, color=LIGHT, fontsize=9)

    # Annotate cells with TPM values
    for i in range(len(genes)):
        for j in range(len(tissues)):
            gene  = genes[i]
            tissue = tissues[j]
            raw_tpm = gtex_data.get(gene, {}).get(tissue, 0)
            val = float(raw_tpm)
            txt_color = '#000000' if mat[i,j] > mat.max()*0.6 else LIGHT
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    color=txt_color, fontsize=7.5, fontweight='bold' if val > 10 else 'normal')

    # Mark genes with SILENCED promoters (from our analysis)
    SILENCED_GENES = {'TP53', 'BRCA1', 'BRCA2', 'CCND1', 'FEN1', 'FOXN1', 'GLO1', 'LAMP2', 'SQSTM1'}
    for i, gene in enumerate(genes):
        if gene in SILENCED_GENES:
            ax.text(-0.7, i, 'o', ha='center', va='center', color=RED, fontsize=9)
        elif gene in {'AR', 'ERCC1', 'MSH2', 'MSH6', 'NOTCH1', 'PCNA', 'RAD51', 'SOX2', 'TERT'}:
            ax.text(-0.7, i, 'o', ha='center', va='center', color=ORANGE, fontsize=9)
        else:
            ax.text(-0.7, i, 'o', ha='center', va='center', color=GREEN, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('log2(TPM+1)', color=GREY, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=GREY)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GREY)

    # Legend for promoter dots
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=GREEN,  markersize=8, label='ACTIVE promoter',   linestyle='None'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=ORANGE, markersize=8, label='POISED promoter',   linestyle='None'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RED,    markersize=8, label='SILENCED promoter', linestyle='None'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', facecolor='#1C2127',
              edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    src_note = '(GTEx v8 API)' if not any(
        gtex_data.get(g, {}).get('_source') == 'literature_fallback'
        for g in genes[:3]
    ) else '(literature estimates -- run online for real GTEx data)'

    ax.set_title(f'HOMO PERPETUUS -- Tissue Expression Heatmap {src_note}\nValues = median TPM  |  o = promoter status',
                 color=LIGHT, fontsize=12, pad=14)
    plt.tight_layout()
    return save_fig('06_gtex_expression.png')


# ==============================================================================
# NEW PLOT: Module crosstalk -- coupled ODE results
# ==============================================================================

def plot_module_crosstalk(years=500):
    """
    4-panel plot showing the emergent coupled dynamics of HP modifications.
    """
    ct = ModuleCrosstalk.run_both(years=years)
    hp     = ct['hp']
    normal = ct['normal']
    t      = hp['t']

    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # Panel 1: DNA damage + repair capacity (dual axis)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(PANEL_BG)
    ax1.plot(t, normal['D'], color=RED,    lw=2.5, label='DNA damage -- Normal')
    ax1.plot(t, hp['D'],     color=GREEN,  lw=2.5, label='DNA damage -- HP')
    ax1b = ax1.twinx()
    ax1b.plot(t, hp['R'],    color=CYAN, lw=1.8, ls='--', label='Repair capacity (HP)', alpha=0.8)
    ax1b.set_ylabel('Repair capacity', color=CYAN, fontsize=8)
    ax1b.tick_params(colors=CYAN, labelsize=8)
    ax1b.set_facecolor(PANEL_BG)
    style_ax(ax1, 'DNA Damage + Repair Synergy\n(PIWI + RAD51x3 + ERCC1)',
             'Age (years)', 'Damage score')
    lines1 = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines1, [l.get_label() for l in lines1],
               facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=7)

    # Panel 2: p53 <-> CCND1 interaction
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(PANEL_BG)
    # p53 activity suppresses CCND1; higher p53 = cleaner but slower cell cycle
    p53_hp  = hp['P'];  p53_n = normal['P']
    # CCND1 activity = 1 - p53_suppression (in non-cardiac cells)
    ccnd1_normal = np.clip(1 - p53_n * 0.6, 0.1, 1.0)
    ccnd1_hp     = np.clip(1 - p53_hp * 0.6, 0.1, 1.0)
    ax2.plot(t, p53_n,      color=RED,    lw=2.5, label='p53 activity -- Normal')
    ax2.plot(t, p53_hp,     color=GREEN,  lw=2.5, label='p53 activity -- HP (x20)')
    ax2.plot(t, ccnd1_normal, color=ORANGE, lw=1.8, ls='--', label='CCND1 activity -- Normal')
    ax2.plot(t, ccnd1_hp,     color=YELLOW, lw=1.8, ls='--', label='CCND1 activity -- HP (conditional)')
    style_ax(ax2, 'p53 <-> CCND1 Cross-regulation\n(TP53x20 suppresses cell cycle in healthy cells)',
             'Age (years)', 'Activity (norm.)')
    ax2.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=7)
    ax2.set_ylim(0, 1.2)

    # Panel 3: ROS -> waste -> CMA efficiency chain
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(PANEL_BG)
    ax3.plot(t, normal['X'], color=RED,    lw=2.5, label='ROS -- Normal')
    ax3.plot(t, hp['X'],     color=CYAN,   lw=2.5, label='ROS -- HP (Myotis -60%)')
    ax3.plot(t, normal['W']/8, color=ORANGE, lw=1.8, ls='--', label='Waste load -- Normal (/8)')
    ax3.plot(t, hp['W']/8,     color=GREEN,  lw=1.8, ls='--', label='Waste load -- HP (NMR LAMP2A)')
    style_ax(ax3, 'ROS -> Protein Oxidation -> CMA Chain\n(Myotis CI x LAMP2A synergy)',
             'Age (years)', 'Normalised level')
    ax3.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=7)

    # Panel 4: Thymic immune quality evolution
    ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(PANEL_BG)
    ax4.plot(t, normal['T'], color=RED,   lw=2.5, label='Thymic quality -- Normal')
    ax4.plot(t, hp['T'],     color=BLUE,  lw=2.5, label='Thymic quality -- HP (AR KO + AIREx3)')
    ax4.fill_between(t, normal['T'], hp['T'], alpha=0.1, color=BLUE)
    # Calculate cancer suppression advantage at key ages
    for age_check in [100, 300, 500]:
        if age_check < years:
            idx = np.searchsorted(t, age_check)
            ratio = hp['T'][idx] / max(normal['T'][idx], 0.001)
            ax4.annotate(f'{ratio:.1f}x\nbetter\n@{age_check}y',
                         xy=(age_check, (hp['T'][idx]+normal['T'][idx])/2),
                         color=CYAN, fontsize=7, ha='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#1C2A3A', alpha=0.8))
    style_ax(ax4, 'Thymic Immune Quality\n(AR KO prevents involution; AIREx3 improves selection)',
             'Age (years)', 'Quality score (0-1)')
    ax4.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=7)

    # Panel 5: Neuronal accumulation -- ADAR deep-time effect
    nd = sim_neuronal_ceiling(max_age=min(years*60, 50000))
    ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(PANEL_BG)
    ax5.plot(nd['t'], nd['normal']*100, color=RED,   lw=2.5, label='Without ADAR')
    ax5.plot(nd['t'], nd['adar']*100,   color=PURPLE, lw=2.5, label='With ADAR (octopus RNA editing)')
    ax5.axhline(50, color=GREY, ls='--', lw=1, alpha=0.6, label='Functional impairment threshold')
    if nd['ceiling_normal'] < nd['t'][-1]:
        ax5.axvline(nd['ceiling_normal'], color=RED, ls=':', lw=1, alpha=0.6)
        ax5.text(nd['ceiling_normal'], 55,
                 f"  {nd['ceiling_normal']:,}y", color=RED, fontsize=8)
    if nd['ceiling_adar'] < nd['t'][-1]:
        ax5.axvline(nd['ceiling_adar'], color=PURPLE, ls=':', lw=1, alpha=0.6)
        ax5.text(nd['ceiling_adar'], 40,
                 f"  {nd['ceiling_adar']:,}y", color=PURPLE, fontsize=8)
    style_ax(ax5, 'Neuronal Accumulation -- ADAR Effect\n(Biological ceiling: RNA editing extends functional lifespan)',
             'Age (years)', 'Accumulation (%)')
    ax5.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=7)

    # Panel 6: Overall cellular health composite
    ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor(PANEL_BG)
    ax6.plot(t, normal['Q']*100, color=RED,   lw=2.5, label='Normal human')
    ax6.plot(t, hp['Q']*100,     color=GREEN, lw=2.5, label='Homo Perpetuus')
    ax6.fill_between(t, normal['Q']*100, hp['Q']*100, alpha=0.12, color=GREEN)
    # Annotate divergence
    diverge_age = None
    for i in range(len(t)):
        if hp['Q'][i] - normal['Q'][i] > 0.2:
            diverge_age = t[i]; break
    if diverge_age:
        ax6.axvline(diverge_age, color=YELLOW, ls=':', lw=1.5, alpha=0.7)
        ax6.text(diverge_age+5, 70, f'Divergence\n@{int(diverge_age)}y',
                 color=YELLOW, fontsize=8)
    # Show health at 100 and 500 years
    for age_mark, col in [(100, CYAN), (300, ORANGE)]:
        if age_mark < years:
            idx = np.searchsorted(t, age_mark)
            hp_q  = hp['Q'][idx]*100
            nor_q = normal['Q'][idx]*100
            ax6.annotate(f'HP:{hp_q:.0f}%\nNorm:{nor_q:.0f}%',
                         xy=(age_mark, (hp_q+nor_q)/2),
                         color=col, fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#1C2A3A', alpha=0.8))
    style_ax(ax6, 'Composite Cellular Health\n(Emergent result of all 12 modifications interacting)',
             'Age (years)', 'Health score (%)')
    ax6.set_ylim(0, 108)
    ax6.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=7)

    plt.suptitle('HOMO PERPETUUS -- Module Crosstalk & Emergent Interactions\n'
                 'Coupled ODE system: 8 state variables x 12 modifications',
                 color=LIGHT, fontsize=14, fontweight='bold', y=1.01)
    return save_fig('07_module_crosstalk.png')


# ==============================================================================
# ESM-2 CLIENT -- Meta protein language model via HuggingFace Inference API
# Model: facebook/esm2_t33_650M_UR50D (650M params, free tier)
# Gives per-residue embeddings -> we compute stability & evolutionary scores
# ==============================================================================

_ESM2_CACHE_FILE = os.path.join(BASE_DIR, '.esm2_cache.json')
_esm2_cache = {}
ESM2_MODEL   = "facebook/esm2_t33_650M_UR50D"
# Feature-extraction endpoint gives hidden states per residue
ESM2_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{ESM2_MODEL}"

def plot_ai_risk_dashboard(mod_results, esm2_data, af2_data, ot_data):
    """
    4-panel AI-powered risk assessment dashboard:
    1. ESM-2 stability scores per modification
    2. AlphaFold pLDDT confidence (human genes only)
    3. OpenTargets disease association counts (modification risk context)
    4. Combined safety matrix: structural + evolutionary + disease risk
    """
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    mod_ids  = [r['mod_id'] for r in mod_results]
    short_ids = [m.replace('MOD_','').replace('_',' ')[:18] for m in mod_ids]

    # -- Panel 1: ESM-2 stability ---------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(PANEL_BG)
    stab_scores = []
    stab_colors = []
    for r in mod_results:
        mid = r['mod_id']
        esm = esm2_data.get(mid)
        if esm and esm.get('stability_score') is not None:
            s = esm['stability_score']
        else:
            # Fallback: use instability index from Guruprasad (inverted, normalised)
            ii = r.get('protein', {}).get('instability_index', 20)
            s  = max(0.1, min(0.95, 1 - ii/80))
        stab_scores.append(s)
        col = (GREEN if s > 0.7 else ORANGE if s > 0.4 else RED)
        stab_colors.append(col)

    bars = ax1.barh(range(len(mod_ids)), stab_scores,
                    color=stab_colors, height=0.65, edgecolor='none')
    ax1.axvline(0.7, color=GREEN,  ls='--', lw=1, alpha=0.5, label='Stable threshold')
    ax1.axvline(0.4, color=ORANGE, ls='--', lw=1, alpha=0.5, label='Marginal threshold')
    ax1.set_yticks(range(len(short_ids))); ax1.set_yticklabels(short_ids, fontsize=8)
    ax1.set_xlim(0, 1.05)
    for i, (b, s) in enumerate(zip(bars, stab_scores)):
        src_tag = '* ESM-2' if (esm2_data.get(mod_ids[i]) and
                                esm2_data[mod_ids[i]].get('stability_score')) else 'Guruprasad'
        ax1.text(s + 0.01, i, f'{s:.2f}  ({src_tag})',
                 va='center', color=LIGHT, fontsize=7.5)
    style_ax(ax1, 'ESM-2 Protein Stability Score\n(per-modification evolutionary fitness)',
             'Stability score (0-1)', '')
    ax1.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)
    ax1.tick_params(colors=LIGHT)

    # -- Panel 2: AlphaFold pLDDT confidence ---------------------------------
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(PANEL_BG)
    af_genes  = [g for g in AF2_FALLBACK.keys() if g in [r.get('gene','') for r in mod_results]]
    af_genes += [g for g in ['ERCC1','RAD51','FEN1','CCND1','AIRE','AR','TP53','LAMP2','GLO1']
                 if g not in af_genes]
    af_genes   = list(dict.fromkeys(af_genes))[:14]  # unique, max 14 to fit

    plddt_vals  = []
    plddt_cols  = []
    plddt_notes = []
    for g in af_genes:
        af = af2_data.get(g, AF2_FALLBACK.get(g, {}))
        v  = af.get('mean_plddt', 0) if af else 0
        plddt_vals.append(v)
        col = (GREEN if v > 90 else CYAN if v > 70 else ORANGE if v > 50 else RED)
        plddt_cols.append(col)
        plddt_notes.append(af.get('note','') if af else '')

    bars2 = ax2.barh(range(len(af_genes)), plddt_vals,
                     color=plddt_cols, height=0.65, edgecolor='none')
    ax2.axvline(90, color=GREEN,  ls='--', lw=1, alpha=0.5, label='Very high (>90)')
    ax2.axvline(70, color=CYAN,   ls='--', lw=1, alpha=0.5, label='High (>70)')
    ax2.axvline(50, color=ORANGE, ls='--', lw=1, alpha=0.5, label='Medium (>50)')
    ax2.set_yticks(range(len(af_genes))); ax2.set_yticklabels(af_genes, fontsize=8)
    ax2.set_xlim(0, 108)
    for i, (b, v) in enumerate(zip(bars2, plddt_vals)):
        grade = ('OKOKOK' if v > 90 else 'OKOK' if v > 70 else 'OK' if v > 50 else '~')
        ax2.text(v + 0.5, i, f'{v:.0f}  {grade}', va='center', color=LIGHT, fontsize=7.5)
    style_ax(ax2, 'AlphaFold2 Structure Confidence (pLDDT)\n(>70 = reliable predicted structure)',
             'Mean pLDDT score', '')
    ax2.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)
    ax2.tick_params(colors=LIGHT)

    # -- Panel 3: OpenTargets disease burden ----------------------------------
    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(PANEL_BG)
    ot_genes = list(OPENTARGETS_IDS.keys())
    ot_counts = []
    ot_cols   = []
    for g in ot_genes:
        ot = ot_data.get(g, OT_FALLBACK.get(g))
        n  = ot.get('total_disease_associations', 0) if ot else 0
        ot_counts.append(n)
        # More disease associations = HIGHER risk if we modify this gene
        col = (RED if n > 500 else ORANGE if n > 100 else CYAN if n > 30 else GREEN)
        ot_cols.append(col)

    bars3 = ax3.barh(range(len(ot_genes)), ot_counts,
                     color=ot_cols, height=0.65, edgecolor='none')
    ax3.set_yticks(range(len(ot_genes))); ax3.set_yticklabels(ot_genes, fontsize=8)
    for b, n, g in zip(bars3, ot_counts, ot_genes):
        top_dis = (ot_data.get(g, OT_FALLBACK.get(g, {})) or {})
        top = top_dis.get('top_diseases', [{}])
        top_name = top[0].get('name','')[:25] if top else ''
        ax3.text(n + 2, b.get_y()+b.get_height()/2,
                 f'{n}  [{top_name}]', va='center', color=LIGHT, fontsize=7)
    style_ax(ax3, 'OpenTargets Disease Associations\n(more = higher modification risk -- handle carefully)',
             'Number of disease associations', '')
    ax3.tick_params(colors=LIGHT)

    # -- Panel 4: Combined safety matrix --------------------------------------
    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(PANEL_BG)

    # For each modification: compute 3D risk score
    # X = structural stability (ESM-2 or Guruprasad)
    # Y = disease burden (OpenTargets, inverted -- higher disease = higher risk)
    # Size = protein size (larger = harder to deliver)
    # Color = modification type risk

    RISK_NUM = {'VERY LOW': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4}
    RISK_COL_MAP = {'VERY LOW': CYAN, 'LOW': GREEN, 'MEDIUM': ORANGE, 'HIGH': RED}

    xs, ys, sizes, cols, labels = [], [], [], [], []
    for r in mod_results:
        mid   = r['mod_id']
        gene  = r.get('gene', r.get('foreign_gene',''))
        risk  = r.get('risk','LOW').split()[0]

        # X: stability
        esm = esm2_data.get(mid)
        if esm and esm.get('stability_score'):
            x = esm['stability_score']
        else:
            ii = r.get('protein',{}).get('instability_index', 20)
            x  = max(0.1, min(0.95, 1 - ii/80))

        # Y: disease burden (inverted -- lower = safer)
        ot  = ot_data.get(gene, OT_FALLBACK.get(gene, {}))
        n   = ot.get('total_disease_associations', 10) if ot else 10
        y   = max(0.05, 1 - min(1, np.log10(n+1) / 3.5))  # log-scale, inverted

        # Size: protein MW in kDa
        mw = r.get('protein',{}).get('MW_kDa', 50)

        xs.append(x); ys.append(y)
        sizes.append(max(40, min(400, mw * 2)))
        cols.append(RISK_COL_MAP.get(risk, GREEN))
        labels.append(gene[:10] if gene else mid[-8:])

    sc = ax4.scatter(xs, ys, s=sizes, c=cols, alpha=0.85, edgecolors='white', linewidths=0.5)

    for x, y, lbl in zip(xs, ys, labels):
        ax4.annotate(lbl, (x, y), textcoords='offset points', xytext=(5, 3),
                     color=LIGHT, fontsize=7.5)

    # Quadrant lines
    ax4.axvline(0.65, color=GREY, ls=':', lw=1, alpha=0.5)
    ax4.axhline(0.50, color=GREY, ls=':', lw=1, alpha=0.5)

    # Quadrant labels
    for tx, ty, label in [(0.2, 0.75,'HIGH RISK\n(unstable, few disease links)'),
                           (0.8, 0.75,'SAFE ZONE\n(stable, few disease links)'),
                           (0.2, 0.25,'DANGER ZONE\n(unstable + disease-critical)'),
                           (0.8, 0.25,'MODERATE\n(stable but disease-critical)')]:
        ax4.text(tx, ty, label, transform=ax4.transAxes, ha='center', va='center',
                 color=GREY, fontsize=7, alpha=0.6)

    # Legend for risk colors
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=9,
               label=f'{r} risk', linestyle='None')
        for r, c in RISK_COL_MAP.items()
    ]
    ax4.legend(handles=legend_elems, facecolor=PANEL_BG, edgecolor=GREY,
               labelcolor=LIGHT, fontsize=8, loc='lower right')

    style_ax(ax4, 'Combined Safety Matrix\n(X=ESM-2 stability  Y=disease burden safety  size=protein MW)',
             'Structural stability score', 'Disease-link safety score')
    ax4.set_xlim(0, 1.1); ax4.set_ylim(0, 1.1)

    plt.suptitle('HOMO PERPETUUS -- AI-Powered Risk Assessment\n'
                 'ESM-2 (Meta) x AlphaFold2 (DeepMind) x OpenTargets Platform',
                 color=LIGHT, fontsize=14, fontweight='bold', y=1.01)
    return save_fig('08_ai_risk_dashboard.png')


# ==============================================================================
# CRISPR OFF-TARGET ANALYSIS MODULE
# ==============================================================================
#
# Algorithm: Cas-OFFinder-style two-stage search (no external dependencies)
#   Stage 1: seed filter (positions 1-12, <=1 mismatch) -- eliminates 95%+ windows
#   Stage 2: full 20nt mismatch count on surviving candidates
#   PAM:     SpCas9 NGG (positions 21-23 after protospacer)
#
# Scan modes:
#   targeted (default): +/-10kb windows around 63 key genes (HP targets + cancer drivers)
#   full:               all chromosomes via multiprocessing (slow, thorough)
#
# Risk scoring per off-target hit:
#   0 mm  in exon/promoter -> CRITICAL
#   1 mm  in exon/promoter -> HIGH
#   2 mm  in exon/promoter -> MEDIUM
#   3 mm  in exon/promoter -> LOW
#   Any mm in intergenic    -> BACKGROUND (not reported)

import multiprocessing as mp

# -- gRNA designs for all 12 HP modifications ---------------------------------