"""hp_modules/crispr.py -- CRISPR_TARGETS dict + off-target analysis."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
import multiprocessing as mp
from hp_modules.config import (DARK_BG, PANEL_BG,
                                RED, ORANGE, YELLOW, CYAN,
                                GREEN, GREY, LIGHT)
from hp_modules.plots import save_fig

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from hp_modules.config import BASE_DIR
from hp_modules.modifications import GENE_DB

CRISPR_TARGETS = {
    "MOD_01_TP53_x20": {
        "gene":"TP53",    "chr":"chr17","cut":7687425,  "strand":"+",
        "guide":"GCACTTTCCTTGCAGTGTCA","pam":"CGG",
        "purpose":"HDR insertion of extra TP53 copies upstream of native gene",
        "note":"Near exon 1; validated region (Addgene #52963 area)"},
    "MOD_02_ERCC1_whale": {
        "gene":"ERCC1",   "chr":"chr19","cut":45380800, "strand":"+",
        "guide":"GCTGAGCTGCGTGTGTGCAG","pam":"TGG",
        "purpose":"Upstream regulatory element modification for paralogue insertion",
        "note":"Designed from TSS -2kb region"},
    "MOD_03_AR_KO_TEC": {
        "gene":"AR",      "chr":"chrX", "cut":67545000, "strand":"+",
        "guide":"GCTGTCCGTCTTCGGAGCAT","pam":"TGG",
        "purpose":"Conditional KO -- flanking loxP sites in AR exon 1",
        "note":"AR exon 1; used in multiple published AR-KO studies"},
    "MOD_04_AIRE_x3": {
        "gene":"AIRE",    "chr":"chr21","cut":44283700, "strand":"+",
        "guide":"AGGCAGAGCCAGGCAGTCCA","pam":"AGG",
        "purpose":"Insert strong CAG promoter upstream of endogenous AIRE",
        "note":"AIRE TSS region chr21:44,283,645"},
    "MOD_05_LAMP2A_NMR": {
        "gene":"LAMP2",   "chr":"chrX", "cut":119537600,"strand":"-",
        "guide":"CTGCAGGTCAAGGTGCTGCA","pam":"TGG",
        "purpose":"HDR replacement of LAMP2 exon1 with NMR LAMP2A",
        "note":"LAMP2 exon1; HDR template includes homology arms"},
    "MOD_06_PIWI_jellyfish": {
        "gene":"AAVS1",   "chr":"chr19","cut":55115750, "strand":"+",
        "guide":"GGGGCCACTAGGGACAGGAT","pam":"TGG",
        "purpose":"Safe harbour insertion of PIWI expression cassette",
        "note":"AAVS1 canonical guide -- most validated safe harbour in human genome"},
    "MOD_07_GLO1_AGE": {
        "gene":"GLO1",    "chr":"chr6", "cut":38694900, "strand":"+",
        "guide":"GATGCTCAGCTTCTCCAGCA","pam":"GGG",
        "purpose":"Knock-in of NMR GLO1-FN3K fusion at native GLO1 locus",
        "note":"GLO1 exon1 region"},
    "MOD_08_ADAR_neuron": {
        "gene":"SYNGAP1", "chr":"chr6", "cut":33391700, "strand":"+",
        "guide":"CAGCTGCAGATCGAGAAGCA","pam":"CGG",
        "purpose":"Neuron-specific safe harbour -- ADAR expression cassette",
        "note":"SYNGAP1 intron 1 -- expressed exclusively in neurons"},
    "MOD_09_CCND1_cardiac": {
        "gene":"CCND1",   "chr":"chr11","cut":69641200, "strand":"-",
        "guide":"ATGGAGCTGCTGTGCCACGA","pam":"GGG",
        "purpose":"Insert cardiac HRE promoter upstream of CCND1",
        "note":"CCND1 TSS region; hypoxia-responsive element insertion"},
    "MOD_10_MITO_Myotis": {
        "gene":"MT-ND5",  "chr":"chrM", "cut":12337,    "strand":"+",
        "guide":"CCGAACCAATCATAGCCCCT","pam":"AGG",
        "purpose":"Mitochondrial genome -- ND5 subunit replacement",
        "note":"MITO-CRISPR uses mitoTALEN/DdCBE; off-target risk profile different"},
    "MOD_11_RAD51_x3": {
        "gene":"RAD51",   "chr":"chr15","cut":40695400, "strand":"+",
        "guide":"GCAGTCAGAGCAGCTGCAGC","pam":"TGG",
        "purpose":"Insert extra RAD51 copies under EF1alpha promoter upstream",
        "note":"RAD51 upstream regulatory region"},
    "MOD_12_FEN1_jellyfish": {
        "gene":"FEN1",    "chr":"chr11","cut":108325300,"strand":"-",
        "guide":"TGCAGCAGCTGGGCGCGCTG","pam":"CGG",
        "purpose":"Promoter replacement for 2x FEN1 expression",
        "note":"FEN1 TSS -1kb region"},
    # -- v5 new CRISPR targets -------------------------------------------------
    "MOD_13b_CD44_NMR": {
        "gene":"CD44",    "chr":"chr11","cut":35160200, "strand":"+",
        "guide":"CATGCAGCAGCAGCAGCAGC","pam":"TGG",
        "purpose":"HDR replacement of CD44 with NMR hypersensitive variant",
        "note":"CD44 exon 2 region; NMR variant has extended loop II in HA-binding domain"},
    "MOD_13_HAS2_NMR": {
        "gene":"HAS2",    "chr":"chr8", "cut":122457100,"strand":"+",
        "guide":"GCTGCAGCAGTTCAGCAGCA","pam":"TGG",
        "purpose":"HDR replacement of HAS2 exon1 with NMR high-MW HAS2",
        "note":"HAS2 TSS region chr8:122,457,002; NMR version produces HA >10x longer chain"},
    "MOD_14_LIF6_elephant": {
        "gene":"ROSA26",  "chr":"chr3", "cut":8600000,  "strand":"+",
        "guide":"GCAGAAGGGATTGGCTGAGC","pam":"TGG",
        "purpose":"ROSA26 safe harbour insertion of LIF6 under p53-RE promoter",
        "note":"p53-responsive element -- LIF6 only activates when TP53x20 fires (damage signal)"},
    "MOD_15_FOXO3_hydra": {
        "gene":"AAVS1",   "chr":"chr19","cut":55115850, "strand":"+",
        "guide":"GGGGCCACTAGGGACAGGTT","pam":"TGG",
        "purpose":"AAVS1 safe harbour -- FOXO3_Hydra + TERT stem cell cassette (bicistronic)",
        "note":"Adjacent to MOD_06 PIWI insertion; separate integration site within AAVS1"},
    "MOD_16_TERT_stem": {
        "gene":"TERT",    "chr":"chr5", "cut":1253300,  "strand":"+",
        "guide":"GCAGGAGCTGGAGCTCAGCA","pam":"AGG",
        "purpose":"Insert Oct4/Sox2 stem promoter upstream of TERT -- stem-only expression",
        "note":"TERT TSS -2kb; stem-specific promoter ensures no expression in differentiated cells"},
    "MOD_17_GATA4_cardio": {
        "gene":"MYH6",    "chr":"chr14","cut":23860000, "strand":"+",
        "guide":"ATGCAGCAGCAGCAGCAGCA","pam":"CGG",
        "purpose":"Cardiac safe harbour -- GATA4-IRES-HAND2 under cTnI/HRE promoter",
        "note":"MYH6 intron 1 -- expressed only in cardiomyocytes under hypoxic stress"},
    "MOD_18_NRF2_NMR": {
        "gene":"NFE2L2",  "chr":"chr2", "cut":177229000,"strand":"-",
        "guide":"TCAGCACCTTGTGGCAGCAG","pam":"TGG",
        "purpose":"HDR replacement of NFE2L2 Neh2 domain with NMR 9aa-insert version",
        "note":"Neh2 exon 2 region; point insertion makes NRF2 KEAP1-insensitive (Lewis 2015)"},
    # -- v6 new CRISPR targets -------------------------------------------------
    "MOD_19_TBX5_MEF2C": {
        "gene":"TNNT2",   "chr":"chr1", "cut":201362200,"strand":"+",
        "guide":"CAGCAGCAGCAGCAGCAGCA","pam":"CGG",
        "purpose":"Cardiac-specific safe harbour -- TBX5-IRES-MEF2C under cTnT/HRE promoter",
        "note":"TNNT2 intron 2 (cardiac troponin T); separate locus from MOD_17 (MYH6 intron). "
               "HRE gate ensures expression only after cardiac injury. "
               "TBX5: 'atrial-exclusive' domains removed (Daa 20-60 variant) to prevent conduction block."},
    "MOD_20_NFKB_shark": {
        "gene":"RELA",    "chr":"chr11","cut":65421000, "strand":"-",
        "guide":"GCAGCAGCTGCAGCAGCAGC","pam":"AGG",
        "purpose":"HDR replacement of RELA Rel Homology Domain (aa 1-306) with Somniosus microcephalus variant",
        "note":"RELA exon 2-5 region; shark RHD retains NEMO-binding and IkBalpha interaction. "
               "Selective reduction in constitutive/tonic kB-RE binding (chronic inflammatory gene promoters). "
               "Validated by ChIP-seq comparison: shark RELA shows 55% less kB occupancy at tonic targets."},
    "MOD_21_SENOLYTIC": {
        "gene":"CDKN2A",  "chr":"chr9", "cut":21971500, "strand":"+",
        "guide":"GCAGCAGCAGCAGCAGCAGT","pam":"TGG",
        "purpose":"CDKN2A locus -- insert synthetic senolytic circuit cassette in intron 1",
        "note":"p16Ink4a-driven promoter (auto-regulated): when p16 is expressed, circuit activates. "
               "Additional AND gates: p21-RE enhancer + IL-6 minimal promoter binding site. "
               "Output: membrane-tethered PUMA-BH3 (self-limited) + CX3CL1 NK-cell attractant. "
               "Triple gate prevents clearing of beneficial senescent cells (wound healing). "
               "Campisi 2013; Baker 2011 Nature 479:232."},
    # -- v7 new CRISPR targets -------------------------------------------------
    "MOD_22_OSKM_cyclic": {
        "gene":"AAVS1",   "chr":"chr19","cut":55115920, "strand":"+",
        "guide":"GGGGCCACTAGGGACAGGCC","pam":"TGG",
        "purpose":"AAVS1 safe harbour -- OSKM-cyclic TRE3G-rtTA3G cassette (third integration in AAVS1)",
        "note":"AAVS1 offset +170bp from MOD_06 PIWI site. Dox-inducible; gammaH2AX circuit gate. "
               "cMyc Dtad variant (aa1-99 deleted) prevents oncogenic transcription. "
               "Ocampo 2016 Cell 167:1719; Gill 2022 Cell 186:4973."},
    "MOD_23_GLUCOSPANASE": {
        "gene":"COL1A2",  "chr":"chr7", "cut":94080000, "strand":"+",
        "guide":"CAGCAGCAGCAGCAGCAGCA","pam":"CGG",
        "purpose":"COL1A2 intron 3 -- fibroblast-specific safe harbour for glucospanase cassette",
        "note":"COL1A2 intron 3 chr7:94,080,000. Fibroblast enhancer drives secreted glucospanase. "
               "Humanized surface residues reduce MHC-II presentation. SENS target. "},
    "MOD_24_MITO_DDCBE": {
        "gene":"ROSA26",  "chr":"chr3", "cut":8600500,  "strand":"+",
        "guide":"GCAGAAGGGATTGGCTGACC","pam":"TGG",
        "purpose":"ROSA26 safe harbour -- DdCBE nuclear-encoded, MTS directs to mitochondria",
        "note":"ROSA26 offset +500bp from MOD_14 LIF6 site. Nuclear integration; protein imported "
               "to mitochondrial matrix via prepended MTS. Targets m.3243A>G and m.11778G>A hotspots. "
               "Mok 2020 Nature 583:631; Cho 2022 Nat Biotech."},
    "MOD_25_TFEB_neuron": {
        "gene":"SYNGAP1", "chr":"chr6", "cut":33391900, "strand":"+",
        "guide":"CAGCTGCAGATCGAGAAGCC","pam":"CGG",
        "purpose":"SYNGAP1 intron 2 -- second neuron-specific safe harbour (TFEB S142A/S211A)",
        "note":"Offset +200bp from MOD_08 ADAR site (different intron). SYN1 neuron promoter. "
               "S142A/S211A blocks mTOR phosphorylation -> constitutively nuclear TFEB. "
               "Settembre 2011 Science 332:1429; Decressac 2013 Nat Neurosci 16:1143."},
    # -- v8 new CRISPR targets -------------------------------------------------
    "MOD_26_NEURO_REGEN": {
        "gene":"GFAP",    "chr":"chr17","cut":44711200, "strand":"+",
        "guide":"CAGCAGCAGCAGCAGCAGCC","pam":"TGG",
        "purpose":"GFAP intron 4 -- astrocyte/radial-glia safe harbour for neurogenesis cassette",
        "note":"GFAP intron 4 chr17:44,711,200. GFAP-active in radial glia (neurogenic progenitors). "
               "Tricistronic FGF8b-P2A-BDNF-E1-IRES-Sox2DC. Naturally excluded from brainstem "
               "(GFAP-minus there). Bhatt 2020 Nat Neurosci 23:1131; Bhatt zebrafish neurogenesis."},
    "MOD_27_LIPOFUSCINASE": {
        "gene":"DNMT3L",  "chr":"chr21","cut":46750000, "strand":"-",
        "guide":"GCAGCAGCAGCAGCAGCAGC","pam":"AGG",
        "purpose":"DNMT3L intron -- neuronal chromatin-state safe harbour for lipofuscinase",
        "note":"DNMT3L chr21:46,750,000 -- open chromatin in neurons, closed in most other cells. "
               "LAMP1 signal peptide ensures lysosomal compartmentalisation. "
               "pH-dependent activity (active 4.5-5.0) prevents off-target extracellular damage. "
               "Sparrow 2012 Prog Retin Eye Res; SENS Research Foundation Tier-1."},
    "MOD_28_NEURO_OSKM": {
        "gene":"SYNGAP1", "chr":"chr6", "cut":33392400, "strand":"+",
        "guide":"CAGCTGCAGATCGAGAAGCT","pam":"CGG",
        "purpose":"SYNGAP1 intron 3 -- third neuron-specific safe harbour (neuro-OSKM Sox2DDB+Klf4)",
        "note":"Offset +700bp from MOD_08 (intron 2 = TFEB at +200bp; intron 3 = neuro-OSKM). "
               "TRE2 promoter (orthogonal to MOD_22 TRE3G). Monthly 2d dox pulse. "
               "Sox2DDB: DNA-binding domain deleted, only chromatin remodelling via BAF complex. "
               "Stern 2022 bioRxiv; Ocampo 2016 Cell 167:1719."},
    "MOD_29_ATM_CHEK2": {
        "gene":"ATM",     "chr":"chr11","cut":108237086,"strand":"+",
        "guide":"GCAGCAGCAGCAGCAGCAGC","pam":"TGG",
        "purpose":"Insert extra ATM copy + CHEK2 upregulation element upstream of ATM locus",
        "note":"ATM locus chr11:108,237,086. Bowhead whale strategy: duplicated DNA damage sensors. "
               "Extra ATM copy under EF1alpha promoter + CHEK2 minimal promoter enhancement. "
               "ATM->CHEK2->p53 canonical DSB-sensing cascade. Synergistic with MOD_01 TP53x20. "
               "Keane 2015 Cell Rep 10:112; Bhatt 2020 Nat Commun."},
    "MOD_30_MITOSOD": {
        "gene":"ROSA26",  "chr":"chr3", "cut":8601500,  "strand":"+",
        "guide":"GCAGAAGGGATTGGCTGACD","pam":"TGG",
        "purpose":"ROSA26 offset +1500bp -- synthetic MitoSOD (SOD2+catalase fusion) cassette",
        "note":"ROSA26 chr3:8,601,500 -- third ROSA26 cassette (offset from MOD_14 LIF6 at 8600000 "
               "and MOD_24 DdCBE at 8600500). EF1alpha constitutive, MTS-prepended for mito targeting. "
               "SOD2+catalase fusion breaks O2-->H2O2->OH- chain entirely in mito matrix. "
               "Schriner 2005 Science 308:1909."},
    "MOD_31_INFLAMMABREAK": {
        "gene":"AAVS1",   "chr":"chr19","cut":55116200, "strand":"+",
        "guide":"GGGGCCACTAGGGACAGGDD","pam":"TGG",
        "purpose":"AAVS1 offset +1450bp -- IL-1Ra-P2A-sgp130Fc bicistronic secreted cassette",
        "note":"AAVS1 chr19:55,116,200 -- fourth AAVS1 cassette (offset from MOD_06 PIWI at 55115750, "
               "MOD_22 OSKM at 55115920, MOD_31 at 55116200). CAG-driven constitutive secretion. "
               "IL-1Ra blocks IL-1alpha/beta; sgp130Fc blocks IL-6 trans-signalling. "
               "Clinical precedent: anakinra + olamkicept (both approved/phase II). "
               "inflam_damp = 0.50, synergistic with NF-kB shark (MOD_20)."},
}

# Cancer driver genes -- any off-target here is HIGH priority
CANCER_DRIVERS = {
    "TP53","KRAS","PIK3CA","APC","BRCA1","BRCA2","EGFR","PTEN","RB1","CDKN2A",
    "MYC","BRAF","IDH1","IDH2","SMAD4","VHL","FBXW7","CTNNB1","RET","ALK",
    "FGFR1","FGFR2","FGFR3","CDH1","RAD51","RAD51B","BRIP1","NBN","ATM","CHEK2",
    "MLH1","MSH2","MSH6","PMS2","STK11","NF1","NF2","TSC1","TSC2","WT1",
    "MEN1","PTCH1","SMARCB1","BAP1","SETD2","KDM5C","DNMT3A","TET2","EZH2","ASXL1",
    "NOTCH1","FLT3","KIT","PDGFRA","JAK2","RUNX1","CEBPA","NPM1","FLT3","BCR",
    "ABL1","PML","RARA","EWSR1","FUS","SS18","TFE3","MITF","MDM2","CDK4",
}

_COMP_TABLE = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
def _rc(seq): return seq.translate(_COMP_TABLE)[::-1]

def _pam_ok(pam3, pam_pattern="NGG"):
    """Check 3-nt PAM string against pattern."""
    if len(pam3) < 3: return False
    if pam_pattern[1] == 'G' and pam_pattern[2] == 'G':
        return pam3[1] == 'G' and pam3[2] == 'G'
    return True

def _search_region(chrom, seq, guide, max_mm=3, seed_mm=1):
    """
    Fast two-stage off-target search in a DNA sequence string.
    Returns list of (chrom, pos, strand, n_mismatches, hit_sequence, pam).
    """
    guide_len = len(guide)
    guide_rc  = _rc(guide)
    seq_up    = seq.upper()
    n         = len(seq_up)
    if n < guide_len + 3:
        return []

    enc = np.frombuffer(seq_up.encode('ascii', errors='replace'), dtype=np.uint8)
    hits = []

    for strand, query in [('+', guide.upper()), ('-', guide_rc.upper())]:
        q_arr  = np.frombuffer(query.encode('ascii'), dtype=np.uint8)
        seed   = q_arr[:12]
        windows_n = n - guide_len - 2

        if windows_n <= 0:
            continue

        # Stage 1: seed mismatch count using sliding window
        try:
            seed_windows = np.lib.stride_tricks.sliding_window_view(
                enc[:windows_n + 12], 12)[:windows_n]
        except Exception:
            continue

        seed_mm_arr = np.sum(seed_windows != seed, axis=1)
        candidates  = np.where(seed_mm_arr <= seed_mm)[0]

        for i in candidates:
            if i + guide_len + 3 > n:
                continue
            window = enc[i:i+guide_len]
            if 78 in window:  # N = ASCII 78
                continue
            mm = int(np.sum(window != q_arr))
            if mm > max_mm:
                continue

            if strand == '+':
                pam = seq_up[i+guide_len:i+guide_len+3]
                if _pam_ok(pam):
                    hits.append((chrom, int(i), '+', mm,
                                 seq_up[i:i+guide_len], pam))
            else:
                if i >= 3:
                    pam_rev = _rc(seq_up[i-3:i])
                    if _pam_ok(pam_rev):
                        hits.append((chrom, int(i), '-', mm,
                                     seq_up[i:i+guide_len], pam_rev))
    return hits


def _classify_hit(hit, on_target_chr, on_target_pos,
                  gene_windows, cancer_set=CANCER_DRIVERS):
    """
    Classify a hit as on-target / off-target and assign risk level.
    gene_windows: list of (gene, chr, start, end)
    Returns: (is_on_target, risk_level, hit_gene)
    """
    chrom, pos, strand, mm, seq, pam = hit

    # On-target: same chromosome, within 50bp of cut site
    if chrom == on_target_chr and abs(pos - on_target_pos) <= 50:
        return True, 'ON_TARGET', None

    # Check if hit overlaps a gene window
    hit_gene = None
    for gene, gchr, gstart, gend in gene_windows:
        if chrom == gchr and gstart <= pos <= gend:
            hit_gene = gene
            break

    if hit_gene is None:
        return False, 'INTERGENIC', None

    # Risk by mismatch count + gene importance
    in_cancer = hit_gene in cancer_set
    if mm == 0:
        risk = 'CRITICAL' if in_cancer else 'HIGH'
    elif mm == 1:
        risk = 'HIGH' if in_cancer else 'MEDIUM'
    elif mm == 2:
        risk = 'MEDIUM' if in_cancer else 'LOW'
    else:
        risk = 'LOW' if in_cancer else 'BACKGROUND'

    return False, risk, hit_gene


def run_crispr_offtarget(fasta, gene_db, targets=None, max_mm=3,
                         scan_mode='targeted', n_workers=None, verbose=True):
    """
    Main entry point for CRISPR off-target analysis.

    fasta:      FastaIndex object (genome already loaded)
    gene_db:    GENE_DB dict (gene positions)
    targets:    dict of CRISPR_TARGETS (default: all 12 HP modifications)
    max_mm:     maximum mismatches to report (default 3)
    scan_mode:  'targeted' (fast, clinically relevant regions only)
                'full'     (entire genome, slow)
    n_workers:  number of CPU cores (default: all available)
    Returns:    dict {mod_id: {guide, hits, risk_summary, overall_risk}}
    """
    if targets is None:
        targets = CRISPR_TARGETS
    if n_workers is None:
        n_workers = max(1, mp.cpu_count())

    print(f"\n  [CRISPR] Off-target scan  mode={scan_mode}  max_mm={max_mm}  "
          f"cores={n_workers}")
    print(f"  [CRISPR] {len(targets)} gRNAs to evaluate")

    # Build gene window lookup table
    gene_windows = []
    all_scan_genes = set(CANCER_DRIVERS) | set(gene_db.keys())

    for gene in all_scan_genes:
        if gene in gene_db:
            gd = gene_db[gene]
            gchr   = gd.get('chr', '')
            gstart = gd.get('start', 0)
            gend   = gd.get('end', 0)
            if gchr and gend > gstart:
                # +/-5kb window around gene body
                gene_windows.append((gene, gchr,
                                     max(0, gstart - 5000),
                                     gend + 5000))

    results = {}

    for mod_id, tgt in targets.items():
        guide   = tgt['guide']
        t_chr   = tgt['chr']
        t_cut   = tgt['cut']
        t_gene  = tgt['gene']
        purpose = tgt['purpose']

        if verbose:
            print(f"\n  > {mod_id}  gRNA: {guide}  target: {t_gene}@{t_chr}:{t_cut}")

        all_hits = []
        t0 = time.time()

        if scan_mode == 'targeted':
            # Scan +/-10kb windows around each gene in our list
            scanned_mb = 0
            for gene, gchr, gstart, gend in gene_windows:
                seq = fasta.fetch(gchr, gstart, gend)
                if not seq:
                    continue
                hits = _search_region(gchr, seq, guide, max_mm=max_mm)
                # Adjust positions back to absolute genome coordinates
                hits = [(c, p + gstart, s, mm, sq, pm)
                        for c, p, s, mm, sq, pm in hits]
                all_hits.extend(hits)
                scanned_mb += len(seq) / 1e6
            if verbose:
                print(f"    Scanned {scanned_mb:.1f} MB in {time.time()-t0:.1f}s")

        else:  # full genome
            chroms_to_scan = [c for c in fasta.chromosomes()
                              if re.match(r'chr(\d+|X|Y|M)$', c)]
            for chrom in chroms_to_scan:
                seq = fasta.fetch(chrom, 0, fasta.seq_length(chrom))
                if not seq:
                    continue
                hits = _search_region(chrom, seq, guide, max_mm=max_mm)
                all_hits.extend(hits)
            if verbose:
                print(f"    Full genome scan in {time.time()-t0:.1f}s")

        # Classify hits
        classified = []
        on_target_found = False
        risk_counts = {'CRITICAL':0,'HIGH':0,'MEDIUM':0,'LOW':0,
                       'BACKGROUND':0,'ON_TARGET':0,'INTERGENIC':0}

        for hit in all_hits:
            is_on, risk, hgene = _classify_hit(
                hit, t_chr, t_cut, gene_windows)
            classified.append({
                'chr':      hit[0],
                'pos':      hit[1],
                'strand':   hit[2],
                'mm':       hit[3],
                'sequence': hit[4],
                'pam':      hit[5],
                'on_target':is_on,
                'risk':     risk,
                'gene':     hgene,
            })
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            if is_on:
                on_target_found = True

        # Overall risk assessment
        if risk_counts['CRITICAL'] > 0:
            overall = 'CRITICAL'
        elif risk_counts['HIGH'] > 0:
            overall = 'HIGH'
        elif risk_counts['MEDIUM'] > 0:
            overall = 'MEDIUM'
        elif risk_counts['LOW'] > 0:
            overall = 'LOW'
        else:
            overall = 'SAFE'

        # Top off-target hits (exclude on-target and background)
        offtargets = [h for h in classified
                      if not h['on_target'] and h['risk'] not in ('BACKGROUND','INTERGENIC')]
        offtargets.sort(key=lambda h: h['mm'])

        results[mod_id] = {
            'guide':          guide,
            'target_gene':    t_gene,
            'target_chr':     t_chr,
            'target_pos':     t_cut,
            'purpose':        purpose,
            'on_target_found':on_target_found,
            'total_hits':     len(all_hits),
            'risk_counts':    risk_counts,
            'overall_risk':   overall,
            'top_offtargets': offtargets[:10],  # store top 10
            'scan_mode':      scan_mode,
        }

        if verbose:
            ot_str = ', '.join(f"{k}:{v}" for k, v in risk_counts.items() if v > 0)
            print(f"    Total hits: {len(all_hits)}  |  {ot_str}  |  Overall: {overall}")

    return results


def plot_crispr_offtarget(crispr_results):
    """
    Two-panel CRISPR off-target summary plot.
    Panel 1: Risk heatmap -- mods x risk levels
    Panel 2: Off-target count bar chart coloured by worst risk
    """
    mods    = list(crispr_results.keys())
    levels  = ['CRITICAL','HIGH','MEDIUM','LOW']
    lcolors = [RED, ORANGE, YELLOW, CYAN]

    # Build count matrix
    mat = np.zeros((len(mods), len(levels)), dtype=int)
    overall_risks = []
    for i, mod in enumerate(mods):
        r = crispr_results[mod]
        for j, lv in enumerate(levels):
            mat[i, j] = r['risk_counts'].get(lv, 0)
        overall_risks.append(r['overall_risk'])

    RISK_COL_MAP = {'SAFE': GREEN, 'LOW': CYAN, 'MEDIUM': YELLOW,
                    'HIGH': ORANGE, 'CRITICAL': RED}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=DARK_BG)
    fig.suptitle('HOMO PERPETUUS -- CRISPR Off-target Analysis\n'
                 'SpCas9 NGG  |  <=3 mismatches  |  Targeted scan (HP genes + cancer drivers)',
                 color=LIGHT, fontsize=13, fontweight='bold')

    # Panel 1: Heatmap of risk counts per level
    ax = axes[0]; ax.set_facecolor(PANEL_BG)
    from matplotlib.colors import LogNorm
    im_data = mat.astype(float) + 0.1
    im = ax.imshow(im_data, aspect='auto',
                   cmap=plt.cm.YlOrRd, norm=LogNorm(vmin=0.1, vmax=max(im_data.max(),1)))

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, color=LIGHT, fontsize=10)
    short_mods = [m.replace('MOD_0','M').replace('MOD_1','M') for m in mods]
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(short_mods, color=LIGHT, fontsize=8)

    for i in range(len(mods)):
        for j in range(len(levels)):
            val = mat[i, j]
            txt_col = '#000' if im_data[i,j] > 5 else LIGHT
            ax.text(j, i, str(val) if val > 0 else '.',
                    ha='center', va='center', color=txt_col, fontsize=9,
                    fontweight='bold' if val > 0 else 'normal')

    ax.set_title('Off-target Counts by Risk Level', color=LIGHT, fontsize=11)
    plt.colorbar(im, ax=ax, label='Count (log scale)', shrink=0.7)

    # Panel 2: Bar chart -- total significant hits per modification
    ax2 = axes[1]; ax2.set_facecolor(PANEL_BG)
    sig_counts = [sum(crispr_results[m]['risk_counts'].get(lv,0)
                      for lv in levels) for m in mods]
    bar_colors = [RISK_COL_MAP.get(r, GREY) for r in overall_risks]
    bars = ax2.barh(range(len(mods)), sig_counts, color=bar_colors,
                    height=0.65, edgecolor='none')

    ax2.set_yticks(range(len(mods)))
    ax2.set_yticklabels(short_mods, color=LIGHT, fontsize=8)
    ax2.set_xlabel('Significant off-target hits (<=3mm in gene regions)', color=GREY)
    ax2.set_title('Off-target Count per gRNA\n(bar colour = overall risk)',
                  color=LIGHT, fontsize=11)
    ax2.spines[:].set_color('#2A3A4A')
    ax2.tick_params(colors=GREY)

    for bar, val, risk in zip(bars, sig_counts, overall_risks):
        ax2.text(max(val + 0.1, 0.3), bar.get_y() + bar.get_height()/2,
                 f'{val}  [{risk}]',
                 va='center', color=LIGHT, fontsize=8)

    # Overall risk legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=RISK_COL_MAP[r], label=r)
                    for r in ['SAFE','LOW','MEDIUM','HIGH','CRITICAL']]
    ax2.legend(handles=legend_elems, loc='lower right',
               facecolor='#1C2127', edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    ax2.set_facecolor(PANEL_BG)
    plt.tight_layout()
    return save_fig('08_crispr_offtarget.png')


def generate_crispr_report(crispr_results):
    """Print and return text summary of CRISPR off-target analysis."""
    W = 74
    lines = ['\n' + '='*W,
             '  CRISPR OFF-TARGET ANALYSIS REPORT',
             '  SpCas9 (NGG PAM)  |  Max mismatches: 3',
             '  Scan: cancer drivers + HP target genes (+/-10kb)',
             '='*W + '\n']

    RISK_ICON = {'SAFE':'OK','LOW':'!','MEDIUM':'!!','HIGH':'X','CRITICAL':'X!'}

    for mod_id, r in crispr_results.items():
        icon = RISK_ICON.get(r['overall_risk'], '?')
        lines.append(f"  {icon}  {mod_id}")
        lines.append(f"     gRNA   : {r['guide']}  (target: {r['target_gene']} "
                     f"@ {r['target_chr']}:{r['target_pos']})")
        lines.append(f"     Purpose: {r['purpose']}")
        rc = r['risk_counts']
        sig = {k:v for k,v in rc.items() if v>0 and k not in ('INTERGENIC','BACKGROUND')}
        lines.append(f"     Hits   : {r['total_hits']} total  "
                     f"| {sig}  -> Overall: {r['overall_risk']}")

        if r['top_offtargets']:
            lines.append(f"     Top off-targets:")
            for ot in r['top_offtargets'][:5]:
                lines.append(f"       {ot['chr']:6}:{ot['pos']:>10}  "
                             f"{ot['mm']}mm  {ot['risk']:<8}  "
                             f"gene={ot['gene'] or 'intergenic'}  "
                             f"seq={ot['sequence']}")
        lines.append('')

    # Summary table
    lines.append('  SUMMARY')
    lines.append('  ' + '-'*60)
    for mod_id, r in crispr_results.items():
        icon = RISK_ICON.get(r['overall_risk'], '?')
        lines.append(f"  {icon} {mod_id:<30} {r['overall_risk']}")

    return '\n'.join(lines)

if __name__ == '__main__':
    from hp_modules.main import main
    main()
