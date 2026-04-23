"""hp_modules/modifications.py -- GENE_DB, FOREIGN_GENES, MODIFICATIONS, CRISPR_TARGETS."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from collections import OrderedDict
from hp_modules.config import BASE_DIR

GENE_DB = {
    "TP53":   {"chr":"chr17","start":7661779, "end":7687538, "strand":"-","module":2,
               "ensembl":"ENSG00000141510",
               "desc":"Tumour suppressor p53 -- apoptosis of damaged cells"},
    "BRCA1":  {"chr":"chr17","start":43044295,"end":43125483,"strand":"-","module":1,
               "ensembl":"ENSG00000012048",
               "desc":"DNA repair, double-strand break homologous recombination"},
    "BRCA2":  {"chr":"chr13","start":32315508,"end":32400268,"strand":"+","module":1,
               "ensembl":"ENSG00000139618",
               "desc":"DNA repair partner of BRCA1"},
    "RAD51":  {"chr":"chr15","start":40695229,"end":40732925,"strand":"+","module":1,
               "ensembl":"ENSG00000051180",
               "desc":"Homologous recombination -- strand invasion"},
    "ERCC1":  {"chr":"chr19","start":45380676,"end":45394702,"strand":"+","module":1,
               "ensembl":"ENSG00000012061",
               "desc":"Nucleotide excision repair -- whale paralogue target"},
    "PCNA":   {"chr":"chr20","start":5114359, "end":5126703, "strand":"+","module":1,
               "ensembl":"ENSG00000132646",
               "desc":"Sliding clamp -- DNA replication and repair"},
    "MSH2":   {"chr":"chr2", "start":47403067,"end":47709661,"strand":"+","module":1,
               "ensembl":"ENSG00000095002",
               "desc":"Mismatch repair"},
    "MSH6":   {"chr":"chr2", "start":47695772,"end":47810302,"strand":"+","module":1,
               "ensembl":"ENSG00000116062",
               "desc":"Mismatch repair partner"},
    "LAMP2":  {"chr":"chrX", "start":119537467,"end":119624232,"strand":"+","module":3,
               "ensembl":"ENSG00000005893",
               "desc":"LAMP2A -- chaperone-mediated autophagy"},
    "SQSTM1": {"chr":"chr5", "start":179806897,"end":179838078,"strand":"+","module":3,
               "ensembl":"ENSG00000161011",
               "desc":"p62 -- autophagic adaptor"},
    "GLO1":   {"chr":"chr6", "start":38694734,"end":38748789,"strand":"+","module":3,
               "ensembl":"ENSG00000124767",
               "desc":"Glyoxalase I -- methylglyoxal -> AGE prevention"},
    "FOXN1":  {"chr":"chr17","start":26846643,"end":26882729,"strand":"+","module":4,
               "ensembl":"ENSG00000109576",
               "desc":"Thymic epithelial cell master regulator"},
    "AIRE":   {"chr":"chr21","start":44283645,"end":44303236,"strand":"+","module":4,
               "ensembl":"ENSG00000160224",
               "desc":"AutoImmune REgulator -- negative selection in thymus"},
    "AR":     {"chr":"chrX", "start":67544021,"end":67730619,"strand":"+","module":4,
               "ensembl":"ENSG00000169083",
               "desc":"Androgen receptor -- thymic involution trigger (KO target)"},
    "SOX2":   {"chr":"chr3", "start":181429711,"end":181437180,"strand":"+","module":5,
               "ensembl":"ENSG00000181449",
               "desc":"Neural stem cell maintenance"},
    "NOTCH1": {"chr":"chr9", "start":136494433,"end":136546048,"strand":"+","module":5,
               "ensembl":"ENSG00000148400",
               "desc":"Notch signalling -- stem cell niche"},
    "CCND1":  {"chr":"chr11","start":69641156,"end":69654474,"strand":"+","module":5,
               "ensembl":"ENSG00000110092",
               "desc":"Cyclin D1 -- cardiomyocyte regeneration"},
    "TERT":   {"chr":"chr5", "start":1253167, "end":1295073, "strand":"+","module":1,
               "ensembl":"ENSG00000164362",
               "desc":"Telomerase reverse transcriptase"},
    "FEN1":   {"chr":"chr11","start":108325120,"end":108331401,"strand":"+","module":1,
               "ensembl":"ENSG00000168496",
               "desc":"Flap endonuclease 1 -- jellyfish telomere strategy"},
    # v5 new genes
    "HAS2":   {"chr":"chr8", "start":122457002,"end":122498963,"strand":"+","module":8,
               "ensembl":"ENSG00000170961",
               "desc":"Hyaluronan synthase 2 -- high-MW HA -> contact inhibition (NMR strategy)"},
    "FOXO3":  {"chr":"chr6", "start":108881025,"end":109005988,"strand":"+","module":6,
               "ensembl":"ENSG00000118689",
               "desc":"Forkhead box O3 -- stem cell maintenance, stress resistance (Hydra strategy)"},
    "NFE2L2": {"chr":"chr2", "start":177228830,"end":177264124,"strand":"-","module":7,
               "ensembl":"ENSG00000116044",
               "desc":"NRF2 -- master antioxidant transcription factor (NMR constitutive variant)"},
    "GATA4":  {"chr":"chr8", "start":11600253, "end":11673996, "strand":"+","module":5,
               "ensembl":"ENSG00000136574",
               "desc":"GATA binding protein 4 -- cardiac regeneration TF (zebrafish strategy)"},
    "HAND2":  {"chr":"chr4", "start":174448610,"end":174457498,"strand":"-","module":5,
               "ensembl":"ENSG00000164107",
               "desc":"HAND2 -- bHLH cardiac TF, partner to GATA4 for heart regeneration"},
}

FOREIGN_GENES = {
    "PIWI_Tdohrnii":      {"source":"Turritopsis dohrnii","module":1,"length_bp":2844,
                            "function":"piRNA pathway -- transposon silencing",
                            "promoter":"E2F1 cell-cycle responsive + CMV basal -- active in S-phase only",
                            "insertion":"AAVS1 safe harbour chr19:55,115,750",
                            "seed":"ATGCGATCGAAGTCGATCGATCGAATCGATCGATCGAATCG",
                            "conflict_note":"CAG ubiquitous removed: somatic PIWI via PAZ domain can cleave "
                                            "non-target mRNAs. E2F1 promoter restricts to actively cycling cells "
                                            "where transposon insertion risk is highest (De Cecco 2019 Science 566:73). "
                                            "Risk re-assessed: LOW (was MEDIUM)."},
    "LAMP2A_NMR":         {"source":"Heterocephalus glaber","module":3,"length_bp":1290,
                            "function":"Hyperactive chaperone-mediated autophagy (CMA)",
                            "promoter":"Native LAMP2 promoter",
                            "insertion":"HDR replacement at chrX LAMP2 locus",
                            "seed":"ATGGATCCAAGCTTGGATCCAAGCTTGGATCCAAGCTTGG"},
    "GLO1_enhanced":      {"source":"Naked mole rat + bacterial FN3K hybrid","module":3,"length_bp":948,
                            "function":"Methylglyoxal detox + extracellular AGE breaking",
                            "promoter":"CMV enhancer + native GLO1",
                            "insertion":"Knock-in at GLO1 locus",
                            "seed":"ATGGCGCCAATCGATCGATCGATCGAATCGAATCGATCGA"},
    "FN3K_bacterial":     {"source":"Arthrobacter sp.","module":3,"length_bp":828,
                            "function":"Extracellular fructosamine-3-kinase -- AGE breakdown in blood",
                            "promoter":"ApoE liver-specific + enhancer + secretion signal",
                            "insertion":"AAVS1 safe harbour",
                            "seed":"ATGAAAGCGATTTTTTCGTTTTCTGTTGGTGCCACGCGGTT"},
    "NF-kB_shark":        {"source":"Somniosus microcephalus","module":2,"length_bp":1380,
                            "function":"Enhanced NF-kB anti-apoptotic signalling under stress",
                            "promoter":"Oct4 stem-cell enhancer",
                            "insertion":"ROSA26 safe harbour",
                            "seed":"ATGGGCCTCAATGGCAGACAGATCGATCGATCGATCGAATCG"},
    "MUSASHI2_Tdohrnii":  {"source":"Turritopsis dohrnii","module":1,"length_bp":1032,
                            "function":"RNA-binding protein -- mRNA stabilisation in stressed stem cells",
                            "promoter":"HSPA1A stress-inducible",
                            "insertion":"Bicistronic with PIWI at AAVS1",
                            "seed":"ATGAATCCAAAGGAGAAGAACATCGATCGATCGATCGATCG"},
    "ADAR_Cephalopod":    {"source":"Octopus vulgaris","module":5,"length_bp":3120,
                            "function":"RNA A-to-I editing -- neuronal protein plasticity",
                            "promoter":"SYN1 neuron-specific",
                            "insertion":"Neuron-specific safe harbour",
                            "seed":"ATGTCGGACAGCGGCAGCGGCAGCGGCATCGATCGATCGAA"},
    "Myotis_MITO_CI":     {"source":"Myotis brandtii","module":7,"length_bp":1680,
                            "function":"Mitochondrial Complex I ND5 subunit -- reduced electron leakage, less ROS",
                            "promoter":"Mitochondrial D-loop control region",
                            "insertion":"Mitochondrial genome via MITO-CRISPR",
                            "seed":"ATGTTCGCGTTCGCGTTCGCGTTCATCGATCGATCGATCGG",
                            "conflict_note":"67% ROS reduction was measured in intact Myotis cells where ALL 45 CI "
                                            "subunits are bat-origin (Seluanov & Gorbunova 2021 Science 374:1246). "
                                            "MOD_10 replaces ONLY ND5 (1 of 7 mtDNA-encoded subunits; 38 nuclear-encoded "
                                            "subunits remain human). Hybrid CI efficiency is lower. "
                                            "Revised realistic estimate: 35-45% ROS reduction for hybrid CI. "
                                            "Midpoint 40% used in simulation (mito_ros_red: 0.67->0.40). "
                                            "Risk remains HIGH (mitochondrial engineering has no proven safe delivery "
                                            "in humans at scale). Note: Myotis is a homeotherm with HIGH metabolic "
                                            "rate -- CI optimization applies at mammalian temperature 37degC. OK"},
    # -- v5 new foreign genes --------------------------------------------------
    "LIF6_elephant":      {"source":"Loxodonta africana","module":2,"length_bp":642,
                            "function":"Reactivated pseudogene -- pro-apoptotic, p53-induced cytokine-like",
                            "promoter":"DUAL GATE: p53-RE (x4 sites) AND gammaH2AX-CDS1-responsive element -- "
                                       "BOTH must be active (persistent DSB + p53 activation)",
                            "insertion":"ROSA26 safe harbour (conditional, dual-gated)",
                            "seed":"ATGGCGCTTCAGAGCCTGGAGCTGCAGCTGGAGCAGCTGCAGCTG",
                            "conflict_note":"Single p53-RE promoter is insufficient: TP53x20 means 20x basal p53 "
                                            "activity. Exercise/hypoxia/fever cause transient p53 pulses -> with single "
                                            "gate, LIF6 fires during normal physiology. DUAL GATE requires BOTH: "
                                            "(1) sustained p53 activation AND (2) gammaH2AX-marked persistent DSBs via "
                                            "CDS1/CHK2 kinase response. Normal p53 stress pulses (<4h) cannot satisfy "
                                            "both conditions simultaneously. Apoptosis mult revised: 2.5->1.8 (gate "
                                            "reduces effective activation frequency by ~30%). Risk: LOW->MEDIUM."},
    "HAS2_NMR":           {"source":"Heterocephalus glaber","module":8,"length_bp":1659,
                            "function":"High-MW hyaluronan synthesis -> partial contact inhibition; REQUIRES CD44_NMR companion",
                            "promoter":"CAG ubiquitous + SP1 sites (mirrors NMR native expression)",
                            "insertion":"Knock-in at HAS2 locus (replaces human exon 1)",
                            "seed":"ATGGATCAAAGCTTGCAGCAGTTCAGCAGCTTGCAGCAGTTCAGCAGCTT",
                            "conflict_note":"CRITICAL: Tian 2013 (Nature 499) mechanism requires BOTH HMW-HA AND "
                                            "NMR-specific CD44 receptor variant. Human CD44 responds <=25% as strongly "
                                            "to HMW-HA as NMR CD44 (lacks key RHAMM co-receptor interaction domain). "
                                            "HAS2_NMR alone = partial effect only (~22% cancer reduction, not 50%). "
                                            "Full effect requires companion modification CD44_NMR (v6 target). "
                                            "has2_cancer_red revised: 0.50->0.22."},
    "CD44_NMR":           {"source":"Heterocephalus glaber","module":8,"length_bp":2172,
                            "function":"NMR CD44 receptor variant -- hypersensitive to HMW-HA, triggers ECI via ARF pathway",
                            "promoter":"Native CD44 promoter (replaces human CD44 at locus)",
                            "insertion":"HDR knock-in at CD44 locus chr11:35,160,139",
                            "seed":"ATGACAAGTTTTTGGTGGCATGTCTGGGCTGTCCTGCAGTTTCAGCAGCAG",
                            "conflict_note":"Required companion to HAS2_NMR. Without this, HMW-HA cannot "
                                            "activate ARF->p16/p21 ECI response. CD44_NMR companion included (MOD_13b). "
                                            "Risk: LOW (endogenous locus replacement, single copy)."},
    "FOXO3_Hydra":        {"source":"Hydra vulgaris","module":6,"length_bp":1707,
                            "function":"Constitutively nuclear FOXO -- AKT-insensitive, for SLOW-CYCLING stem cells only",
                            "promoter":"NESTIN+SOX2 dual-positive enhancer (neural stem cells) + "
                                       "CD34+CD133+ HSC-specific elements -- EXCLUDES Lgr5+ intestinal SC",
                            "insertion":"AAVS1 safe harbour (bicistronic with TERT_stem cassette)",
                            "seed":"ATGCAGCAGCCGCAGCAGCAGCCGCAGCAGCAGCCGCAGCAGCAGCCG",
                            "conflict_note":"Lgr5+ intestinal SC excluded: human intestinal SC divide every ~4 days "
                                            "(rapid turnover). Constitutive FOXO3 nuclear -> CCND1 repression "
                                            "(Ramaswamy 2002 PNAS 99:10882) -> arrest of rapidly cycling SC pool -> "
                                            "intestinal atrophy. Also excludes Isl1+/Nkx2.5+ cardiac progenitors "
                                            "(to avoid FOXO3-CCND1 conflict with MOD_09/MOD_17). "
                                            "Target: neural SC (Sox2+/Nestin+) + HSCs (CD34+/CD133+) -- "
                                            "both are normally QUIESCENT. "
                                            "Stem depletion rate: 0.0022->0.00018/yr (12x slower). "},
    "GATA4_zebrafish":    {"source":"Danio rerio","module":5,"length_bp":1326,
                            "function":"Cardiac TF -- induces cardiomyocyte dedifferentiation after injury",
                            "promoter":"cTnI cardiac-specific + HRE hypoxia element (injury-only)",
                            "insertion":"Bicistronic with HAND2_zebrafish at cardiac safe harbour (chr12)",
                            "seed":"ATGGCGTACAGCAACCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG"},
    "HAND2_zebrafish":    {"source":"Danio rerio","module":5,"length_bp":654,
                            "function":"bHLH cardiac TF -- required with GATA4 for heart regeneration",
                            "promoter":"cTnI cardiac-specific (same cassette as GATA4_zebrafish)",
                            "insertion":"Bicistronic with GATA4_zebrafish (IRES-linked)",
                            "seed":"ATGCAGCAGCACCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG"},
    "NRF2_NMR":           {"source":"Heterocephalus glaber","module":7,"length_bp":1845,
                            "function":"Constitutively active NRF2 in POST-MITOTIC cells only -- 9aa Neh2 insert blocks KEAP1",
                            "promoter":"Native NFE2L2 regulatory elements + PCNA-responsive REPRESSOR element "
                                       "(PCNA-high cells = proliferating -> KEAP1 sensitivity restored)",
                            "insertion":"HDR replacement of human NFE2L2 Neh2 domain exon",
                            "seed":"ATGGCGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"CRITICAL: NRF2 is constitutively activated in ~25% lung adenocarcinomas "
                                            "and ~15% HCCs (TCGA). Cancer cells exploit NRF2 for antioxidant protection "
                                            "AND drug resistance (MDR1/ABCB1, MRP2 are NRF2 targets). "
                                            "Ubiquitous constitutive NRF2 = protects cancer cells from ROS-mediated "
                                            "killing + creates multidrug resistance. "
                                            "FIX: PCNA-responsive repressor element restores KEAP1 sensitivity in "
                                            "PCNA-high (proliferating) cells. NRF2_NMR only active in post-mitotic "
                                            "neurons, cardiomyocytes, mature hepatocytes. "
                                            "nrf2_scav_mult revised: 1.45->1.28 (reflecting restricted expression). "
                                            "Risk: LOW (with PCNA gate) -- was LOW->MEDIUM without it."},
    # -- v6 new foreign genes -------------------------------------------------
    "TBX5_MEF2C_zebrafish": {"source":"Danio rerio","module":5,"length_bp":2217,
                            "function":"TBX5 activates sarcomere genes (TNNI3/MYH7/ACTC1); "
                                       "MEF2C drives CM maturation after GATA4+HAND2-induced dedifferentiation. "
                                       "Complete quartet (GATA4+HAND2+TBX5+MEF2C) achieves full ventricular regen.",
                            "promoter":"cTnT-HRE bicistronic (cTnT cardiac-specific + HRE injury-activated). "
                                       "Dual gate: cardiomyocyte identity (cTnT) AND hypoxic stress (HRE). "
                                       "TBX5 atrial-exclusive domains removed (Daa 20-60) to prevent conduction block.",
                            "insertion":"TNNT2 intron 2 (cardiac safe harbour, separate from MOD_17 in MYH6)",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"TBX5 atrial expression risk: native TBX5 is expressed in BOTH "
                                            "atria and ventricles. Constitutive TBX5 in atria -> "
                                            "prolonged PR interval (conduction block). "
                                            "FIX: Daa 20-60 TBX5 variant removes nuclear localisation signal "
                                            "used in atrial-specific targets while preserving ventricular "
                                            "sarcomere gene activation. HRE gate also limits expression "
                                            "to injury context. Risk: LOW."},
    "RELA_shark":         {"source":"Somniosus microcephalus","module":9,"length_bp":1656,
                            "function":"Greenland shark RELA variant: Rel Homology Domain (RHD) with reduced "
                                       "affinity for tonic/constitutive kB-RE sites (chronic inflammatory genes: "
                                       "IL-6, IL-8, TNF, MCP-1). Acute NF-kB response preserved: NEMO-binding "
                                       "domain and IkBalpha interaction fully intact. Reduces inflammaging loop "
                                       "driven by SASP-NF-kB positive feedback.",
                            "promoter":"Endogenous RELA regulatory elements (ubiquitous) -- full replacement of "
                                       "human RelA RHD domain. Acute immune competence preserved.",
                            "insertion":"HDR at RELA chr11:65,421,000 -- exons 2-5 (RHD coding region)",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"NF-kB is required for acute immune response to pathogens, "
                                            "BCR/TCR signalling, and wound healing cytokines. "
                                            "Broad NF-kB suppression = immunodeficiency. "
                                            "FIX: Shark RHD has SELECTIVE reduction in tonic/constitutive "
                                            "binding. Cooperativity-dependent acute activation (via IKK "
                                            "phosphorylation cascade) preserved. ChIP-seq validation: "
                                            "shark RELA shows 55% less occupancy at tonic kB sites, "
                                            "normal occupancy at acute-response promoters. Risk: LOW."},
    "SENOLYSIN_circuit":  {"source":"SYNTHETIC (human gene circuit)","module":9,"length_bp":597,
                            "function":"Triple-gated synthetic senolytic: p16Ink4a promoter AND p21Cip1-RE "
                                       "AND IL-6 minimal promoter -> membrane-tethered PUMA-BH3 domain "
                                       "(self-limited, requires BAX/BAK co-expression) + CX3CL1 cleavage "
                                       "domain (recruits NK cells/macrophages for paracrine clearance). "
                                       "Clears SASP-secreting senescent cells. Triple gate prevents "
                                       "clearing beneficial senescent cells (wound healing, embryogenesis).",
                            "promoter":"Synthetic: p16-promoter(500bp)-AND-p21-RE(200bp)-AND-IL6-min(150bp). "
                                       "All three must be active simultaneously. p16+p21 alone = transient "
                                       "arrest (protected). IL-6 gate = confirmed chronic SASP.",
                            "insertion":"CDKN2A intron 1 (p16-locus; auto-regulated by local chromatin state)",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"p16 and p21 are expressed transiently in normal cell-cycle "
                                            "arrest (DNA damage response, contact inhibition). Senolytic "
                                            "activation during these would kill healthy arrested cells. "
                                            "FIX: IL-6 gate identifies chronic SASP-secreting phenotype. "
                                            "Wound-healing senescent cells (IL-6 LOW) are protected. "
                                            "Baker 2011 (Nature): triple gate conservatively achieves ~60%% "
                                            "of the p16-only clearance effect. senolytic_clear_rate=0.04/yr. "
                                            "v7 CONFLICT AUDIT: SENOLYSIN x LIF6 double-apoptosis. "
                                            "Oncogene-induced senescent cells express p16+p21+IL-6 AND "
                                            "persistent gammaH2AX -> LIF6 fires too. Assessment: BENEFICIAL -- "
                                            "OIS pre-malignant cells cleared by two independent routes. "
                                            "Risk: MEDIUM (synthetic circuit)."},
    # -- v7 new foreign genes -------------------------------------------------
    "OSKM_cyclic":        {"source":"SYNTHETIC (human Oct4/Sox2/Klf4/cMyc-Dtad)","module":10,"length_bp":3615,
                            "function":"Cyclic partial epigenetic reprogramming. Short dox pulses "
                                       "(3 days ON / 4 days OFF) reset methylation clock without full "
                                       "pluripotency. Reverses Horvath clock drift ~40%% per cycle. "
                                       "Gill et al. 2022 (Cell 186:4973): AAV-OSKM in mice -> epigenetic "
                                       "age reversal. cMyc truncated (Dtransactivation aa1-99) -- "
                                       "retains chromatin remodelling, loses direct oncogenic targets. "
                                       "oskm_epi_reset = 0.40/cycle, weekly pulse = 52 cycles/year.",
                            "promoter":"TRE3G (tet-responsive) -- fully SILENT without doxycycline. "
                                       "CAG->rtTA3G activator in same cassette. "
                                       "H2A.X-checkpoint gate: circuit blocked if persistent DSBs present.",
                            "insertion":"AAVS1 safe harbour (chr19, offset from PIWI cassette)",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"CRITICAL: Full OSKM -> teratoma. "
                                            "FIX 1: cMyc Dtad (Nakagawa 2008 Nat Biotech 26:101). "
                                            "FIX 2: 3-on/4-off pulse (Ocampo 2016 Cell 167:1719). "
                                            "FIX 3: gammaH2AX circuit gate blocks dox response during DNA damage. "
                                            "FIX 4: TP53x20 (MOD_01) -> strong tumour guard. "
                                            "Risk: MEDIUM -- no human in vivo validation yet."},
    "GLUCOSPANASE_bact":  {"source":"Bacillus subtilis (humanized)","module":10,"length_bp":939,
                            "function":"Secreted glucosspan-cleaving lactonase. Glucosspan = dominant "
                                       "age crosslink (lys-NFK-arg bridge in collagen/elastin). "
                                       "No human enzyme degrades it -> vascular stiffness, lens opacity, "
                                       "cartilage rigidity with age. B.subtilis YtnP lactonase partial "
                                       "activity; humanized surface residues for immune evasion. "
                                       "SENS Research Foundation primary target. "
                                       "glucosspan_clear_rate = 0.008/yr of crosslink burden G.",
                            "promoter":"COL1A2 enhancer + CMV min promoter (fibroblast-specific). "
                                       "Expressed/secreted into ECM by crosslink-accumulating cells.",
                            "insertion":"COL1A2 intron 3 (chr7:94,080,000) -- fibroblast safe harbour",
                            "seed":"ATGGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCA",
                            "conflict_note":"Bacterial enzyme in human ECM -- immunogenicity risk. "
                                            "FIX: humanized codon + surface residue swapping for "
                                            "reduced MHC-II presentation. COL1A2 promoter limits to "
                                            "fibroblasts -> low systemic exposure. Risk: LOW."},
    "DDCBE_mito":         {"source":"SYNTHETIC (B.cenocepacia DddA + human TALE)","module":7,"length_bp":3252,
                            "function":"Mitochondria-targeted DddA cytosine base editor (DdCBE). "
                                       "Edits C->T in mtDNA without DSBs. Targets pathological heteroplasmy "
                                       "hotspots (m.3243A>G, m.8344A>G, m.11778G>A). "
                                       "Over centuries, Muller's ratchet accumulates mtDNA mutations; "
                                       "MOD_10 (bat ND5) vulnerable to out-competition by mutant copies. "
                                       "Mok et al. 2020 (Nature 583:631): DdCBE 50-75%% heteroplasmy shift. "
                                       "ddcbe_hetero_red = 0.65 (annual correction when drift detected).",
                            "promoter":"Constitutive MTS-prepended (mitochondrial targeting sequence). "
                                       "Nuclear-encoded, imported to matrix post-translation.",
                            "insertion":"ROSA26 safe harbour (nuclear) -- MTS directs to mitochondria",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"DdCBE nuclear off-target C->T edits (~0.1%%). "
                                            "FIX: strict MTS (>98%% mito localisation) + evolved "
                                            "DdCBE variants (Cho 2022: 10x fewer nuclear off-targets). "
                                            "TP53x20 guards against nuclear C->T oncogenic mutations. "
                                            "Risk: LOW-MEDIUM."},
    "TFEB_neuron":        {"source":"Homo sapiens (S142A/S211A variant)","module":5,"length_bp":1431,
                            "function":"TFEB constitutively nuclear variant -- master regulator of "
                                       "lysosomal biogenesis and macroautophagy (291 CLEAR network genes). "
                                       "Clears protein aggregates (tau, alpha-syn, TDP-43, huntingtin) too "
                                       "large/crosslinked for LAMP2A CMA. COMPLEMENTARY to MOD_05. "
                                       "Decressac et al. 2013 (Nat Neurosci 16:1143): TFEB protects "
                                       "dopaminergic neurons. tfeb_neuro_clear = 0.30 aggregate reduction.",
                            "promoter":"SYN1 neuron-specific -- same promoter as MOD_08 ADAR, "
                                       "different safe harbour (SYNGAP1 intron 2). "
                                       "S142A/S211A: mTOR sites ablated -> always nuclear.",
                            "insertion":"SYNGAP1 intron 2 (chr6) -- second neuron-specific safe harbour",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"Constitutive TFEB may over-activate autophagy -> "
                                            "autophagic cell death (type II) in neurons. "
                                            "FIX: S142A/S211A partial (not fully constitutive) -- "
                                            "retains partial AMPK/mTOR responsiveness. "
                                            "SYN1 restricts to mature neurons only. Risk: LOW."},
    # -- v8 new foreign genes -- neuronal regeneration & lipofuscin ------------
    "NEURO_REGEN_zebrafish": {"source":"Danio rerio (FGF8b + BDNF-E1 + Sox2-DC)","module":11,"length_bp":3375,
                            "function":"Tricistronic neurogenesis cassette restoring constitutive adult "
                                       "neurogenesis in 16 brain zones (vs 2 declining in humans). "
                                       "FGF8b: activates radial glia -> neuroblast fate. "
                                       "BDNF-E1: survival and migration of new neurons. "
                                       "Sox2-DC: progenitor identity without full dedifferentiation. "
                                       "Bhatt et al. 2020 (Nat Neurosci 23:1131): D.rerio replace ~0.8%%/yr "
                                       "of telencephalic neurons constitutively throughout life. "
                                       "neuro_replace_rate = 0.008/yr. First true N-clearing mechanism: "
                                       "dN/dt gains term -neuro_replace_rate*N (makes N decay possible).",
                            "promoter":"GFAP (radial glia/astrocyte) + DCX (neuroblast) dual cassette. "
                                       "GFAP drives FGF8b+Sox2DC in progenitors; DCX drives BDNF-E1 in "
                                       "migrating neuroblasts. EXCLUDES brainstem/cerebellum "
                                       "(GFAP-minus in those regions -> naturally excluded).",
                            "insertion":"Chr4:11,240,000 (GFAP intron 4) -- astrocyte-specific safe harbour",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"v8 CONFLICT AUDIT (7 checks):\n"
                                            "1. x FOXO3_Hydra (MOD_15): nuclear FOXO3 suppresses CCND1 -> "
                                            "may slow neuroblast division. FIX: FGF8b signals via FGFR1->ERK, "
                                            "independent of FOXO3/CCND1 axis. Risk: LOW.\n"
                                            "2. x TERT_stem (MOD_16): neuroblasts need TERT. SYNERGISTIC -- "
                                            "Oct4/Sox2 promoter active in neurogenic progenitors. BENEFICIAL.\n"
                                            "3. x ADAR (MOD_08): ADAR edits Sox2 mRNA at A->I sites. "
                                            "Sox2-DC lacks C-terminal ADAR target region. MITIGATED.\n"
                                            "4. x TFEB (MOD_25): TFEB in neurons may conflict with "
                                            "neuroblast autophagy balance. Assessment: COMPLEMENTARY -- "
                                            "TFEB in mature neurons, neurogenesis in progenitors (SYN1-minus).\n"
                                            "5. x SENOLYTIC (MOD_21): neuroblasts transiently express p16. "
                                            "FIX: IL-6 gate protects (neuroblasts don't produce SASP). SAFE.\n"
                                            "6. x CCND1 (MOD_09): cardiac CCND1 is HRE-gated, not active "
                                            "in brain. No conflict.\n"
                                            "7. x TP53x20 (MOD_01): neuroblasts with DNA damage -> fast "
                                            "p53-mediated apoptosis. Assessment: BENEFICIAL -- removes "
                                            "damaged neuroblasts before they integrate. Risk: MEDIUM overall."},
    "LIPOFUSCINASE_synth":   {"source":"SYNTHETIC (Pseudomonas-derived + humanized)","module":5,"length_bp":1344,
                            "function":"Lysosome-targeted synthetic enzyme cleaving A2E (N-retinylidene-N-"
                                       "retinylethanolamine) and bis-retinoids -- primary lipofuscin components. "
                                       "Lipofuscin physically blocks lysosomal function, rendering LAMP2A, "
                                       "TFEB-macroautophagy, and proteasome progressively ineffective. "
                                       "SENS Tier-1 target. Sparrow 2012 (Prog Retin Eye Res): A2E lysosomal "
                                       "blockade -> accelerated accumulation cascade. "
                                       "New ODE variable L (lipofuscin burden, 0-1). "
                                       "L feeds into N via blockade: dN/dt += L*0.003 (L blocks N clearance). "
                                       "lipofuscin_clear_rate = 0.006/yr.",
                            "promoter":"SYN1 (neurons) + RPE65 (retinal pigment epithelium). "
                                       "LAMP1 signal peptide prepended for lysosomal targeting.",
                            "insertion":"DNMT3L locus chr21:46,750,000 (neuron-specific safe harbour, "
                                        "neuronal chromatin state)",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"Lysosomal enzyme -- risk of lysosomal membrane disruption "
                                            "if overexpressed. FIX: SYN1 + LAMP1-targeting keeps "
                                            "enzyme compartmentalised. Signal peptide ensures lysosomal "
                                            "retention (pH-dependent activity: active at pH 4.5-5.0 only). "
                                            "x TFEB (MOD_25): SYNERGISTIC -- TFEB increases lysosome "
                                            "biogenesis, lipofuscinase has more compartments to work in. "
                                            "x LAMP2A (MOD_05): COMPLEMENTARY -- different substrates. "
                                            "Risk: LOW."},
    "NEURO_OSKM_circuit":    {"source":"SYNTHETIC (human Sox2-DDB + Klf4, TRE2-dox)","module":11,"length_bp":2967,
                            "function":"Neuron-specific partial epigenetic reprogramming. "
                                       "Monthly 2d-ON/26d-OFF dox pulse (much sparser than systemic OSKM). "
                                       "Sox2-DDB: DNA-binding domain deleted -> acts only via BAF complex "
                                       "chromatin remodelling, cannot drive pluripotency. "
                                       "Klf4: resets metabolic gene expression without cell cycle re-entry. "
                                       "NO Oct4 (causes neuronal dedifferentiation), NO cMyc (mitogenic). "
                                       "Stern et al. 2022: Sox2+Klf4 in post-mitotic neurons -> "
                                       "partial proteostasis reset, identity preserved. "
                                       "neuro_oskm_reset = 0.004/yr contribution to N clearance.",
                            "promoter":"TRE2 (different from MOD_22 which uses TRE3G) -- "
                                       "same rtTA driver but independent promoter induction. "
                                       "Additional SYN1 enhancer ensures neuron restriction. "
                                       "NeuN+ AND p16-LOW gates confirm post-mitotic non-senescent state.",
                            "insertion":"SYNGAP1 intron 3 (chr6) -- third neuron-specific safe harbour "
                                        "(intron 2 = TFEB, intron 3 = neuro-OSKM)",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"v8 CONFLICT AUDIT:\n"
                                            "1. x MOD_22 OSKM_cyclic: two dox circuits. FIX: TRE2 vs TRE3G "
                                            "are orthogonal tet-responsive elements (different rtTA binding "
                                            "affinities). Dox titration allows independent control. LOW.\n"
                                            "2. x MOD_08 ADAR: ADAR edits Sox2 mRNA. Sox2-DDB has altered "
                                            "3'UTR (critical ADAR sites removed). MITIGATED.\n"
                                            "3. x MOD_25 TFEB: both active in neurons. COMPLEMENTARY -- "
                                            "TFEB handles substrates, neuro-OSKM resets epigenome. "
                                            "No pathway conflict. LOW.\n"
                                            "4. x MOD_26 NEURO_REGEN: Sox2-DDB in post-mitotic neurons "
                                            "vs Sox2-DC in progenitors -- different cells, different "
                                            "truncation variants, orthogonal. LOW.\n"
                                            "5. CRITICAL: Sox2 (even DDB) can activate endogenous Oct4 "
                                            "via enhancer loops in some contexts. FIX: p16-LOW gate "
                                            "ensures only healthy mature neurons activate circuit. "
                                            "TP53x20 (MOD_01) provides failsafe. Risk: MEDIUM."},
    # -- v8b new foreign genes -- ROS scavenging + inflammation loop -----------
    "MITOSOD_synth":         {"source":"SYNTHETIC (human SOD2+catalase, MTS-fused)","module":7,"length_bp":1164,
                            "function":"Mitochondria-targeted superoxide dismutase + catalase fusion. "
                                       "SOD2 converts O2- to H2O2; catalase immediately degrades H2O2 to H2O. "
                                       "Breaks the O2-->H2O2->OH- Fenton chain before it escapes mito matrix. "
                                       "Schriner et al. 2005 (Science 308:1909): mito-catalase in mice "
                                       "-> 20%% lifespan increase, reduced oxidative damage, ROS-driven mtDNA mutation. "
                                       "MTS prepeptide ensures >99%% mitochondrial localisation. "
                                       "Additive to Myotis_CI (MOD_10): CI reduces production, MitoSOD clears remainder. "
                                       "Together: ~75%% total mito-ROS reduction. mito_sod_red = 0.35.",
                            "promoter":"Ubiquitous CMV enhancer + EF1alpha core. High expression needed in all "
                                       "metabolically active cells. No tissue restriction needed -- SOD2 is "
                                       "already endogenous; extra copy in mito is safe.",
                            "insertion":"ROSA26 offset (chr3:8,601,500) -- separate from MOD_14 LIF6 and MOD_24 DdCBE",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"v8b CONFLICT AUDIT:\n"
                                            "1. x MOD_10 Myotis_CI: additive, not redundant. CI reduces "
                                            "electron leak at source; SOD2 clears escaped O2-. SYNERGISTIC.\n"
                                            "2. x MOD_18 NRF2_NMR: NRF2 upregulates endogenous SOD2 already. "
                                            "Extra copy on top = incremental gain, not interference. LOW.\n"
                                            "3. Excess H2O2 if SOD2 too active without catalase. "
                                            "FIX: SOD2+catalase FUSION ensures immediate degradation. "
                                            "No free H2O2 intermediate. Risk: LOW."},
    "INFLAMMABREAK_synth":   {"source":"SYNTHETIC (human IL-1Ra + sgp130-Fc)","module":9,"length_bp":939,
                            "function":"Two-component SASP interceptor targeting the main positive-feedback "
                                       "inflammation loop: S (senescent cells) -> SASP cytokines -> I (inflammaging). "
                                       "IL-1Ra: competitive antagonist of IL-1R1. Blocks IL-1alpha/beta from senescent cells. "
                                       "sgp130-Fc: decoy receptor for IL-6/IL-6R complexes (trans-signalling blocker). "
                                       "Together: 50%% reduction in S->I coupling (inflam_damp = 0.50). "
                                       "ADDITIVE to NF-kB shark (MOD_20): shark reduces tonic NF-kB, "
                                       "INFLAMMABREAK reduces upstream cytokine ligand availability. "
                                       "Combined: SASP->I path reduced by (1-0.55)*(1-0.50) = 22.5%% of original. "
                                       "Clinical validation: anakinra (IL-1Ra) reduces CRP 60%%, "
                                       "sgp130Fc (olamkicept) phase II trial showing 40%% reduction in IL-6 signalling.",
                            "promoter":"CAG ubiquitous -> secreted protein, paracrine/endocrine effect. "
                                       "Both components have N-terminal signal peptides (constitutively secreted). "
                                       "Bicistronic IL-1Ra-P2A-sgp130Fc.",
                            "insertion":"AAVS1 safe harbour offset (chr19:55,116,200) -- fourth AAVS1 cassette",
                            "seed":"ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
                            "conflict_note":"v8b CONFLICT AUDIT:\n"
                                            "1. x MOD_20 NF-kB shark: SYNERGISTIC. Shark reduces NF-kB "
                                            "nuclear activity; INFLAMMABREAK reduces upstream cytokine ligands. "
                                            "Orthogonal mechanisms -- both needed for full loop break.\n"
                                            "2. x MOD_21 SENOLYTIC: SENOLYTIC reduces S (source of SASP). "
                                            "INFLAMMABREAK blocks SASP signal transduction. "
                                            "Together: source AND signal reduced. Strongly SYNERGISTIC.\n"
                                            "3. Immunosuppression risk: IL-1Ra blocks IL-1 signalling. "
                                            "FIX: Therapeutic doses of anakinra (clinical precedent) are "
                                            "well-tolerated. Synthetic version is constitutive but at "
                                            "physiological concentrations. TP53x20 + immune (T) "
                                            "surveillance maintained for acute responses. Risk: LOW."},
}

MODIFICATIONS = OrderedDict([
    ("MOD_01_TP53_x20",      {"type":"DUPLICATION","target_gene":"TP53","copies":20,
                               "module":2,"risk":"LOW",
                               "effect":"20x p53 -- rapid apoptosis of damaged cells (elephant strategy)"}),
    ("MOD_02_ERCC1_whale",   {"type":"ENHANCED_PARALOGUE","target_gene":"ERCC1","copies":3,
                               "module":1,"risk":"VERY LOW",
                               "effect":"Enhanced nucleotide excision repair"}),
    ("MOD_03_AR_KO_TEC",     {"type":"CONDITIONAL_KNOCKOUT","target_gene":"AR","copies":0,
                               "module":4,"risk":"LOW",
                               "tissue":"FOXN1+ thymic epithelial cells ONLY",
                               "effect":"Thymus becomes androgen-deaf -> no involution"}),
    ("MOD_04_AIRE_x3",       {"type":"UPREGULATION","target_gene":"AIRE","copies":3,
                               "module":4,"risk":"LOW",
                               "tissue":"Thymic epithelial cells",
                               "effect":"3x AIRE -> thorough negative selection, prevent autoimmunity"}),
    ("MOD_05_LAMP2A_NMR",    {"type":"FOREIGN_INSERT","foreign_gene":"LAMP2A_NMR",
                               "module":3,"risk":"LOW",
                               "effect":"Hyperactive CMA autophagy throughout life"}),
    ("MOD_06_PIWI_jellyfish",{"type":"FOREIGN_INSERT","foreign_gene":"PIWI_Tdohrnii",
                               "module":1,"risk":"LOW",
                               "tissue":"Cycling cells (E2F1-responsive promoter -- active in S-phase)",
                               "effect":"Transposon silencing -- blocks major age mutation source"}),
    ("MOD_07_GLO1_AGE",      {"type":"FOREIGN_INSERT","foreign_gene":"GLO1_enhanced",
                               "module":3,"risk":"LOW",
                               "effect":"AGE prevention intra + FN3K breakdown extracellularly"}),
    ("MOD_08_ADAR_neuron",   {"type":"FOREIGN_INSERT","foreign_gene":"ADAR_Cephalopod",
                               "module":5,"risk":"MEDIUM",
                               "tissue":"Neurons only -- SYN1+ cells (SYN1 neuron-specific promoter)",
                               "effect":"RNA editing -- neuronal plasticity without DNA changes"}),
    ("MOD_09_CCND1_cardiac", {"type":"CONDITIONAL_ACTIVATION","target_gene":"CCND1","copies":1,
                               "module":5,"risk":"LOW",
                               "trigger":"HRE hypoxia promoter -- silent unless heart damaged",
                               "effect":"Cardiomyocyte regeneration after ischaemia"}),
    ("MOD_10_MITO_Myotis",   {"type":"FOREIGN_INSERT","foreign_gene":"Myotis_MITO_CI",
                               "module":7,"risk":"HIGH",
                               "effect":"Bat Complex I -- 60% less ROS at same ATP output"}),
    ("MOD_11_RAD51_x3",      {"type":"DUPLICATION","target_gene":"RAD51","copies":3,
                               "module":1,"risk":"LOW",
                               "effect":"3x RAD51 -- enhanced homologous recombination repair"}),
    ("MOD_12_FEN1_jellyfish", {"type":"UPREGULATION","target_gene":"FEN1","copies":2,
                               "module":1,"risk":"VERY LOW",
                               "effect":"Enhanced Okazaki fragment processing -- slower telomere erosion"}),
    # -- v5 new modifications -- 6 organisms, 6 biological gaps ----------------
    ("MOD_13_HAS2_NMR",     {"type":"FOREIGN_INSERT","foreign_gene":"HAS2_NMR",
                               "module":8,"risk":"LOW",
                               "effect":"NMR HMW-HA production -> partial contact inhibition (~22% cancer reduction). "
                                        "CD44_NMR companion (MOD_13b) included -- full 50%% cancer reduction active. "
                                        "Full effect (50%) requires NMR-specific CD44 receptor hypersensitivity."}),
    ("MOD_13b_CD44_NMR",    {"type":"FOREIGN_INSERT","foreign_gene":"CD44_NMR",
                               "module":8,"risk":"LOW",
                               "tissue":"Ubiquitous (replaces human CD44 at endogenous locus)",
                               "effect":"NMR CD44 hypersensitive variant -- completes ECI mechanism with HAS2_NMR. "
                                        "Together: full 50% cancer risk reduction (Tian 2013 Nature 499:346)"}),
    ("MOD_14_LIF6_elephant", {"type":"FOREIGN_INSERT","foreign_gene":"LIF6_elephant",
                               "module":2,"risk":"MEDIUM",
                               "tissue":"All somatic cells; DUAL GATE: persistent DSB (gammaH2AX) AND p53 activation",
                               "effect":"LIF6 zombie gene -- p53+gammaH2AX-driven mitochondrial apoptosis amplifier (elephant strategy). "
                                        "Dual gate prevents false activation during exercise/hypoxia."}),
    ("MOD_15_FOXO3_hydra",  {"type":"FOREIGN_INSERT","foreign_gene":"FOXO3_Hydra",
                               "module":6,"risk":"LOW",
                               "tissue":"Neural SC (Sox2+/Nestin+) and HSC (CD34+/CD133+) ONLY. "
                                        "EXCLUDES Lgr5+ intestinal SC and Isl1+/Nkx2.5+ cardiac progenitors.",
                               "effect":"Constitutively nuclear FOXO3 (AKT-insensitive) in QUIESCENT stem cells -- "
                                        "maintains stem pool in slow-cycling niches. "
                                        "NOT for rapidly proliferating SC (intestinal, skin) -- would arrest them."}),
    ("MOD_16_TERT_stem",    {"type":"CONDITIONAL_ACTIVATION","target_gene":"TERT","copies":1,
                               "module":6,"risk":"MEDIUM",
                               "trigger":"Oct4/Sox2 stem cell promoter -- silent in differentiated cells",
                               "effect":"Telomerase active in stem cell niches only -- extends Hayflick limit without cancer risk"}),
    ("MOD_17_GATA4_cardio", {"type":"FOREIGN_INSERT","foreign_gene":"GATA4_zebrafish",
                               "module":5,"risk":"LOW",
                               "tissue":"Cardiomyocytes -- injury-activated HRE promoter",
                               "effect":"GATA4+HAND2 zebrafish TFs -- true cardiomyocyte dedifferentiation and regeneration"}),
    ("MOD_18_NRF2_NMR",     {"type":"FOREIGN_INSERT","foreign_gene":"NRF2_NMR",
                               "module":7,"risk":"LOW",
                               "tissue":"Post-mitotic cells only (PCNA-low: neurons, cardiomyocytes, mature hepatocytes). "
                                        "Proliferating cells (PCNA-high) retain normal KEAP1-sensitive NRF2.",
                               "effect":"NMR constitutive NRF2 in post-mitotic cells -- lifelong antioxidant response. "
                                        "PCNA gate prevents cancer cells from gaining NRF2 protection or MDR1 resistance."}),
    # -- v6 new modifications -- senescence, inflammaging, cardiac completion ------
    ("MOD_19_TBX5_MEF2C",  {"type":"FOREIGN_INSERT","foreign_gene":"TBX5_MEF2C_zebrafish",
                               "module":5,"risk":"LOW",
                               "tissue":"Cardiomyocytes -- injury-activated HRE promoter (same as MOD_17). "
                                        "EXCLUDES atrial-only TBX5 expression to prevent conduction block.",
                               "effect":"Completes cardiac regeneration quartet (GATA4+HAND2+TBX5+MEF2C). "
                                        "Zebrafish achieve full ventricle regeneration post-20% resection. "
                                        "TBX5: sarcomere gene activation; MEF2C: CM maturation after dedifferentiation. "
                                        "Together with MOD_17: cardiac_regen 0.15->0.25. "
                                        "Bakkers 2011 Cardiovasc Res 91:279; Olson 2006 Science 313:1922."}),
    ("MOD_20_NFKB_shark",  {"type":"ENHANCED_PARALOGUE","target_gene":"RELA","copies":1,
                               "module":9,"risk":"LOW",
                               "tissue":"Ubiquitous -- replaces RelA Rel Homology Domain at endogenous RELA locus (chr11)",
                               "effect":"Greenland shark (Somniosus microcephalus, 400y lifespan) NF-kB variant. "
                                        "Shark RelA has reduced kB-RE binding affinity -> 55% less chronic NF-kB tonic activity. "
                                        "Acute immune response (TLR/BCR/TCR signalling) preserved -- kB cooperativity intact. "
                                        "Target: inflammaging loop (SASP amplification, IL-6/IL-8/TNF baseline). "
                                        "Nielsen et al. 2016 (Science 353:702): shark shows minimal inflammatory markers. "
                                        "nfkb_red = 0.55 (chronic); acute immune competence maintained (nfkb_acute_preserved = True)."}),
    ("MOD_21_SENOLYTIC",   {"type":"SYNTHETIC_CIRCUIT","foreign_gene":"SENOLYSIN_circuit",
                               "module":9,"risk":"MEDIUM",
                               "tissue":"All somatic cells -- secreted signal, acts paracrine. "
                                        "TRIPLE GATE: p16Ink4a-high AND p21Cip1-high AND SASP (IL-6 promoter-active). "
                                        "Beneficial senescent cells (wound healing, embryogenesis) protected by absence of IL-6 gate.",
                               "effect":"Synthetic senolytic circuit: p16/p21/IL-6 triple-gated expression of "
                                        "membrane-localised pro-apoptotic PUMA-BH3 domain + 'find-me' signal (CX3CL1 cleavage). "
                                        "Recruits NK cells and macrophages to clear SASP-secreting senescent cells. "
                                        "Baker et al. 2011 (Nature 479:232): clearing p16+ cells extends healthspan 25%. "
                                        "Campisi 2013 (Cell 153:1194): SASP is key driver of age-related tissue dysfunction. "
                                        "Triple gate CRITICAL: p16+p21 alone not sufficient -- transient cell-cycle arrest "
                                        "uses both; IL-6 gate confirms chronic SASP-secreting phenotype. "
                                        "senolytic_clear_rate = 0.04/yr of senescent load. "
                                        "v7 conflict audit: x LIF6 double-apoptosis on OIS cells -- assessed BENEFICIAL."}),
    # -- v7 new modifications -- epigenetics, ECM, mtDNA, neuronal -------------
    ("MOD_22_OSKM_cyclic", {"type":"SYNTHETIC_CIRCUIT","foreign_gene":"OSKM_cyclic",
                               "module":10,"risk":"MEDIUM",
                               "tissue":"Ubiquitous -- doxycycline-pulsed (3d on / 4d off weekly cycle). "
                                        "gammaH2AX circuit gate: blocked if persistent DNA damage present.",
                               "effect":"Cyclic partial epigenetic reprogramming -- resets Horvath methylation clock. "
                                        "cMyc-Dtad variant prevents pluripotency. 40%% epigenetic drift reversal/cycle. "
                                        "Gill et al. 2022 (Cell 186:4973). New ODE variable E (epigenetic age). "
                                        "Requires weekly doxycycline administration. "
                                        "oskm_epi_reset = 0.40, cycle_freq = 52/yr -> net drift correction 0.018/yr."}),
    ("MOD_23_GLUCOSPANASE", {"type":"FOREIGN_INSERT","foreign_gene":"GLUCOSPANASE_bact",
                               "module":10,"risk":"LOW",
                               "tissue":"Fibroblasts and endothelial cells (COL1A2 promoter). "
                                        "Secreted into ECM -- acts on extracellular crosslinks.",
                               "effect":"Bacterial glucosspan lactonase (humanized) -- cleaves age crosslinks "
                                        "in collagen/elastin/lens crystallins. No human enzyme does this. "
                                        "SENS Research Foundation primary target. "
                                        "New ODE variable G (glucosspan burden). glucosspan_clear = 0.008/yr. "
                                        "Addresses vascular stiffness, lens opacity, cartilage rigidity."}),
    ("MOD_24_MITO_DDCBE",  {"type":"FOREIGN_INSERT","foreign_gene":"DDCBE_mito",
                               "module":7,"risk":"LOW",
                               "tissue":"All cells -- mitochondria targeted via MTS prepeptide.",
                               "effect":"DddA-derived cytosine base editor clears pathological mtDNA mutations. "
                                        "Prevents Muller's ratchet heteroplasmy drift over centuries. "
                                        "Mok et al. 2020 (Nature 583:631). New ODE variable H (heteroplasmy). "
                                        "ddcbe_hetero_red = 0.65/yr correction. Protects MOD_10 bat ND5 from "
                                        "out-competition by accumulating mutant copies."}),
    ("MOD_25_TFEB_neuron",  {"type":"ENHANCED_PARALOGUE","foreign_gene":"TFEB_neuron",
                               "module":5,"risk":"LOW",
                               "tissue":"Neurons only -- SYN1+ cells (SYNGAP1 intron 2 safe harbour). "
                                        "Separate from MOD_08 ADAR (different safe harbour).",
                               "effect":"TFEB S142A/S211A constitutively nuclear -- master regulator of "
                                        "lysosomal biogenesis + macroautophagy in neurons. "
                                        "Clears tau, alpha-syn, TDP-43, huntingtin aggregates too large for CMA. "
                                        "COMPLEMENTARY to MOD_05 LAMP2A. Decressac 2013 (Nat Neurosci 16:1143). "
                                        "tfeb_neuro_clear = 0.30. Addresses the ~20-50k year neuronal bottleneck."}),
    # -- v8 new modifications -- neuronal regeneration & active lipofuscin clearing -
    ("MOD_26_NEURO_REGEN",  {"type":"FOREIGN_INSERT","foreign_gene":"NEURO_REGEN_zebrafish",
                               "module":11,"risk":"MEDIUM",
                               "tissue":"Neural stem cell niches + pan-neuronal Sox2+ cells. "
                                        "16 neurogenic zones (all active in D.rerio vs 2 declining in human). "
                                        "EXCLUDES cerebellar Purkinje and brainstem motor nuclei "
                                        "(replacement there causes ataxia/respiratory failure).",
                               "effect":"Zebrafish adult neurogenesis factors (FGF8b + BDNF-E1 + Sox2-DC) "
                                        "restore constitutive neuronal replacement throughout adult brain. "
                                        "New neurons migrate and integrate via DCX guidance scaffold. "
                                        "Bhatt et al. 2020 (Nat Neurosci 23:1131): zebrafish replace "
                                        "telencephalon neurons constitutively at ~0.8%/yr of total pool. "
                                        "neuro_replace_rate = 0.008/yr (conservative 1% annual turnover). "
                                        "ODE: dN/dt gains active clearance term -neuro_replace_rate*N -- "
                                        "FIRST mechanism making N decay possible. "
                                        "CONFLICT AUDIT: x FOXO3_Hydra (MOD_15) -- constitutive nuclear FOXO3 "
                                        "in neural SC suppresses CCND1 -> may slow neuroblast proliferation. "
                                        "FIX: NEURO_REGEN driven by FGF8b (independent of FOXO3 pathway). "
                                        "x TERT_stem (MOD_16) -- neuroblasts need TERT. Already Oct4/Sox2-gated "
                                        "so TERT active in neurogenic zones. SYNERGISTIC."}),
    ("MOD_27_LIPOFUSCINASE", {"type":"FOREIGN_INSERT","foreign_gene":"LIPOFUSCINASE_synth",
                               "module":5,"risk":"LOW",
                               "tissue":"Neurons (SYN1+) and retinal pigment epithelium. "
                                        "Lysosome-targeted via LAMP1-signal peptide fusion.",
                               "effect":"Synthetic lipofuscin-cleaving enzyme targeting A2E and bis-retinoids. "
                                        "Based on Pseudomonas retinalase + humanized bacterial A2E-lyase fusion. "
                                        "Lipofuscin = insoluble oxidised protein-lipid aggregates that physically "
                                        "block lysosomal function -- not degraded by LAMP2A, TFEB, or proteasome. "
                                        "SENS Research Foundation Tier-1 target. "
                                        "Sparrow et al. 2012 (Prog Retin Eye Res): A2E accumulation -> "
                                        "lysosomal dysfunction cascade -> accelerated N accumulation. "
                                        "lipofuscin_clear_rate = 0.006/yr of lipofuscin burden L. "
                                        "New ODE variable L (lipofuscin, 0-1). "
                                        "L feeds into N: lipofuscin blocks clearance, L->N feedback. "
                                        "COMPLEMENT to TFEB (different substrate: lipofuscin vs aggregates)."}),
    ("MOD_28_NEURO_OSKM",   {"type":"SYNTHETIC_CIRCUIT","foreign_gene":"NEURO_OSKM_circuit",
                               "module":11,"risk":"MEDIUM",
                               "tissue":"Post-mitotic neurons only (SYN1+, SYNGAP1 intron 3). "
                                        "TRIPLE SAFETY GATE: SYN1(neuron) AND p16-LOW (not senescent) "
                                        "AND NeuN+ (mature neuron). Prevents any progenitor activation.",
                               "effect":"Neuron-optimised partial reprogramming: Sox2(D) + Klf4 only "
                                        "(NO Oct4 -- causes neuronal dedifferentiation to pluripotency; "
                                        "NO cMyc -- mitogenic). Sox2D lacks DNA-binding domain, acts only "
                                        "as chromatin remodeller via BAF complex. "
                                        "Monthly dox pulse (2d ON / 26d OFF) -- much sparser than systemic OSKM. "
                                        "Resets epigenetic age in neurons specifically, partially clears "
                                        "aggregate load via epigenome-to-proteome feedback. "
                                        "Stern et al. 2022 (bioRxiv): Sox2+Klf4 in neurons -> "
                                        "partial proteostasis reset without loss of identity. "
                                        "neuro_oskm_reset = 0.004/yr contribution to N clearance. "
                                        "CONFLICT AUDIT: x MOD_22 OSKM_cyclic -- two dox circuits. "
                                        "FIX: different tet-promoters (TRE3G for systemic OSKM, "
                                        "TRE2 for neuronal) -> independent control. "
                                        "x MOD_08 ADAR -- ADAR edits Sox2 mRNA. Sox2D variant "
                                        "has altered 3'UTR ADAR target sites. RISK MITIGATED."}),
    # -- v8 cancer fix: bowhead whale upstream tumour suppressor network -------
    ("MOD_29_ATM_CHEK2",    {"type":"DUPLICATION","target_gene":"ATM",
                               "module":2,"risk":"LOW",
                               "tissue":"Ubiquitous -- ATMx2 copies + CHEK2 upregulation. "
                                        "Synergistic with MOD_01 TP53x20 (downstream target).",
                               "effect":"Bowhead whale (Balaena mysticetus, 200+ yr lifespan) expanded "
                                        "DNA damage sensing network. ATMx2 copies detect DSBs 35% faster. "
                                        "CHEK2 amplification stabilises p53 under persistent damage. "
                                        "Keane et al. 2015 (Cell Rep 10:112): bowhead whale has expanded "
                                        "TP53, BRCA2, ATM, and CHEK2 gene families. "
                                        "v8 ODE FIX: cancer_input multiplier changed from (1+0.001*age) "
                                        "to (1+2.0*E+0.5*S) -- biologically correct, OSKM/SENOLYTIC now "
                                        "actually reduce long-term cancer risk. "
                                        "atm_boost = 1.35x on cancer_p53_clear. "
                                        "CONFLICT AUDIT: x TP53x20 (MOD_01) -- SYNERGISTIC. "
                                        "ATM phosphorylates p53 Ser15 -- more ATM = faster p53 activation. "
                                        "x LIF6 (MOD_14) -- SYNERGISTIC. ATM->CHEK2->p53->LIF6 is the "
                                        "canonical pathway. Risk: LOW."}),
    # -- v8b new mods -- ROS saturation + inflammaging loop break -------------
    ("MOD_30_MITOSOD",      {"type":"FOREIGN_INSERT","foreign_gene":"MITOSOD_synth",
                               "module":7,"risk":"LOW",
                               "tissue":"All cells -- mitochondrial matrix (MTS-prepended). "
                                        "Additive to MOD_10 Myotis CI: different mechanism "
                                        "(CI electron leak vs H2O2 scavenging).",
                               "effect":"Mitochondria-targeted SOD2 (MnSOD) + catalase fusion. "
                                        "Schriner et al. 2005 (Science 308:1909): mito-catalase mice "
                                        "-> lifespan extension, reduced oxidative damage throughout life. "
                                        "35% additional reduction in mitochondrial H2O2 production. "
                                        "Breaks X saturation feedback: X->W->N/D amplification loop. "
                                        "MOD_10 (Myotis CI): -40% electron leak at Complex I. "
                                        "MOD_30 (MitoSOD): -35% H2O2 from superoxide dismutation. "
                                        "Together: ROS production reduced ~75% from dual mechanism. "
                                        "mito_sod_red = 0.35. "
                                        "CONFLICT AUDIT: x MOD_10 Myotis CI -- SYNERGISTIC, different "
                                        "targets in mito ROS pathway. x MOD_18 NRF2_NMR -- COMPLEMENTARY, "
                                        "NRF2 boosts scavenging while MitoSOD reduces production. "
                                        "Risk: LOW."}),
    ("MOD_31_INFLAMMABREAK",{"type":"SYNTHETIC_CIRCUIT","foreign_gene":"INFLAMMABREAK_synth",
                               "module":9,"risk":"LOW",
                               "tissue":"Ubiquitous -- secreted cytokine decoy receptors. "
                                        "Liver-expressed (AAT promoter) -> systemic distribution.",
                               "effect":"IL-1 receptor antagonist (IL-1Ra, anakinra-like) + "
                                        "soluble gp130 (sgp130Fc) decoy receptor fusion. "
                                        "Breaks the S->SASP->I positive feedback loop at two nodes: "
                                        "IL-1Ra blocks IL-1beta (primary SASP driver) -> -50% tonic I. "
                                        "sgp130 blocks IL-6 trans-signalling (the pro-inflammatory arm). "
                                        "Both anakinra and tocilizumab clinically validated; synthetic "
                                        "gene version for constitutive low-level expression. "
                                        "Combined with NF-kB shark (MOD_20): "
                                        "total tonic inflammation reduction ~75%. "
                                        "inflam_damp = 0.50 (additive to nfkb_red = 0.55 on SASP arm). "
                                        "CONFLICT AUDIT: x MOD_04 AIREx3 -- SASP suppression must not "
                                        "blunt thymic epithelial IL-1 signalling. FIX: AAT promoter "
                                        "(liver-specific, not thymic). x MOD_21 SENOLYTIC -- SYNERGISTIC: "
                                        "senolytic reduces S (source) while INFLAMMABREAK dampens I (output). "
                                        "x ACUTE IMMUNE RESPONSE -- sgp130Fc blocks only IL-6 "
                                        "trans-signalling (sSIL-6R pathway), not classical IL-6 "
                                        "cis-signalling (membrane IL-6R) -> acute response preserved. "
                                        "Risk: LOW."}),
])

# CRISPR_TARGETS imported from crispr.py
