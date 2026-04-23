"""hp_modules/config.py — paths, colours, codon/AA tables."""
import os, sys

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FASTA_CANDIDATES = [
    "/mnt/user-data/uploads/HumanGenome.fa",
    os.path.join(BASE_DIR, "HumanGenome.fa"),
    "HumanGenome.fa",
]
GTF_CANDIDATES = [
    # Plain
    "/mnt/user-data/uploads/gencode.v38.annotation.gtf",
    "/mnt/user-data/uploads/gencode.gtf",
    "/mnt/user-data/uploads/annotation.gtf",
    os.path.join(BASE_DIR, "gencode.v38.annotation.gtf"),
    os.path.join(BASE_DIR, "annotation.gtf"),
    # Gzipped  ← was missing
    "/mnt/user-data/uploads/gencode.v38.annotation.gtf.gz",
    "/mnt/user-data/uploads/gencode.gtf.gz",
    os.path.join(BASE_DIR, "gencode.v38.annotation.gtf.gz"),
    os.path.join(BASE_DIR, "gencode.gtf.gz"),
    os.path.join(BASE_DIR, "annotation.gtf.gz"),
]

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
DARK_BG  = '#0D1117'
PANEL_BG = '#111820'
BLUE     = '#2E9BFF'
GREEN    = '#39D353'
ORANGE   = '#FF7F50'
PURPLE   = '#9966FF'
RED      = '#FF4444'
CYAN     = '#00E5FF'
YELLOW   = '#FFD700'
GREY     = '#8B949E'
LIGHT    = '#C9D1D9'

# ─── CODON / AA TABLES ───────────────────────────────────────────────────────
CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
AA_PROPS = {
    'A':{'mw': 89.1,'charge': 0,'polar':False,'hphob': 1.8},
    'R':{'mw':174.2,'charge':+1,'polar':True, 'hphob':-4.5},
    'N':{'mw':132.1,'charge': 0,'polar':True, 'hphob':-3.5},
    'D':{'mw':133.1,'charge':-1,'polar':True, 'hphob':-3.5},
    'C':{'mw':121.2,'charge': 0,'polar':True, 'hphob': 2.5},
    'E':{'mw':147.1,'charge':-1,'polar':True, 'hphob':-3.5},
    'Q':{'mw':146.2,'charge': 0,'polar':True, 'hphob':-3.5},
    'G':{'mw': 75.0,'charge': 0,'polar':False,'hphob':-0.4},
    'H':{'mw':155.2,'charge':+1,'polar':True, 'hphob':-3.2},
    'I':{'mw':131.2,'charge': 0,'polar':False,'hphob': 4.5},
    'L':{'mw':131.2,'charge': 0,'polar':False,'hphob': 3.8},
    'K':{'mw':146.2,'charge':+1,'polar':True, 'hphob':-3.9},
    'M':{'mw':149.2,'charge': 0,'polar':False,'hphob': 1.9},
    'F':{'mw':165.2,'charge': 0,'polar':False,'hphob': 2.8},
    'P':{'mw':115.1,'charge': 0,'polar':False,'hphob':-1.6},
    'S':{'mw':105.1,'charge': 0,'polar':True, 'hphob':-0.8},
    'T':{'mw':119.1,'charge': 0,'polar':True, 'hphob':-0.7},
    'W':{'mw':204.2,'charge': 0,'polar':False,'hphob':-0.9},
    'Y':{'mw':181.2,'charge': 0,'polar':True, 'hphob':-1.3},
    'V':{'mw':117.1,'charge': 0,'polar':False,'hphob': 4.2},
    '*':{'mw':  0.0,'charge': 0,'polar':False,'hphob': 0.0},
}

# Known real protein lengths for validation
KNOWN_PROTEIN_LENGTHS = {
    'TP53': 393, 'BRCA1': 1863, 'BRCA2': 3418, 'RAD51': 339,
    'ERCC1': 297, 'PCNA': 261, 'MSH2': 934, 'MSH6': 1360,
    'LAMP2': 410, 'SQSTM1': 440, 'GLO1': 184,
    'FOXN1': 648, 'AIRE': 545, 'AR': 919,
    'SOX2': 317, 'NOTCH1': 2555, 'CCND1': 295, 'TERT': 1132, 'FEN1': 380,
    # v5 additions
    'HAS2':   552,   # Hyaluronan synthase 2 (human)
    'FOXO3':  673,   # Forkhead box O3
    'NFE2L2': 605,   # NRF2 transcription factor
    'GATA4':  442,   # GATA binding protein 4
    'HAND2':  217,   # Heart and neural crest derivatives expressed 2
}

# ══════════════════════════════════════════════════════════════════════════════
# UNIPROT API CLIENT  — fetches real validated protein sequences
# ══════════════════════════════════════════════════════════════════════════════

import urllib.request
import urllib.parse
import ssl
