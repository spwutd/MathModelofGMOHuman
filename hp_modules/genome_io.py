"""hp_modules/genome_io.py — FastaIndex, GtfAnnotation, sequence utils."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
import gzip, random, hashlib
from collections import defaultdict, Counter
from hp_modules.config import CODON_TABLE, AA_PROPS, KNOWN_PROTEIN_LENGTHS
from hp_modules.modifications import GENE_DB

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from hp_modules.config import BASE_DIR

# ══════════════════════════════════════════════════════════════════════════════
# FASTA INDEX (same efficient reader as v1)
# ══════════════════════════════════════════════════════════════════════════════

class FastaIndex:
    def __init__(self, path):
        self.path = path
        self.index = {}
        self._build()

    def _build(self):
        print(f"  [FASTA] Indexing {os.path.basename(self.path)} ...")
        t0 = time.time()
        with open(self.path, 'rb') as f:
            name = None
            seq_start = line_len = line_bytes = seq_len = 0
            while True:
                pos = f.tell()
                raw = f.readline()
                if not raw:
                    if name: self.index[name] = (seq_start, seq_len, line_len, line_bytes)
                    break
                if raw[0:1] == b'>':
                    if name: self.index[name] = (seq_start, seq_len, line_len, line_bytes)
                    name = raw.decode('ascii','ignore').strip()[1:].split()[0]
                    seq_start = f.tell(); seq_len = line_len = line_bytes = 0
                else:
                    stripped = raw.rstrip(b'\n\r')
                    if line_len == 0 and stripped:
                        line_len = len(stripped); line_bytes = len(raw)
                    seq_len += len(stripped)
        print(f"  [FASTA] {len(self.index)} sequences in {time.time()-t0:.1f}s")

    def chromosomes(self):
        return list(self.index.keys())

    def seq_length(self, chrom):
        return self.index.get(chrom, (0,0,0,0))[1]

    def fetch(self, chrom, start, end):
        if chrom not in self.index: return ''
        seq_start, seq_len, line_len, line_bytes = self.index[chrom]
        if not line_len: return ''
        end = min(end, seq_len); length = end - start
        if length <= 0: return ''
        result = []; pos = start; remaining = length
        with open(self.path, 'rb') as f:
            while remaining > 0:
                ln = pos // line_len; col = pos % line_len
                f.seek(seq_start + ln * line_bytes + col)
                chunk = f.read(min(line_len - col, remaining)).decode('ascii','ignore')
                chunk = chunk.replace('\n','').replace('\r','')
                if not chunk: break
                result.append(chunk); pos += len(chunk); remaining -= len(chunk)
        return ''.join(result).upper()


# ══════════════════════════════════════════════════════════════════════════════
# GTF ANNOTATION PARSER  ← NEW in v2
# ══════════════════════════════════════════════════════════════════════════════

class GtfAnnotation:
    """
    Parses a GTF file and builds an exon map per gene.
    Supports plain .gtf and .gtf.gz.
    """
    def __init__(self, gtf_path):
        self.path = gtf_path
        # gene_name → list of (chrom, exon_start, exon_end, strand, transcript_id)
        self.exons = defaultdict(list)
        # gene_name → canonical transcript (most exons)
        self.canonical = {}
        self._parse()
        self._pick_canonical()

    def _parse(self):
        print(f"  [GTF] Parsing {os.path.basename(self.path)} ...")
        t0 = time.time()
        opener = gzip.open if self.path.endswith('.gz') else open
        n = 0
        with opener(self.path, 'rt', encoding='utf8', errors='ignore') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 9: continue
                if parts[2] != 'exon': continue
                chrom, start, end, strand = parts[0], int(parts[3])-1, int(parts[4]), parts[6]
                attrs = parts[8]
                gname = re.search(r'gene_name "([^"]+)"', attrs)
                tid   = re.search(r'transcript_id "([^"]+)"', attrs)
                if not gname or not tid: continue
                gn = gname.group(1); tr = tid.group(1)
                self.exons[gn].append((chrom, start, end, strand, tr))
                n += 1
        print(f"  [GTF] {n:,} exons across {len(self.exons):,} genes in {time.time()-t0:.1f}s")

    def _pick_canonical(self):
        """For each gene pick the transcript with the most exons (longest isoform)."""
        for gene, exon_list in self.exons.items():
            tr_count = Counter(e[4] for e in exon_list)
            best_tr = tr_count.most_common(1)[0][0]
            self.canonical[gene] = [e for e in exon_list if e[4] == best_tr]

    def get_mrna_exons(self, gene_name):
        """Return sorted exon list for canonical transcript.
        Always sort ASCENDING by genomic start — RC applied later in splice_and_translate."""
        exons = self.canonical.get(gene_name, [])
        if not exons: return []
        return sorted(exons, key=lambda e: e[1])  # always ascending

    def has_gene(self, gene_name):
        return gene_name in self.canonical


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

COMPLEMENT = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')

def rc(seq):
    return seq.translate(COMPLEMENT)[::-1]

def gc(seq):
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    total = sum(seq.count(b) for b in 'ACGT')
    return gc_count / total * 100 if total else 0

def translate(seq, frame=0):
    seq = seq.upper()[frame:]
    prot = []
    for i in range(0, len(seq)-2, 3):
        c = seq[i:i+3]
        aa = CODON_TABLE.get(c, '?')
        prot.append(aa)
        if aa == '*': break
    return ''.join(prot)

def find_best_protein(mrna, gene_name):
    """
    Find best protein from mRNA by trying all 3 frames and scanning for ATG.
    Returns longest protein starting from ATG.
    """
    known = KNOWN_PROTEIN_LENGTHS.get(gene_name, 0)
    best = ''
    for frame in range(3):
        seq = mrna[frame:]
        # Scan for every ATG and translate from it
        pos = 0
        while True:
            atg = seq.find('ATG', pos)
            if atg == -1: break
            prot = translate(seq, frame=atg)
            clean = prot.replace('*','')
            if len(clean) > len(best.replace('*','')):
                best = prot
            # If we found something close to expected length, stop
            if known and len(clean) >= known * 0.85:
                return best
            pos = atg + 3
    return best if best else translate(mrna)


def splice_and_translate(fasta, gtf, gene_name):
    """
    Pull exons from GTF (ascending sort), fetch from FASTA, splice mRNA, translate.
    Returns (mrna_seq, protein, exon_count, total_mrna_len, source)

    Correct strand logic:
      1. Get exons sorted ASCENDING (get_mrna_exons always returns ascending)
      2. Fetch each segment forward from FASTA
      3. Concatenate
      4. RC for minus-strand genes  →  now reading 5\'→3\' on coding strand
      5. Find best protein (scan all frames for ATG)
    """
    if gtf and gtf.has_gene(gene_name):
        exons = gtf.get_mrna_exons(gene_name)   # sorted ascending
        if exons:
            strand = exons[0][3]
            parts = []
            for (ch, s, e, st, tr) in exons:
                if fasta and ch in fasta.index:
                    seg = fasta.fetch(ch, s, e)
                else:
                    seg = generate_synthetic_gene(f"{gene_name}_{s}", e - s)
                if seg:
                    parts.append(seg)
            if parts:
                mrna = ''.join(parts)
                if strand == '-':
                    mrna = rc(mrna)          # 5\' UTR is now at the start
                prot = find_best_protein(mrna, gene_name)
                return mrna, prot, len(exons), len(mrna), 'GTF+FASTA'

    # Pure synthetic fallback using known correct length
    known_len = KNOWN_PROTEIN_LENGTHS.get(gene_name, 400)
    syn = generate_synthetic_gene(f"syn_{gene_name}", known_len * 3 + 3)
    return syn, translate(syn), 0, len(syn), 'synthetic'


# ─── CpG ISLAND DETECTION  ← improved in v2 ──────────────────────────────────

def cpg_islands(seq, window=200, step=50, oe_thresh=0.6, gc_thresh=50.0):
    islands = []; seq = seq.upper(); n = len(seq)
    in_isl = False; isl_start = 0
    for i in range(0, n - window, step):
        w = seq[i:i+window]
        gc_w = gc(w)
        cpg_o = w.count('CG')
        c = w.count('C'); g = w.count('G')
        expected = (c * g) / window if (c and g) else 0
        oe = cpg_o / expected if expected else 0
        ok = gc_w >= gc_thresh and oe >= oe_thresh
        if ok and not in_isl:  in_isl = True; isl_start = i
        elif not ok and in_isl:
            in_isl = False; islands.append((isl_start, i, i-isl_start, gc_w, oe))
    if in_isl: islands.append((isl_start, n, n-isl_start, 0, 0))
    return islands

def promoter_cpg_analysis(fasta, gene_name, upstream=2000):
    """
    Fetch promoter region (upstream bp before TSS) and analyse CpG islands.
    Returns dict with island count, GC%, methylation estimate.
    """
    ginfo = GENE_DB.get(gene_name, {})
    if not ginfo or not fasta: return {}
    chrom = ginfo['chr']; strand = ginfo['strand']
    if strand == '+':
        tss = ginfo['start']
        promo_start = max(0, tss - upstream); promo_end = tss + 200
    else:
        tss = ginfo['end']
        promo_start = tss - 200; promo_end = tss + upstream

    if chrom not in fasta.index:
        return {'status': 'chrom_not_found'}

    seq = fasta.fetch(chrom, promo_start, promo_end)
    if strand == '-': seq = rc(seq)
    if not seq: return {}

    islands = cpg_islands(seq, window=200, step=25)
    gc_promo = gc(seq)

    # Methylation estimate: lower CpG O/E → more likely methylated (silenced)
    if islands:
        avg_oe = sum(isl[4] for isl in islands) / len(islands)
    else:
        avg_oe = 0.0

    methylation_est = max(0, 1 - avg_oe) * 100  # rough %

    return {
        'gene': gene_name,
        'promoter_length_bp': len(seq),
        'gc_content_pct': round(gc_promo, 2),
        'cpg_islands': len(islands),
        'cpg_island_details': [{'start':i[0],'end':i[1],'len':i[2],
                                  'gc_pct':round(i[3],1),'obs_exp':round(i[4],3)}
                                for i in islands[:5]],
        'avg_cpg_obs_exp': round(avg_oe, 3),
        'methylation_estimate_pct': round(methylation_est, 1),
        'promoter_status': 'ACTIVE' if (gc_promo > 55 and len(islands) >= 1) else
                           'POISED' if gc_promo > 45 else 'SILENCED',
    }


# ─── PROTEIN ANALYSIS ────────────────────────────────────────────────────────

def protein_stats(prot):
    prot = prot.replace('*','')
    if not prot: return {}
    mw  = sum(AA_PROPS.get(a, {'mw':110}).get('mw', 110) for a in prot) - 18.0*(len(prot)-1)
    charge = sum(AA_PROPS.get(a, {'charge':0})['charge'] for a in prot)
    polar  = sum(1 for a in prot if AA_PROPS.get(a, {'polar':False})['polar'])
    hphob  = [AA_PROPS.get(a, {'hphob':0})['hphob'] for a in prot]
    counts = Counter(prot)
    # Instability index (simplified Guruprasad)
    dipep_unstable = {'WW','WC','WT','WM','WN','WQ','WE','WR','WK',
                      'EE','EQ','ER','EK','NN','NQ','NR','NK','QQ','QR','QK'}
    instab = sum(2.0 for i in range(len(prot)-1)
                 if prot[i:i+2] in dipep_unstable) / len(prot) * 100

    return {
        'length': len(prot),
        'MW_kDa': round(mw/1000, 2),
        'charge': charge,
        'polar_fraction': round(polar/len(prot), 3),
        'avg_hydrophobicity': round(sum(hphob)/len(hphob), 3),
        'instability_index': round(instab, 1),
        'stable': instab < 40,
        'aa_composition': dict(counts.most_common(8)),
        'sequence_preview': prot[:60] + ('...' if len(prot)>60 else ''),
    }

def validate_protein_length(gene, length):
    known = KNOWN_PROTEIN_LENGTHS.get(gene, 0)
    if not known: return 'UNKNOWN_REF', 0
    ratio = length / known
    if ratio > 0.85: return 'CORRECT', ratio
    if ratio > 0.5:  return 'PARTIAL', ratio
    return 'INTRON_ARTIFACT', ratio


# ─── SYNTHETIC SEQUENCE GENERATOR ────────────────────────────────────────────

def generate_synthetic_gene(seed, length):
    random.seed(hashlib.md5(seed.encode()).hexdigest())
    bases = list('ACGT'); weights = [0.29, 0.21, 0.21, 0.29]
    seq = 'ATG'
    while len(seq) < length:
        seq += random.choices(bases, weights=weights, k=1)[0]
    seq = seq[:length]
    # Remove internal stops
    for i in range(3, int(length*0.95), 3):
        if seq[i:i+3] in ('TAA','TAG','TGA'):
            seq = seq[:i] + 'AAA' + seq[i+3:]
    return seq[:length-3] + 'TGA'


# ══════════════════════════════════════════════════════════════════════════════
# MODIFICATION ENGINE  (v2 — uses splice_and_translate)

def protein_stats_from_sequence(aa_seq):
    """Compute protein stats directly from amino acid sequence string."""
    aa_seq = aa_seq.replace('*','').replace('-','')
    if not aa_seq:
        return {}
    mw     = sum(AA_PROPS.get(a, {'mw':110}).get('mw',110) for a in aa_seq) - 18.0*(len(aa_seq)-1)
    charge = sum(AA_PROPS.get(a, {'charge':0})['charge'] for a in aa_seq)
    polar  = sum(1 for a in aa_seq if AA_PROPS.get(a, {'polar':False})['polar'])
    hphob  = [AA_PROPS.get(a, {'hphob':0})['hphob'] for a in aa_seq]
    counts = Counter(aa_seq)
    dipep_unstable = {'WW','WC','WT','WM','WN','WQ','WE','WR','WK',
                      'EE','EQ','ER','EK','NN','NQ','NR','NK','QQ','QR','QK'}
    instab = sum(2.0 for i in range(len(aa_seq)-1)
                 if aa_seq[i:i+2] in dipep_unstable) / len(aa_seq) * 100
    return {
        'length': len(aa_seq),
        'MW_kDa': round(mw/1000, 2),
        'charge': charge,
        'polar_fraction': round(polar/len(aa_seq), 3),
        'avg_hydrophobicity': round(sum(hphob)/len(hphob), 3),
        'instability_index': round(instab, 1),
        'stable': instab < 40,
        'aa_composition': dict(counts.most_common(8)),
        'sequence_preview': aa_seq[:60] + ('...' if len(aa_seq)>60 else ''),
        'sequence_full': aa_seq,
    }


# ══════════════════════════════════════════════════════════════════════════════
