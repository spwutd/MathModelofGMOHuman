"""hp_modules/modification_engine.py — ModificationEngine: processes each mod."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
from hp_modules.config import KNOWN_PROTEIN_LENGTHS
from hp_modules.genome_io import (cpg_islands, generate_synthetic_gene,
                                   find_best_protein, promoter_cpg_analysis,
                                   protein_stats_from_sequence)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from hp_modules.config import BASE_DIR, OUTPUT_DIR
from hp_modules.modifications import MODIFICATIONS, FOREIGN_GENES, GENE_DB
from hp_modules.genome_io import FastaIndex, GtfAnnotation, protein_stats
from hp_modules.ncbi_api import (fetch_ncbi_protein, get_protein_sequence,
                                   get_protein_sequence_extended)

class ModificationEngine:
    def __init__(self, fasta=None, gtf=None):
        self.fasta = fasta
        self.gtf   = gtf
        self.results = []
        self.promoter_data = {}

    def _analyse_gene(self, gene_name):
        """
        Get protein data using priority chain:
          1. UniProt API (real validated sequence)
          2. GTF+FASTA splice
          3. Synthetic correct-length fallback
        """
        aa_seq, aa_len, source = get_protein_sequence_extended(gene_name, self.fasta, self.gtf)
        pstats = protein_stats_from_sequence(aa_seq)

        # CDS info from GTF for diagnostics
        cds_count = cds_total_bp = 0
        if self.gtf and self.gtf.has_gene(gene_name):
            segs = self.gtf.canonical.get(gene_name, [])
            cds_count = len(segs)
            cds_total_bp = sum(e[2]-e[1] for e in segs)

        # Validation
        known = KNOWN_PROTEIN_LENGTHS.get(gene_name, 0)
        if known:
            ratio = aa_len / known
            if ratio > 0.85:   val_status = 'CORRECT'
            elif ratio > 0.50: val_status = 'PARTIAL'
            else:              val_status = 'INTRON_ARTIFACT'
        else:
            val_status = 'UNKNOWN_REF'; ratio = 0

        # CpG islands in a synthetic representation of the CDS
        dummy_mrna = 'ATG' + 'GCG' * (aa_len // 3)  # GC-neutral placeholder
        cpg = cpg_islands(dummy_mrna[:3000])

        return {
            'mrna_length': aa_len * 3,
            'exon_count': cds_count,
            'cds_total_bp_gtf': cds_total_bp,
            'source': source,
            'gc_content_pct': 50.0,  # not applicable for AA-sourced data
            'cpg_islands_in_cds': len(cpg),
            'protein': pstats,
            'validation_status': val_status,
            'validation_ratio': round(ratio, 3) if ratio else 0,
            'sequence_preview': aa_seq[:60] + '...',
        }

    def _analyse_foreign(self, fg_name):
        fg = FOREIGN_GENES[fg_name]

        # Priority: NCBI API → seed-based synthetic
        ncbi_result = fetch_ncbi_protein(fg_name)
        if ncbi_result:
            aa_seq, aa_len, accession = ncbi_result
            pstats = protein_stats_from_sequence(aa_seq)
            source = f'NCBI:{accession}'
        else:
            # Seed-based synthetic (deterministic, reproducible)
            dna = generate_synthetic_gene(fg['seed'], fg['length_bp'])
            aa_seq = find_best_protein(dna, '').replace('*','')
            pstats = protein_stats_from_sequence(aa_seq)
            source = 'synthetic_from_seed'

        return {
            'mrna_length': fg['length_bp'],
            'exon_count': 'N/A (foreign)',
            'source': source,
            'gc_content_pct': 50.0,
            'cpg_islands_in_cds': 0,
            'protein': pstats,
            'validation_status': 'FOREIGN_GENE',
            'validation_ratio': 1.0,
            'sequence_preview': aa_seq[:60] + '...',
            'source_organism': fg.get('source',''),
        }

    def run(self):
        print("\n  Applying modifications...\n")
        for mod_id, mod in MODIFICATIONS.items():
            t = mod['type']
            print(f"  ▶  {mod_id}")
            r = {'mod_id': mod_id, 'type': t,
                 'module': mod.get('module'), 'risk': mod.get('risk'),
                 'effect': mod.get('effect','')}

            if t in ('DUPLICATION','UPREGULATION','ENHANCED_PARALOGUE',
                     'CONDITIONAL_KNOCKOUT','CONDITIONAL_ACTIVATION'):
                gn = mod.get('target_gene','')
                r['gene'] = gn
                if gn in GENE_DB:
                    r.update(self._analyse_gene(gn))
                if t == 'DUPLICATION':
                    r['copies'] = mod.get('copies', 1)
                    r['bp_added'] = r.get('mrna_length', 0) * (mod.get('copies',1) - 1)
                if t == 'CONDITIONAL_KNOCKOUT':
                    r['tissue_specificity'] = mod.get('tissue', 'conditional')
                    r['ko_mechanism'] = 'CRISPR frameshift + Cre-lox conditional'
                if t == 'CONDITIONAL_ACTIVATION':
                    r['trigger'] = mod.get('trigger', 'HRE')

            elif t == 'FOREIGN_INSERT':
                fg = mod.get('foreign_gene','')
                r['foreign_gene'] = fg
                r['source_organism'] = FOREIGN_GENES.get(fg, {}).get('source','')
                r['function'] = FOREIGN_GENES.get(fg, {}).get('function','')
                r['insertion_site'] = FOREIGN_GENES.get(fg, {}).get('insertion','')
                r['promoter'] = FOREIGN_GENES.get(fg, {}).get('promoter','')
                r.update(self._analyse_foreign(fg))

            self.results.append(r)

        # Promoter CpG analysis for all target genes
        print("\n  Analysing promoter CpG islands...")
        for gene_name in GENE_DB:
            pa = promoter_cpg_analysis(self.fasta, gene_name)
            if pa: self.promoter_data[gene_name] = pa
        print(f"  ✓ {len(self.promoter_data)} promoters analysed")
        return self.results


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODELS  (v2 — extended survival model)
# ══════════════════════════════════════════════════════════════════════════════