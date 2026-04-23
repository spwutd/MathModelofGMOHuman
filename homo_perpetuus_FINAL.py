#!/usr/bin/env python3
"""
homo_perpetuus_FINAL.py  —  HOMO PERPETUUS v8.0 FINAL  —  entry point
All logic lives in hp_modules/. This file is ~100 lines.
Run:  python3 homo_perpetuus_FINAL.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hp_modules.config import (
    BASE_DIR, OUTPUT_DIR, DARK_BG, PANEL_BG,
    BLUE, GREEN, ORANGE, PURPLE, RED, CYAN, YELLOW, GREY, LIGHT,
    CODON_TABLE, AA_PROPS, KNOWN_PROTEIN_LENGTHS,
    FASTA_CANDIDATES, GTF_CANDIDATES,
)
from hp_modules.modifications import MODIFICATIONS, FOREIGN_GENES, GENE_DB
from hp_modules.ncbi_api import (
    UNIPROT_ACCESSIONS, NCBI_FOREIGN_ACCESSIONS, FOREIGN_EXPECTED_LENGTHS,
    GTEX_TISSUES_OF_INTEREST,
    fetch_uniprot_sequence, get_protein_sequence, get_protein_sequence_extended,
    fetch_ncbi_protein, fetch_ensembl_cds,
    fetch_gtex_expression, fetch_gtex_for_all_genes, get_gtex_data,
)
from hp_modules.genome_io import (
    FastaIndex, GtfAnnotation, rc, gc, translate,
    find_best_protein, splice_and_translate,
    cpg_islands, promoter_cpg_analysis,
    protein_stats, validate_protein_length, generate_synthetic_gene,
)
from hp_modules.modification_engine import ModificationEngine
from hp_modules.simulation_models   import SimulationModels
from hp_modules.ode_engine          import ModuleCrosstalk, sim_neuronal_ceiling
from hp_modules.plots import (
    save_fig, style_ax, plot_genome, plot_mods_overview,
    plot_protein_validation, plot_promoter_cpg, plot_simulations,
    plot_protein_summary, plot_module_interactions,
    plot_v5_mechanisms, plot_v6_mechanisms, plot_v7_mechanisms,
    plot_gtex_expression, plot_module_crosstalk, plot_ai_risk_dashboard,
)
from hp_modules.report import generate_report
from hp_modules.crispr import (
    CRISPR_TARGETS, run_crispr_offtarget,
    plot_crispr_offtarget, generate_crispr_report,
)
from hp_modules.main import banner, find_file, main

if __name__ == '__main__':
    main()
