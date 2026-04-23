#!/usr/bin/env python3
"""
pipeline/pipeline_report.py
Generates complete pipeline report combining sequences, constructs, and delivery.
"""

import os, json
from datetime import datetime

_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, '..', 'output_final', 'pipeline')
os.makedirs(OUT_DIR, exist_ok=True)


def generate_report(seq_results: dict, constructs: dict, schedule: list,
                    output_dir: str = OUT_DIR) -> str:
    lines = []
    W = 74
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    def h1(t): lines.extend(['='*W, f'  {t}', '='*W])
    def h2(t): lines.extend(['', f'-- {t} ' + '-'*max(0, W-4-len(t))])
    def row(k, v): lines.append(f'  {k:<28} {v}')

    h1('HOMO PERPETUUS v8.0 -- MOLECULAR PIPELINE REPORT')
    lines.append(f'  Generated: {ts}')
    lines.append(f'  Modifications: {len(constructs)}  |  Phases: {len(schedule)}')
    lines.append('')

    # -- Summary --------------------------------------------------------------
    h2('SUMMARY')
    online  = sum(1 for v in seq_results.values() if v.get('online'))
    synth   = sum(1 for v in seq_results.values() if v.get('source') == 'SYNTHETIC')
    offline = len(seq_results) - online - synth
    aav_ok  = sum(1 for c in constructs.values() if 'AAV compatible' in c.get('aav_status',''))
    need_split = sum(1 for c in constructs.values() if 'split' in c.get('aav_status','').lower() or 'too large' in c.get('aav_status','').lower())

    row('Sequences from NCBI (online):', f'{online}')
    row('Synthetic placeholders:',       f'{synth} (require designed sequences)')
    row('Offline placeholders:',         f'{offline} (require NCBI access)')
    row('AAV-compatible constructs:',    f'{aav_ok}/{len(constructs)}')
    row('Require split-AAV/lentivirus:', f'{need_split}')
    lines.append('')

    # -- Constructs by phase ---------------------------------------------------
    h2('DELIVERY SCHEDULE -- 7 PHASES')
    lines.append('')
    lines.append(f'  {"#":>2}  {"Mod":30} {"Vector":14} {"Insert":>8}  {"Route"}')
    lines.append('  ' + '-'*72)

    for phase in schedule:
        lines.append(f'\n  {phase["name"]}')
        for m in phase['mods']:
            size = f'{m["insert_bp"]}bp' if isinstance(m['insert_bp'], int) else '?'
            lines.append(f'  P{phase["phase"]}  {m["mod"]:30} {m["vector"]:14} {size:>8}  {m["route"]}')

    # -- Individual construct details ------------------------------------------
    h2('CONSTRUCT DETAILS')
    for name, c in constructs.items():
        if 'error' in c:
            lines.append(f'\n  {name}: ERROR -- {c["error"]}')
            continue
        lines.append(f'\n  [{name}]')
        lines.append(f'    Source:    {c["source_organism"]}  ({c["accession"]})')
        lines.append(f'    AA length: {c["aa_length"]}  CDS: {c["cds_bp"]}bp  Insert: {c["insert_bp"]}bp')
        lines.append(f'    Promoter:  {c["promoter"]} ({c["promoter_info"]})')
        lines.append(f'    Site:      {c["insertion_site"]}')
        lines.append(f'    AAV:       {c["aav_status"]}')
        lines.append(f'    Notes:     {c["notes"]}')
        seq_note = seq_results.get(name, {}).get('note', '')
        if seq_note:
            lines.append(f'    WARN Seq:     {seq_note}')

    # -- Critical issues -------------------------------------------------------
    h2('CRITICAL ISSUES REQUIRING ATTENTION')
    issues = []

    for name, c in constructs.items():
        if 'too large' in c.get('aav_status', '') or 'split' in c.get('aav_status', '').lower():
            issues.append(f'  [SIZE] {name}: {c["insert_bp"]}bp -- {c["aav_status"]}')

    for name, seq in seq_results.items():
        if not seq.get('online') and seq.get('source') != 'SYNTHETIC':
            issues.append(f'  [SEQ]  {name}: offline placeholder -- need real {seq.get("accession","?")} from NCBI')

    for issue in issues:
        lines.append(issue)

    if not issues:
        lines.append('  None -- all constructs validated.')

    # -- Next steps ------------------------------------------------------------
    h2('NEXT STEPS')
    steps = [
        '1. SEQUENCE VERIFICATION',
        '   - Run with NCBI internet access to replace all offline placeholders',
        '   - Verify each protein sequence matches expected length +/- 5%',
        '   - For foreign genes: run BLAST to confirm ortholog identity > 70%',
        '',
        '2. CODON OPTIMIZATION (per mod)',
        '   - Submit each CDS to IDT Codon Optimization Tool or Benchling',
        '   - Target: CAI > 0.8 for H.sapiens',
        '   - Avoid internal restriction sites (EcoRI, NotI, XhoI)',
        '   - Remove cryptic splice sites (use ESEfinder)',
        '',
        '3. HOMOLOGY ARM SYNTHESIS',
        '   - Download 800bp genomic flanking sequence from UCSC/Ensembl',
        '     for each insertion site listed in this report',
        '   - Verify no SNPs in homology arms (check dbSNP)',
        '',
        '4. VECTOR PACKAGING',
        '   - Standard AAV: order from Vigene, VectorBioLabs, or UNC Vector Core',
        '   - Split-AAV (ATM, TERT, GLUCOSPANASE): use intein-split approach',
        '   - mitoTALEN (Myotis_MITO_CI): order from Precision BioSciences',
        '',
        '5. GUIDE RNA VALIDATION',
        '   - All 32 guides from CRISPR_TARGETS are pre-designed',
        '   - Validate each in silico: CRISPOR or Benchling CRISPR tool',
        '   - Order as synthetic sgRNA (Synthego or IDT Alt-R system)',
        '',
        '6. SAFETY TESTING ORDER',
        '   - Phase 1 ex vivo first -- lowest systemic risk',
        '   - Each phase: start with lowest-risk mods (LOW-risk only)',
        '   - Minimum 6-month observation between phases',
    ]
    for step in steps:
        lines.append(f'  {step}')

    lines.append('')
    lines.append('='*W)

    report_text = '\n'.join(lines)
    path = os.path.join(output_dir, 'pipeline_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    return path


if __name__ == '__main__':
    import sys; sys.path.insert(0, os.path.join(_DIR, '..'))
    from pipeline.sequence_fetcher import ACCESSIONS, get_sequence
    from pipeline.construct_builder import build_all, export_fasta, export_json
    from pipeline.delivery_planner import full_delivery_schedule, export_schedule

    print('Building full pipeline report...')
    seq_results = {n: get_sequence(n, verbose=True)  for n in ACCESSIONS}
    constructs  = build_all(seq_results, verbose=True)
    schedule    = full_delivery_schedule(constructs, verbose=False)

    fasta_path  = export_fasta(constructs)
    json_path   = export_json(constructs)
    sched_path  = export_schedule(schedule)
    report_path = generate_report(seq_results, constructs, schedule)

    print(f'\nOK FASTA:    {fasta_path}')
    print(f'OK JSON:     {json_path}')
    print(f'OK Schedule: {sched_path}')
    print(f'OK Report:   {report_path}')
