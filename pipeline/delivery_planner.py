#!/usr/bin/env python3
"""
pipeline/delivery_planner.py
Assigns optimal delivery vector and administration route for each construct.
Accounts for: insert size, target tissue, expression persistence needed,
              immune considerations, clinical precedent.
"""

import os, json
_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, '..', 'output_final', 'pipeline')
os.makedirs(OUT_DIR, exist_ok=True)

# -- Vector specs ---------------------------------------------------------------
VECTORS = {
    'AAV9': {
        'max_insert_bp': 4700,
        'integration':   'episomal (non-integrating)',
        'persistence':   'years (dividing cells lose it; post-mitotic: permanent)',
        'tropism':       'CNS, motor neurons, crosses BBB; systemic',
        'best_for':      'neurons, heart, muscle',
        'immune':        'pre-existing NAbs in ~50% population -- screen first',
        'clinical':      'FDA approved (Zolgensma for SMA)',
        'dose_route':    'IV or intrathecal',
    },
    'AAV8': {
        'max_insert_bp': 4700,
        'integration':   'episomal',
        'persistence':   'years in liver',
        'tropism':       'liver (highly efficient), skeletal muscle',
        'best_for':      'liver, systemic secreted proteins',
        'immune':        'pre-existing NAbs in ~20% population',
        'clinical':      'FDA approved (Hemgenix for hemophilia B)',
        'dose_route':    'IV',
    },
    'AAVrh10': {
        'max_insert_bp': 4700,
        'integration':   'episomal',
        'persistence':   'years',
        'tropism':       'liver, CNS, lung -- lower seroprevalence than AAV8/9',
        'best_for':      'patients with AAV8/9 NAbs',
        'immune':        'lower seroprevalence (~15%)',
        'clinical':      'clinical trials (Alzheimer, Pompe)',
        'dose_route':    'IV or intrathecal',
    },
    'AAV2': {
        'max_insert_bp': 4700,
        'integration':   'episomal',
        'persistence':   'long-term in retina',
        'tropism':       'retinal pigment epithelium, inner retina',
        'best_for':      'eye (lipofuscin in RPE)',
        'immune':        'mild ocular immune response manageable',
        'clinical':      'FDA approved (Luxturna for Leber)',
        'dose_route':    'subretinal injection',
    },
    'AAV6': {
        'max_insert_bp': 4700,
        'integration':   'episomal',
        'persistence':   'years in muscle/HSC',
        'tropism':       'hematopoietic stem cells (ex vivo), muscle',
        'best_for':      'ex vivo HSC editing, then transplant',
        'immune':        'ex vivo context avoids systemic immune response',
        'clinical':      'used in sickle cell trials',
        'dose_route':    'ex vivo + autologous transplant',
    },
    'Lentivirus': {
        'max_insert_bp': 8000,
        'integration':   'stable genomic integration (semi-random)',
        'persistence':   'permanent (integrates)',
        'tropism':       'dividing cells; ex vivo preferred',
        'best_for':      'large constructs, HSC, T cells ex vivo',
        'immune':        'ex vivo avoids systemic immune response',
        'clinical':      'FDA approved (Zynteglo for beta-thalassemia)',
        'dose_route':    'ex vivo only',
    },
    'Split-AAV': {
        'max_insert_bp': 9000,
        'integration':   'episomal (two vectors recombine in nucleus)',
        'persistence':   'years',
        'tropism':       'depends on capsid used (AAV8/9 typically)',
        'best_for':      'large genes that dont fit in single AAV',
        'immune':        'same as base AAV serotype',
        'clinical':      'preclinical/early clinical for dystrophin',
        'dose_route':    'IV (both vectors co-administered)',
    },
    'mitoTALEN': {
        'max_insert_bp': None,
        'integration':   'mitochondrial DNA editing (no integration)',
        'persistence':   'one-time editing event',
        'tropism':       'all cells (mitochondria)',
        'best_for':      'mtDNA heteroplasmy correction',
        'immune':        'low (protein delivered, not viral)',
        'clinical':      'preclinical -- DdCBE approach in development',
        'dose_route':    'lipid nanoparticle IV or electroporation ex vivo',
    },
    'LNP-mRNA': {
        'max_insert_bp': 10000,
        'integration':   'transient (mRNA degraded 2-7 days)',
        'persistence':   'days to weeks',
        'tropism':       'liver primary target; lung, spleen',
        'best_for':      'one-shot CRISPR delivery (Cas9+guide), transient',
        'immune':        'ionizable LNP -- manageable, no viral capsid NAbs',
        'clinical':      'FDA approved (mRNA vaccines, NTLA-2001 ATTR)',
        'dose_route':    'IV',
    },
    'Electroporation': {
        'max_insert_bp': 20000,
        'integration':   'depends on template (HDR = stable)',
        'persistence':   'permanent if HDR successful',
        'tropism':       'any cell type ex vivo',
        'best_for':      'HSC, T cells -- precise HDR with highest efficiency',
        'immune':        'no viral vector -- no capsid immunity',
        'clinical':      'standard of care for CAR-T, CRISPR sickle cell',
        'dose_route':    'ex vivo only',
    },
}

# -- Delivery assignment per mod ------------------------------------------------
# Format: mod_name -> (primary_vector, route, notes, phase_order)
# phase_order: 1=first (ex vivo), 2=systemic AAV liver, 3=CNS, 4=mito, 5=inducible
DELIVERY_PLAN = {
    # PHASE 1 -- ex vivo HSC (safest, most controllable)
    'AR_human':           ('Electroporation', 'ex vivo HSC + thymic progenitors', 'Cre plasmid for conditional KO; thymic epithelial cells targeted via FOXN1+ sorting', 1),
    'AIRE_human':         ('AAV6',   'ex vivo thymic epithelial cells',      'TECs isolated from thymic biopsy, ex vivo AAV6 transduction, reimplant', 1),
    'PIWI_Tdohrnii':      ('AAV6',   'ex vivo HSC -> IV transplant',          'E2F1-gated -- only active in cycling cells', 1),
    'SENOLYSIN_circuit':  ('Electroporation', 'ex vivo HSC + iPSC-derived cells', 'Systemic paracrine effect once cells engraft', 1),

    # PHASE 2 -- systemic AAV (liver first -- highest efficiency)
    'GLO1_NMR':           ('AAV8',   'IV systemic',                          'Liver primary target; secreted FN3K acts systemically', 2),
    'LAMP2A_NMR':         ('AAV8',   'IV systemic',                          'Ubiquitous CMA enhancement; liver/kidney first', 2),
    'LIF6_elephant':      ('AAV8',   'IV systemic (ROSA26 target)',           'p53RE+gammaH2AX dual gate -- only fires on damaged cells', 2),
    'HAS2_NMR':           ('AAV8',   'IV systemic',                          'HDR at chr8 -- LNP-delivered Cas9+HDR template', 2),
    'CD44_NMR':           ('AAV8',   'IV systemic',                          'HDR at chr11 -- paired with HAS2_NMR delivery', 2),
    'NRF2_NMR':           ('AAV8',   'IV systemic',                          'PCNA-gated -- safe in liver, targets post-mitotic cells', 2),
    'RELA_shark':         ('LNP-mRNA', 'IV + Cas9 RNP for HDR',              'RHD domain swap requires HDR; LNP for transient Cas9', 2),
    'GLUCOSPANASE_bact':  ('Split-AAV', 'IV systemic',                       '5258bp exceeds AAV; split-AAV with AAV8 capsid', 2),
    'INFLAMMABREAK':      ('AAV8',   'IV systemic',                          'IL-1Ra+sgp130Fc secreted -- liver as production organ', 2),

    # PHASE 3 -- CNS/neuronal (after systemic established)
    'ADAR_Cephalopod':    ('AAV9',   'intrathecal or IV (AAV9 crosses BBB)', 'SYN1 promoter -- neurons only despite systemic delivery', 3),
    'TFEB_human':         ('AAV9',   'intrathecal',                          'S142A/S211A -- SYN1 gated, SYNGAP1 i2 safe harbour', 3),
    'NEURO_REGEN_FGF8b':  ('AAV9',   'intrathecal',                          'GFAP-promoter targets radial glia/astrocytes in brain', 3),
    'LIPOFUSCINASE':      ('AAV2',   'subretinal + AAV9 intrathecal',        'RPE eye (AAV2) + neurons brain (AAV9) -- two injections', 3),
    'NEURO_OSKM_SK':      ('AAV9',   'intrathecal',                          'TRE2 dox-inducible; NeuN+p16-LOW safety gate', 3),

    # PHASE 4 -- cardiac/muscle-specific
    'GATA4_zebrafish':    ('AAV9',   'intracoronary or IV',                  'HRE gate -- only active post-ischemia; MYH6 safe harbour', 4),
    'HAND2_zebrafish':    ('AAV9',   'intracoronary (bicistronic with GATA4)', 'Same vector as GATA4 -- IRES-linked', 4),
    'TBX5_zebrafish':     ('AAV9',   'intracoronary or IV',                  'TNNT2 i2 safe harbour; HRE+cTnT gate', 4),
    'CCND1_human':        ('AAV9',   'intracoronary',                        'HRE-gated; chr11 insertion', 4),

    # PHASE 5 -- mitochondrial (most technically challenging)
    'Myotis_MITO_CI_ND5': ('mitoTALEN', 'IV LNP or electroporation ex vivo', 'mtDNA editing -- no standard AAV route; DdCBE/mitoTALEN approach', 5),
    'DDCBE_mito':         ('LNP-mRNA', 'IV (nuclear encoded -> mito via MTS)', 'Nuclear integration (ROSA26), MTS prepeptide directs to mito', 5),
    'MITOSOD':            ('AAV8',   'IV systemic (ROSA26 nuclear, MTS->mito)', 'Nuclear encoded fusion protein; MTS directs to mitochondria', 5),

    # PHASE 6 -- gene duplications (large genes)
    'TP53_human':         ('AAV8',   'IV systemic',                          'Extra TP53 copy under EF1alpha; chr17 upstream HDR', 6),
    'ERCC1_human':        ('AAV8',   'IV systemic',                          'NER enhancement; chr19 upstream', 6),
    'RAD51_human':        ('AAV8',   'IV systemic',                          'Extra copy chr15; EF1alpha driven', 6),
    'FEN1_human':         ('AAV8',   'IV systemic',                          'Promoter replacement chr11', 6),
    'ATM_human':          ('Split-AAV', 'IV systemic',                       'ATM 11190bp -- requires split-AAV (two vectors, AAV8 capsid)', 6),
    'TERT_human':         ('Split-AAV', 'ex vivo HSC -> IV',                  'TERT 5418bp -- split-AAV; Oct4/Sox2 SC-specific promoter', 6),

    # PHASE 7 -- inducible systems (last -- require ongoing management)
    'FOXO3_Hydra':        ('AAV8',   'IV systemic',                          'NESTIN+SOX2 gated quiescent SC niches; AAVS1 safe harbour', 7),
    'OSKM_cyclic':        ('AAV9',   'IV or intrathecal',                    'TRE3G dox-inducible; AAVS1 safe harbour; weekly dox regimen', 7),
}

def get_delivery_plan(mod_name: str) -> dict:
    """Return delivery details for one mod."""
    if mod_name not in DELIVERY_PLAN:
        return {
            'mod_name': mod_name,
            'vector': 'UNASSIGNED',
            'route': 'TBD',
            'notes': 'Not yet in delivery plan',
            'phase': 99,
        }
    vec, route, notes, phase = DELIVERY_PLAN[mod_name]
    vspec = VECTORS.get(vec, {})
    return {
        'mod_name':    mod_name,
        'vector':      vec,
        'route':       route,
        'notes':       notes,
        'phase':       phase,
        'vector_spec': {
            'max_insert_bp': vspec.get('max_insert_bp'),
            'integration':   vspec.get('integration'),
            'persistence':   vspec.get('persistence'),
            'clinical':      vspec.get('clinical'),
        },
    }

def full_delivery_schedule(constructs: dict, verbose: bool = True) -> list:
    """
    Generate full ordered delivery schedule.
    Returns list of phases, each with list of mods.
    """
    phase_names = {
        1: 'Phase 1 -- Ex vivo (HSC/thymic cells)',
        2: 'Phase 2 -- Systemic AAV (liver/ubiquitous)',
        3: 'Phase 3 -- CNS/Neuronal (intrathecal)',
        4: 'Phase 4 -- Cardiac (intracoronary)',
        5: 'Phase 5 -- Mitochondrial',
        6: 'Phase 6 -- Gene duplications (large)',
        7: 'Phase 7 -- Inducible systems (ongoing)',
    }

    phases = {i: [] for i in range(1, 8)}
    for name in DELIVERY_PLAN:
        plan = get_delivery_plan(name)
        c = constructs.get(name, {})
        phases[plan['phase']].append({
            'mod': name,
            'vector': plan['vector'],
            'route': plan['route'],
            'insert_bp': c.get('insert_bp', '?'),
            'aav_ok': c.get('aav_status', ''),
            'notes': plan['notes'],
        })

    schedule = []
    for ph_num in sorted(phases.keys()):
        mods = phases[ph_num]
        if not mods:
            continue
        entry = {'phase': ph_num, 'name': phase_names.get(ph_num, f'Phase {ph_num}'), 'mods': mods}
        schedule.append(entry)

        if verbose:
            print(f'\n  -- {entry["name"]} --')
            for m in mods:
                size_str = f'{m["insert_bp"]}bp' if isinstance(m["insert_bp"], int) else '?'
                print(f'    [{m["vector"]:12}] {m["mod"]:30} {size_str:>8}  {m["route"]}')

    return schedule

def export_schedule(schedule: list, output_dir: str = OUT_DIR) -> str:
    path = os.path.join(output_dir, 'delivery_schedule.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)
    return path

if __name__ == '__main__':
    from pipeline.sequence_fetcher import ACCESSIONS, get_sequence
    from pipeline.construct_builder import build_all

    print('=== Delivery Planner Test ===')
    results    = {n: get_sequence(n, verbose=False) for n in ACCESSIONS}
    constructs = build_all(results, verbose=False)

    schedule = full_delivery_schedule(constructs, verbose=True)
    path = export_schedule(schedule)
    print(f'\nSchedule saved: {path}')

    # Summary
    total = sum(len(p['mods']) for p in schedule)
    print(f'\nTotal mods scheduled: {total}/{len(DELIVERY_PLAN)}')
    for p in schedule:
        vectors = set(m['vector'] for m in p['mods'])
        print(f'  {p["name"]}: {len(p["mods"])} mods  vectors: {", ".join(sorted(vectors))}')
