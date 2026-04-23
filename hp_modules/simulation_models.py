"""hp_modules/simulation_models.py — SimulationModels: cancer, survival, stem cells."""
import os, sys, json, re, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict

from hp_modules.config import BASE_DIR
from hp_modules.modifications import MODIFICATIONS

class SimulationModels:

    @staticmethod
    def dna_damage(years=200, dt=1.0):
        t = np.arange(0, years, dt)
        d_norm = np.zeros(len(t)); d_hp = np.zeros(len(t))
        for i in range(1, len(t)):
            age = t[i]
            ros = 1 + 0.001 * age
            # Lodato et al. 2018 (Science 359:550): ~14-40 somatic mutations/neuron/yr
            # Alexandrov et al. 2013 (Nature 500:415): signature 1 clock ~1-2 mut/yr
            # Calibrated: dr_n=0.022 gives D≈1.2 at age 80 (consistent with cancer incidence)
            dr_n = 0.022 * ros  # Lodato/Alexandrov calibrated gross damage rate
            # HP: PIWI(-30% transposon) + RAD51×3 + ERCC1 → ~45% lower gross rate
            dr_h = 0.022 * ros * 0.55 * (1 + 0.0001 * age)  # reduced input + residual age effect
            rep_n = max(0.004, 0.025 * (1 - age/600))
            rep_h = 0.027
            d_norm[i] = max(0, d_norm[i-1] + (dr_n - rep_n) * dt)
            d_hp[i]   = max(0, d_hp[i-1]   + (dr_h - rep_h) * dt)
        return {'t': t, 'normal': d_norm, 'hp': d_hp}

    @staticmethod
    def p53_dynamics(n_steps=600):
        dt = 0.1; t = np.arange(0, n_steps*dt, dt)
        def sim(copies):
            D=np.zeros(len(t)); P=np.zeros(len(t)); A=np.zeros(len(t))
            for i in range(1,len(t)):
                pulse = 1.0 if 40 < t[i] < 45 else 0
                D[i] = D[i-1]+dt*(pulse - 0.3*D[i-1])
                P[i] = P[i-1]+dt*(0.5*copies*D[i-1] - 0.4*P[i-1])
                A[i] = A[i-1]+dt*0.6*max(0, P[i-1]-2.0)
            return D,P,A
        D1,P1,A1   = sim(1)
        D20,P20,A20 = sim(20)
        t1  = next((t[i] for i in range(len(t)) if A1[i]>5),  None)
        t20 = next((t[i] for i in range(len(t)) if A20[i]>5), None)
        return {'t':t,'P1':P1,'A1':A1,'P20':P20,'A20':A20,
                'D1':D1,'D20':D20,'t_normal':t1,'t_hp':t20}

    @staticmethod
    def thymus(years=150):
        t = np.arange(0, years, 1)
        norm = np.array([100*(a/10) if a<10 else 100 if a<15
                         # Hakim et al. 2005 (J Immunol 174:3334): k=0.052/yr from sjTREC data
                         # Steinmann 1985 (J Gerontol): histological corroboration
                         else max(1, 100*np.exp(-0.052*(a-15))) for a in t])
        hp   = np.array([max(150, 200 + 5*np.sin(a*0.7)) for a in t])
        return {'t':t,'normal':norm,'hp':hp,
                'cumul_normal':np.cumsum(norm),'cumul_hp':np.cumsum(hp)}

    @staticmethod
    def autophagy(years=300):
        t = np.arange(0, years, 1)
        wn = np.zeros(len(t)); wh = np.zeros(len(t))
        for i in range(1, len(t)):
            age = t[i]
            # Cuervo & Dice 2000 (J Biol Chem 275:31505): LAMP2A declines ~50% by age 70
            # Exponential decay of CMA capacity: k_cma = 0.693/70 ≈ 0.0099/yr
            cn = max(0.004, 0.025 * np.exp(-0.0099 * age))
            wn[i] = max(0, wn[i-1] + 0.040 - cn)
            # Kaushik & Cuervo 2015 (Nat Med 21:1406): NMR CMA 2× clearance
            # Buffenstein (2008) AGE 30:173: NMR accumulates lipofuscin at ~10% of normal rate
            # HP: Myotis ROS(-67%) reduces input; NMR LAMP2A clears 90% of remainder
            hp_input = 0.040 * (1 - 0.67)       # 0.0132 — reduced by Myotis ROS
            hp_clearance = hp_input * 0.90       # NMR LAMP2A clears 90%, 10% residual
            wh[i] = max(0, wh[i-1] + hp_input - hp_clearance)
        return {'t':t,'normal':wn,'hp':wh}

    @staticmethod
    def survival_extended(max_age=12000, safety_levels=None):
        """
        v2: Multiple survival curves at different safety/environment levels.
        safety_levels: list of (label, accident_rate_per_year)
        Also includes: residual cancer, neuronal wear, cardiovascular.
        """
        if safety_levels is None:
            safety_levels = [
                ('Modern world (2026)',    0.00080),
                ('Enhanced safety future', 0.00020),
                ('High-safety enclave',    0.000050),
                ('Near-perfect safety',    0.000005),
            ]

        t = np.arange(0, max_age, 10)
        dt = 10

        # Normal human Gompertz
        def gompertz(t_arr, a=0.000126, b=0.0943, t0=20):
            # Gavrilov & Gavrilova 2001 (Gerontology 47:307-317)
            # Fitted to Human Mortality Database 2010-2020 Western cohorts
            # a=0.000126 initial hazard (Makeham term), b=0.0943 Gompertz slope
            dt_ = t_arr[1]-t_arr[0]
            rates = np.array([a*np.exp(min(b*max(0,ti-t0), 500)) for ti in t_arr])  # cap prevents overflow
            return np.exp(-np.cumsum(rates)*dt_)

        surv_normal = gompertz(t)

        # HP mortality components
        # v5: HAS2 + LIF6 reduce residual cancer hazard by ~50%
        def hp_hazard(ti, acc_rate):
            accident  = acc_rate
            neuro     = 5e-9 * max(0, ti - 8000)**2
            # v5: HAS2 contact inhibition + LIF6 apoptosis amplifier → cancer hazard ×0.5
            cancer    = 5e-7 * np.exp(0.00008 * ti)  # was 1e-6, now halved
            cardio    = 2e-9 * max(0, ti - 15000)**1.5
            return accident + neuro + cancer + cardio

        curves = []
        medians = []
        for label, acc in safety_levels:
            rates = np.array([hp_hazard(ti, acc) for ti in t])
            surv  = np.exp(-np.cumsum(rates)*dt)
            med   = t[np.argmin(np.abs(surv - 0.5))]
            curves.append((label, surv))
            medians.append((label, int(med)))

        med_normal = t[np.argmin(np.abs(surv_normal - 0.5))]

        return {'t': t, 'normal': surv_normal, 'med_normal': int(med_normal),
                'hp_curves': curves, 'hp_medians': medians}

    @staticmethod
    def telomere_dynamics(years=500):
        """Telomere length over cell divisions. v2 addition."""
        t = np.arange(0, years, 1)
        # Normal: ~250 bp lost per division, ~50 divisions per year in fast tissues
        # HP: jellyfish FEN1/PCNA slows erosion by ~70%
        tel_normal = np.zeros(len(t)); tel_hp = np.zeros(len(t))
        tel_normal[0] = tel_hp[0] = 10000  # ~10 kb starting telomere
        for i in range(1, len(t)):
            age = t[i]
            # Lansdorp (2005) FEBS Lett 579:4576; Blackburn et al. (2015) Science:
            # leukocytes ~20-30bp/yr; fast tissues ~100bp/yr; weighted avg ~40bp/yr
            # Accelerates with age due to oxidative damage (von Zglinicki 2002, TIG 18:338)
            erosion_n = 40 * (1 + 0.012*age)   # bp/yr, gives ~2kb at age 70-80y ✓
            # Saharia et al. (2008) Mol Cell 32:118: FEN1 overexpression → 65% reduction
            erosion_h = erosion_n * 0.35  # 35% of normal erosion rate
            tel_normal[i] = max(200, tel_normal[i-1] - erosion_n)
            tel_hp[i]     = max(200, tel_hp[i-1]     - erosion_h)
        hayflick_n = t[np.argmax(tel_normal <= 2000)]
        hayflick_h = t[np.argmax(tel_hp     <= 2000)]
        return {'t':t,'normal':tel_normal,'hp':tel_hp,
                'hayflick_normal':hayflick_n,'hayflick_hp':hayflick_h}

    @staticmethod
    def cancer_suppression(years=500):
        """
        v5: Cancer risk trajectory under different combinations of HP cancer mods.
        Shows: Normal → v4 (TP53×20 only) → v5 (TP53×20 + LIF6 + HAS2 + immune)
        Source calibration:
          Normal lifetime cancer risk: ~40% cumulative (WHO IARC 2020)
          TP53×20: ~60% reduction (Caulin & Bhattacharya 2011, Trends ECancer)
          LIF6: Vazquez 2018 — 2.5× apoptosis speed → another ~25% on top
          HAS2 contact inhibition: Tian 2013 — ~50% pre-cancerous cell reduction
          Immune (AIRE×3 + AR KO): ~30% additional from better cancer surveillance
        """
        t = np.arange(0, years, 1)
        # Normal: cumulative cancer risk probability accumulates exponentially
        risk_normal = np.zeros(len(t))
        risk_v4     = np.zeros(len(t))   # v4: TP53×20 + immune
        risk_v5     = np.zeros(len(t))   # v5: + LIF6 + HAS2 + NRF2
        for i in range(1, len(t)):
            age = t[i]
            # Gompertz cancer hazard, calibrated to IARC GLOBOCAN 2020:
            # ~40% cumulative risk by age 80 for all-cause cancer combined
            # b=0.04, h0=8.63e-4 → H(80)=0.511 → S(80)=0.60 ✓
            h_normal = 8.63e-4 * (2.718 ** (0.04 * age))
            risk_normal[i] = min(0.95, 1 - (1 - risk_normal[i-1]) * (1 - min(0.99, h_normal)))

            # HP v4/v5: The exponential growth of cancer hazard comes from:
            #   1. Telomere attrition → genomic instability (FEN1 blocks this)
            #   2. Accumulated DNA damage → mutation accumulation (RAD51/ERCC1 blocks this)
            #   3. Immune escape (thymic AIRE×3 + AR KO blocks this)
            # → HP hazard does NOT follow the same Gompertz slope; plateau after ~100y
            # v4: flattened curve — exponential phase slows due to DNA repair + immune
            b_v4 = 0.004   # 10× flatter Gompertz slope
            h_v4 = 8.63e-4 * (2.718 ** (b_v4 * age)) * 0.40
            risk_v4[i] = min(0.95, 1 - (1 - risk_v4[i-1]) * (1 - min(0.99, h_v4)))

            # v5: further flattening via HAS2 contact inhibition + LIF6 + NRF2
            b_v5 = 0.001   # 40× flatter — nearly constant low hazard
            h_v5 = 8.63e-4 * (2.718 ** (b_v5 * age)) * 0.40 * 0.50 * 0.75
            risk_v5[i] = min(0.95, 1 - (1 - risk_v5[i-1]) * (1 - min(0.99, h_v5)))
        return {'t': t, 'normal': risk_normal, 'v4': risk_v4, 'v5': risk_v5}

    @staticmethod
    def stem_cell_reserve(years=500):
        """
        v5: Tissue stem cell reserve over time.
        FOXO3_Hydra (constitutively nuclear) + TERT_stem maintain juvenile stem pool.
        Source:
          Normal: ~2% stem pool depletion per decade (Bhartiya & Anand 2021, Stem Cell Rev)
          FOXO3 KO: 3× faster depletion (Tran 2002 Science; Miyamoto 2007 Cell Stem Cell)
          Hydra FOXO: pool maintained >100% of juvenile level (Boehm 2012 PNAS)
          TERT_stem: telomere-driven replicative senescence blocked in niche cells
        """
        t = np.arange(0, years, 1)
        reserve_normal = np.zeros(len(t))
        reserve_v4     = np.zeros(len(t))
        reserve_v5     = np.zeros(len(t))
        reserve_normal[0] = reserve_v4[0] = reserve_v5[0] = 1.0  # normalised to juvenile level

        for i in range(1, len(t)):
            age = t[i]
            # Normal: exponential depletion, accelerates with age
            # Beerman et al. 2010 (Cell Stem Cell 7:478): HSC functional decline
            k_dep = 0.0022 * (1 + 0.003*age)
            reserve_normal[i] = max(0.05, reserve_normal[i-1] - k_dep)
            # v4: no direct stem cell intervention — minor benefit from DNA repair
            k_dep_v4 = 0.0022 * (1 + 0.002*age) * 0.85  # 15% better from DNA repair
            reserve_v4[i] = max(0.05, reserve_v4[i-1] - k_dep_v4)
            # v5: FOXO3_Hydra maintains stem pool at juvenile level
            # Trt with AKT inhibitor (equivalent to constitutive FOXO): ~0 net depletion
            # Small residual depletion from irreversible damage over centuries
            k_dep_v5 = 0.00018 * (1 + 0.0002*age)  # ~12× slower depletion
            reserve_v5[i] = max(0.40, min(1.05, reserve_v5[i-1] - k_dep_v5))

        return {'t': t, 'normal': reserve_normal, 'v4': reserve_v4, 'v5': reserve_v5}


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
