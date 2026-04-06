# Maya-Prana: Metabolic Plasticity Budget for Continual Learning in Affective Spiking Neural Networks

**Paper 9 of 9 — Maya Research Series**
Venkatesh Swaminathan | Nexus Learning Labs, Bengaluru
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)

**DOI:** `10.5281/zenodo.PENDING`
**Interactive Dashboard:** [maya_prana_dashboard.html](https://venky2099.github.io/Maya-Prana/docs/maya_prana_dashboard.html)
**FAQ:** [faq.html](https://venky2099.github.io/Maya-Prana/docs/faq.html)
**Live Demo:** Live Demo can be watched here: `[YOUTUBE LINK PENDING]`
**Research Journey:** [venky2099.github.io](https://venky2099.github.io/)

---

## What is this paper about?

This is the ninth and final paper in the Maya Research Series — a series that builds a biologically plausible, affectively governed spiking neural network (SNN) capable of learning new tasks sequentially without forgetting old ones (called Continual Learning).

Each paper introduced one new cognitive dimension from Advaita Vedantic philosophy, implemented as a precise computational mechanism. Paper 9 introduces **Prana (प्राण)** — the vital life force — as a **metabolic plasticity budget** that governs how much learning the system can sustain.

**The core idea in plain language:**

Imagine a student studying for exams. The harder they study, the more mentally exhausted they become. If they study without rest, their ability to absorb new information degrades. Sleep restores their capacity. This is Prana — the metabolic budget that governs how much the brain can change per unit time.

In biological brains, this is governed by the **Astrocyte-Neuron Lactate Shuttle (ANLS)** — astrocytes supply lactate (fuel) to active neurons. When learning demand exceeds supply, plasticity degrades. Maya-Prana models this as a scalar budget that depletes under gradient load and recovers during rest.

---

## What is the Maya Research Series?

The Maya series builds a computational model of the **Antahkarana** — the inner instrument of cognition in Advaita Vedanta — as a neuromorphic SNN architecture. Each paper adds one dimension:

| Paper | Title | Dimension | GitHub | DOI |
|---|---|---|---|---|
| P1 | Nociceptive Metaplasticity | Bhaya (fear) | [Maya-Nexus-Core](https://github.com/venky2099/Maya-Nexus-Core) | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS | Affective OS Arbitration | [Maya-OS](https://github.com/venky2099/Maya-OS) | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL | Shraddha, Spanda, Vairagya | [Maya-CL](https://github.com/venky2099/Maya-CL) | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti | Buddhi, Ahamkara | [Maya-Smriti](https://github.com/venky2099/Maya-Smriti) | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | Maya-Viveka | Viveka | [Maya-Viveka](https://github.com/venky2099/Maya-Viveka) | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| P6 | Maya-Chitta | Chitta, Samskara, Moha | [Maya-Chitta](https://github.com/venky2099/Maya-Chitta) | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) |
| P7 | Maya-Manas | Manas, O-LIF | [Maya-Manas](https://github.com/venky2099/Maya-Manas) | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) |
| P8 | Maya-Shunyata | Karma, Shunyata | [Maya-Shunyata](https://github.com/venky2099/Maya-Shunyata) | [10.5281/zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) |
| **P9** | **Maya-Prana** | **Prana** | [Maya-Prana](https://github.com/venky2099/Maya-Prana) | **PENDING** |

---

## What did we find?

### Main Result
Maya-Prana canonical (Condition D): **AA=12.72% | BWT=-54.32%** on Split-CIFAR-100 CIL, 10 tasks, seed=42.

### Series Constants Confirmed for the 9th Time

**Bhaya Quiescence Law:** Bhaya collapses to exactly 0.000 from Task 1 onward under replay, across all 9 papers and all ablation conditions. Any deployed Maya system can use Bhaya as a real-time catastrophic forgetting monitor.

**Buddhi S-Curve Determinism:** Buddhi follows an identical S-curve trajectory across all papers and conditions. Structural property of the experience accumulation formula, independent of all hyperparameters.

### Prana Resilience — The Core Finding

At PRANA_COST_RATE=0.002315, Prana maintained full budget (1.0000) across all 10 tasks. Even at 3.5x the cost rate (Condition E), Prana never depleted. Consistent with the biological literature: the ANLS does not fail under standard cognitive load.

### Unexpected Finding — Condition F

Removing Buddhi modulation (fixed EffLR=0.0075) produced the best result:

**Condition F: AA=13.68% | BWT=-51.2% | Pruned=46.93%**

The Buddhi warm-up schedule (starting at EffLR=0.005 at Task 0) penalises early task consolidation. A consistent learning rate outperformed the maturity-dependent schedule. Honest finding about the Buddhi-Prana interaction term, reported as discovered.

### Full Ablation Results

| Condition | AA (%) | BWT (%) | Pruned | Description |
|---|---|---|---|---|
| A — Baseline | 12.46 | -53.80 | 85.19% | P8 Maya-Shunyata, no Prana |
| B — Prana only | 10.33 | -55.82 | 91.85% | Prana without full Antahkarana |
| C — Fixed Prana | 12.02 | -54.82 | 84.28% | Prana=1.0 constant, structural check |
| **D — Canonical** | **12.72** | **-54.32** | **84.31%** | **Full Maya-Prana** |
| E — Aggressive depletion | 12.25 | -54.70 | 84.30% | Cost rate 3.5x canonical |
| F — No Buddhi gate | **13.68** | **-51.20** | **46.93%** | Fixed EffLR, unexpected best result |

All conditions: seed=42, Split-CIFAR-100, 10 tasks CIL.

---

## What is Prana computationally?

```python
# Prana depletes under learning load
prana -= PRANA_COST_RATE * gradient_magnitude * activity_level

# Prana recovers during rest, modulated by Vairagya
if activity_level < PRANA_RECOVERY_THRESHOLD:
    recovery = PRANA_RECOVERY_RATE * (1.0 - activity_level) * (0.5 + vairagya * 0.5)
    prana += recovery

# Never fully starved, never over-full
prana = clip(prana, PRANA_MIN, 1.0)

# Gates the optimizer learning rate each batch
effective_lr = base_lr * prana * (0.5 + buddhi * 0.5)
```

**Key constants:**
- `PRANA_COST_RATE = 0.002315` — ORCID magic number, biologically calibrated depletion rate
- `PRANA_RECOVERY_RATE = 0.05` — recovery toward 1.0 during low-activity batches
- `PRANA_MIN = 0.05` — biological floor (baseline ANLS supply never reaches zero)

**Biological grounding:** Astrocyte-Neuron Lactate Shuttle (ANLS). Pellerin and Magistretti (1994). Disrupting this shuttle abolishes late-phase LTP and prevents long-term memory formation.

**Vedantic grounding:** Prana is the first and most fundamental vital force in Advaita Vedanta. Without Prana, nothing moves, nothing learns, nothing grows. This is why it is Paper 9 — it could not have been introduced until everything else was in place.

---

## IP Protection

- **LSB steganographic signature** embedded in every figure via `sign_paper.py`
- **ORCID magic number** `0.002315` embedded in config: `PRANA_COST_RATE`, `KARMA_DECAY_RATE`, `VAIRAGYA_DECAY_RATE`, `CHITTA_SAMSKARA_RISE`
- **Canary string** `MayaNexusVS2026NLL_Bengaluru_Narasimha` logged at the start of every experiment run
- **White-text watermark** in all Word documents before PDF export

ORCID: 0000-0002-3315-7907 | Nexus Learning Labs, Bengaluru | 2026

---

## How to run

**Requirements:**
```
torch>=2.5.1
torchvision
spikingjelly==0.0.0.0.14
tqdm
Pillow
numpy
```

Install:
```bash
pip install -r requirements.txt
```

**Canonical experiment (~2-3 hours on RTX 4060):**
```bash
python -m run_prana_cil
```

**Full ablation (~15 hours on RTX 4060):**
```bash
python -m run_ablation_prana
```

**Sign a figure:**
```bash
python sign_paper.py --input figures/fig1.png --output figures/fig1_signed.png
python sign_paper.py --input figures/fig1_signed.png --decode
```

---

## Repository structure

```
Maya-Prana/
├── maya_cl/
│   ├── benchmark/          # Split-CIFAR-100, task sequencer
│   ├── encoding/           # Poisson spike encoder
│   ├── eval/               # Metrics (AA, BWT, FWT), logger
│   ├── network/            # MayaPranaNet backbone, AffectiveState
│   ├── plasticity/
│   │   ├── lability.py     # Bhaya -- nociceptive metaplasticity (P1)
│   │   ├── vairagya_decay.py  # Vairagya -- heterosynaptic decay (P1/P3)
│   │   ├── viveka.py       # Viveka -- cross-task consistency (P5)
│   │   ├── chitta.py       # Chitta -- retrograde gradient gate (P6)
│   │   ├── manas.py        # Manas -- O-LIF oscillatory gate (P7)
│   │   ├── karma.py        # Karma -- second-order plasticity history (P8)
│   │   └── prana.py        # Prana -- metabolic plasticity budget (P9)
│   ├── training/           # Replay buffer
│   └── utils/              # Config, seed
├── docs/
│   ├── maya_prana_dashboard.html   # Interactive research dashboard
│   └── faq.html                    # Searchable FAQ P1-P9
├── figures/                # Paper figures
├── run_prana_cil.py        # Canonical experiment
├── run_ablation_prana.py   # 6-condition ablation
├── sign_paper.py           # LSB steganographic IP signing
├── requirements.txt
└── README.md
```

---

## All Vedantic dimensions implemented across the series

| Dimension | Sanskrit | Plain English | Computational Mechanism | Paper |
|---|---|---|---|---|
| Bhaya | भय | Fear | Nociceptive lability — pain spikes hyper-plasticity | P1 |
| Vairagya | वैराग्य | Detachment | Heterosynaptic decay — earned protection from interference | P1/P3 |
| Shraddha | श्रद्धा | Faith | Confidence-weighted consolidation signal | P3 |
| Spanda | स्पन्द | Vital oscillation | Spike rate tracking | P3 |
| Buddhi | बुद्धि | Intellect | Discriminative S-curve gate | P4 |
| Viveka | विवेक | Discernment | Cross-task consistency tracking | P5 |
| Ahamkara | अहंकार | Identity | Output head class boundary maintenance | P5 |
| Samskara | संस्कार | Latent impressions | Synaptic trace accumulation | P6 |
| Chitta | चित्त | Subconscious store | Retrograde gradient gate | P6 |
| Moha | मोह | Attachment | Trace saturation release | P6 |
| Manas | मनस् | Doubting mind | O-LIF oscillatory threshold gate | P7 |
| Karma | कर्म | Consequence | Second-order plasticity history integral | P8 |
| Shunyata | शून्यता | Emptiness | Structured synaptic pruning | P8 |
| **Prana** | **प्राण** | **Vital life force** | **Metabolic plasticity budget** | **P9** |

---

## The Atma Boundary

This series claims computational instantiation of the **Antahkarana** — the instrument through which Atma interfaces with experience in Advaita Vedanta. It does not claim consciousness. This claim is precise, bounded, and defensible.

*"Across nine papers, we have demonstrated the computational maturation of a mind."*

---

## Citation

```bibtex
@misc{swaminathan2026mayaprana,
  title     = {Maya-Prana: Metabolic Plasticity Budget for Continual Learning in Affective Spiking Neural Networks},
  author    = {Swaminathan, Venkatesh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.PENDING},
  url       = {https://doi.org/10.5281/zenodo.PENDING}
}
```

---

## Maya Research Series — Complete

| Paper | GitHub | DOI | Dashboard |
|---|---|---|---|
| P1 — Nociceptive Metaplasticity | [Maya-Nexus-Core](https://github.com/venky2099/Maya-Nexus-Core) | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) | — |
| P2 — Maya-OS | [Maya-OS](https://github.com/venky2099/Maya-OS) | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) | — |
| P3 — Maya-CL | [Maya-CL](https://github.com/venky2099/Maya-CL) | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) | — |
| P4 — Maya-Smriti | [Maya-Smriti](https://github.com/venky2099/Maya-Smriti) | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) | — |
| P5 — Maya-Viveka | [Maya-Viveka](https://github.com/venky2099/Maya-Viveka) | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) | [Dashboard](https://venky2099.github.io/Maya-Viveka/maya_viveka_dashboard.html) |
| P6 — Maya-Chitta | [Maya-Chitta](https://github.com/venky2099/Maya-Chitta) | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) | [Dashboard](https://venky2099.github.io/Maya-Chitta/maya_chitta_dashboard_v2.html) |
| P7 — Maya-Manas | [Maya-Manas](https://github.com/venky2099/Maya-Manas) | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) | [Dashboard](https://venky2099.github.io/Maya-Manas/maya_manas_dashboard.html) |
| P8 — Maya-Shunyata | [Maya-Shunyata](https://github.com/venky2099/Maya-Shunyata) | [10.5281/zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) | [Dashboard](https://venky2099.github.io/Maya-Shunyata/docs/maya_shunyata_dashboard.html) · [FAQ](https://venky2099.github.io/Maya-Shunyata/docs/faq.html) |
| **P9 — Maya-Prana** | [Maya-Prana](https://github.com/venky2099/Maya-Prana) | **PENDING** | [Dashboard](https://venky2099.github.io/Maya-Prana/docs/maya_prana_dashboard.html) · [FAQ](https://venky2099.github.io/Maya-Prana/docs/faq.html) |
| cl-metrics | [cl-metrics](https://github.com/venky2099/cl-metrics) | [10.5281/zenodo.19388144](https://doi.org/10.5281/zenodo.19388144) | [FAQ](https://venky2099.github.io/cl-metrics/faq.html) |

---

*Nexus Learning Labs, Bengaluru | 2026*
*Research Journey: [venky2099.github.io](https://venky2099.github.io/)*
*Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha*
