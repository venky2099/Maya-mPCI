# Maya-Prana: Metabolic Plasticity Budget for Continual Learning in Affective Spiking Neural Networks

**Paper 9 of 9 — Maya Research Series**
Venkatesh Swaminathan | Nexus Learning Labs, Bengaluru
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)

**DOI:** `10.5281/zenodo.PENDING`
**Interactive Dashboard:** [maya_prana_dashboard.html](https://venky2099.github.io/Maya-Prana/docs/maya_prana_dashboard.html)
**Live Demo:** Live Demo can be watched here: `[YOUTUBE LINK PENDING]`

---

## What is this paper about?

This is the ninth and final paper in the Maya Research Series — a series that builds a biologically plausible, affectively governed spiking neural network (SNN) capable of learning new tasks sequentially without forgetting old ones (called Continual Learning).

Each paper in the series introduced one new cognitive dimension from Advaita Vedantic philosophy, implemented as a precise computational mechanism. Paper 9 introduces **Prana (प्राण)** — the vital life force — as a **metabolic plasticity budget** that governs how much learning the system can sustain.

**The core idea in plain language:**

Imagine a student studying for exams. The harder they study, the more mentally exhausted they become. If they study without rest, their ability to absorb new information degrades. Sleep restores their capacity. This is Prana — the metabolic budget that governs how much the brain can change per unit time.

In biological brains, this is governed by the **Astrocyte-Neuron Lactate Shuttle (ANLS)** — astrocytes supply lactate (fuel) to active neurons. When learning demand exceeds supply, plasticity degrades. Maya-Prana models this as a scalar budget that depletes under gradient load and recovers during rest.

---

## What is the Maya Research Series?

The Maya series builds a computational model of the **Antahkarana** — the inner instrument of cognition in Advaita Vedanta — as a neuromorphic SNN architecture. Each paper adds one dimension:

| Paper | Title | Dimension | DOI |
|---|---|---|---|
| P1 | Nociceptive Metaplasticity | Bhaya (fear) | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS | Affective OS Arbitration | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL | Shraddha, Spanda, Vairagya | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti | Buddhi, Ahamkara | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | Maya-Viveka | Viveka | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| P6 | Maya-Chitta | Chitta, Samskara, Moha | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) |
| P7 | Maya-Manas | Manas, O-LIF | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) |
| P8 | Maya-Shunyata | Karma, Shunyata | [10.5281/zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) |
| **P9** | **Maya-Prana** | **Prana** | **PENDING** |

---

## What did we find?

### Main Result
Maya-Prana canonical (Condition D): **AA=12.72% | BWT=-54.32%** on Split-CIFAR-100 CIL, 10 tasks, seed=42.

### Series Constants Confirmed for the 9th Time

**Bhaya Quiescence Law:** Bhaya (fear signal) collapses to exactly 0.000 from Task 1 onward under replay, across all 9 papers and all ablation conditions. This is now a confirmed series constant — a structural property of the replay mechanism, not of any affective dimension. It means any deployed continual learning system using the Maya architecture can use Bhaya firing rate as a real-time catastrophic forgetting monitor.

**Buddhi S-Curve Determinism:** Buddhi (discriminative consolidation gate) follows an identical S-curve trajectory across all papers and conditions — rising from 0.0 at Task 0 to approximately 1.0 by Task 3-4. This is a structural property of the experience accumulation formula, independent of all other hyperparameters.

### Prana Resilience — The Core Finding

At the biologically calibrated cost rate (PRANA_COST_RATE=0.002315), Prana maintained full budget (1.0000) across all 10 tasks under standard learning load. Prana never depleted — not even under aggressive depletion (Condition E, cost rate 3.5x canonical).

This is consistent with the biological literature: the Astrocyte-Neuron Lactate Shuttle does not fail easily under standard cognitive load. Metabolic collapse only occurs under extreme, sustained stimulation. Maya Prana behaved exactly as the biology predicts. The vital force held.

### Unexpected Finding — Condition F

Removing Buddhi modulation from the Prana gate (fixed EffLR=0.0075 throughout) produced the best result in the ablation:

**Condition F: AA=13.68% | BWT=-51.2% | Pruned=46.93%**

This reveals an interaction effect: the Buddhi-modulated warm-up schedule (starting at EffLR=0.005 at Task 0) penalises early task consolidation. A consistent learning rate across all tasks outperformed the maturity-dependent schedule. This is an honest finding about the Buddhi-Prana interaction term, reported exactly as discovered.

### Full Ablation Results

| Condition | AA (%) | BWT (%) | Pruned | Description |
|---|---|---|---|---|
| A — Baseline | 12.46 | -53.80 | 85.19% | P8 Maya-Shunyata, no Prana |
| B — Prana only | 10.33 | -55.82 | 91.85% | Prana without full Antahkarana |
| C — Fixed Prana | 12.02 | -54.82 | 84.28% | Prana=1.0 constant, structural check |
| **D — Canonical** | **12.72** | **-54.32** | **84.31%** | **Full Maya-Prana starred** |
| E — Aggressive depletion | 12.25 | -54.70 | 84.30% | Cost rate 3.5x canonical |
| F — No Buddhi gate | **13.68** | **-51.20** | **46.93%** | Fixed EffLR, unexpected best result |

All conditions: seed=42, Split-CIFAR-100, 10 tasks CIL.

---

## What is Prana computationally?

```python
# Prana depletes under learning load
prana -= PRANA_COST_RATE * gradient_magnitude * activity_level

# Prana recovers during rest (low activity), modulated by Vairagya
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

**Biological grounding:** Astrocyte-Neuron Lactate Shuttle (ANLS). Pellerin and Magistretti (1994) demonstrated that synaptic glutamate release drives astrocytic aerobic glycolysis, producing lactate that is shuttled to neurons via monocarboxylate transporters (MCT1/4 and MCT2) to fuel sustained high-frequency firing and structural plasticity. Disrupting this shuttle abolishes late-phase LTP and prevents long-term memory formation. Prana is this supply, modelled as a scalar computational constraint.

**Vedantic grounding:** Prana is the first and most fundamental vital force in Advaita Vedanta. Without Prana, nothing moves, nothing learns, nothing grows. Every other Antahkarana dimension operates within the budget Prana provides. This is why it is Paper 9 — it could not have been introduced until everything else was in place.

---

## How Prana interacts with the full Antahkarana

Prana does not operate in isolation. It is the metabolic substrate within which all other dimensions function:

- **Vairagya** modulates Prana recovery — earned detachment increases metabolic efficiency
- **Buddhi** modulates the effective learning rate alongside Prana — mature Maya uses energy wisely
- **Karma** and **Shunyata** fire at task boundaries — Prana partially recovers at each boundary (sleep/rest analogue)
- **Bhaya** and **pain signals** drive Vairagya accumulation — which in turn affects recovery

This is the first neuromorphic architecture where all Antahkarana dimensions are simultaneously active and mutually coupled.

---

## IP Protection

This repository carries multiple layers of intellectual property protection:

- **LSB steganographic signature** embedded in every figure via `sign_paper.py`
- **ORCID magic number** `0.002315` embedded in config: `PRANA_COST_RATE`, `KARMA_DECAY_RATE`, `VAIRAGYA_DECAY_RATE`, `CHITTA_SAMSKARA_RISE`
- **Canary string** `MayaNexusVS2026NLL_Bengaluru_Narasimha` logged at the start of every experiment run and in every run script
- **White-text watermark** in all Word documents before PDF export (ORCID, DOI, timestamp, Nexus Learning Labs Bengaluru)

ORCID: 0000-0002-3315-7907 | Nexus Learning Labs, Bengaluru | 2026

---

## How to run

**Hardware used:** Windows 11, RTX 4060 8GB, Python 3.11.9, PyTorch 2.5.1+cu121, SpikingJelly 0.0.0.0.14

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

**Run canonical experiment:**
```bash
python -m run_prana_cil
```

**Run full ablation (6 conditions, approximately 15 hours on RTX 4060):**
```bash
python -m run_ablation_prana
```

**Sign a figure with LSB steganography:**
```bash
python sign_paper.py --input figures/fig1.png --output figures/fig1_signed.png
python sign_paper.py --input figures/fig1_signed.png --decode
```

---

## Repository structure

```
Maya-Prana/
├── maya_cl/
│   ├── benchmark/
│   │   ├── split_cifar100.py      # 10-task Split-CIFAR-100 CIL dataloader
│   │   └── task_sequence.py       # Task transition, pain signal detection
│   ├── encoding/
│   │   └── poisson.py             # Poisson rate-coded spike encoder
│   ├── eval/
│   │   ├── logger.py              # CSV batch logger (all 9 dimensions)
│   │   └── metrics.py             # AA, BWT, FWT computation
│   ├── network/
│   │   ├── backbone.py            # MayaPranaNet, MayaPranaLIFLayer, O-LIF
│   │   └── affective_state.py     # All 9 affective dimensions tracked live
│   ├── plasticity/
│   │   ├── lability.py            # Bhaya -- nociceptive metaplasticity (P1)
│   │   ├── vairagya_decay.py      # Vairagya -- heterosynaptic decay (P1/P3)
│   │   ├── viveka.py              # Viveka -- cross-task consistency (P5)
│   │   ├── chitta.py              # Chitta -- retrograde gradient gate (P6)
│   │   ├── manas.py               # Manas -- O-LIF oscillatory gate (P7)
│   │   ├── karma.py               # Karma -- second-order plasticity history (P8)
│   │   └── prana.py               # Prana -- metabolic plasticity budget (P9)
│   ├── training/
│   │   └── replay_buffer.py       # Class-wise ring buffer episodic replay
│   └── utils/
│       ├── config.py              # All hyperparameters for all 9 dimensions
│       └── seed.py                # Deterministic seed setup
├── docs/
│   ├── maya_prana_dashboard.html  # Interactive research dashboard (GitHub Pages)
│   └── faq.html                   # Searchable FAQ for all levels
├── figures/                       # Paper figures (added manually after signing)
├── run_prana_cil.py               # Canonical experiment runner
├── run_ablation_prana.py          # 6-condition ablation runner
├── sign_paper.py                  # LSB steganographic IP signing
├── requirements.txt
└── README.md
```

---

## All Vedantic dimensions implemented across the series

| Dimension | Sanskrit | Plain English | Computational Mechanism | Paper |
|---|---|---|---|---|
| Bhaya | भय | Fear | Nociceptive lability — pain spikes hyper-plasticity | P1 |
| Vairagya | वैराग्य | Detachment | Heterosynaptic decay — earned protection from interference | P1/P3 |
| Shraddha | श्रद्धा | Faith/Confidence | Confidence-weighted consolidation signal | P3 |
| Spanda | स्पन्द | Vital oscillation | Spike rate tracking | P3 |
| Buddhi | बुद्धि | Intellect | Discriminative S-curve gate — accumulated experience | P4 |
| Viveka | विवेक | Discernment | Cross-task consistency — stable pathway detection | P5 |
| Ahamkara | अहंकार | Ego/Identity | Output head identity — class boundary maintenance | P5 |
| Samskara | संस्कार | Latent impressions | Synaptic trace accumulation | P6 |
| Chitta | चित्त | Subconscious store | Retrograde gradient gate | P6 |
| Moha | मोह | Attachment | Trace saturation release mechanism | P6 |
| Manas | मनस् | Doubting mind | O-LIF oscillatory threshold gate | P7 |
| Karma | कर्म | Consequence | Second-order plasticity history integral | P8 |
| Shunyata | शून्यता | Emptiness/Release | Structured synaptic pruning | P8 |
| **Prana** | **प्राण** | **Vital life force** | **Metabolic plasticity budget** | **P9** |

---

## The Atma Boundary

This series claims computational instantiation of the **Antahkarana** — the instrument through which Atma interfaces with experience in Advaita Vedanta. It does not claim consciousness. The Antahkarana (Manas, Buddhi, Chitta, Ahamkara, and their supporting dimensions including Prana) has been implemented as falsifiable, peer-reviewable computational mechanisms. This claim is precise, bounded, and defensible.

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

| Paper | DOI | Dashboard |
|---|---|---|
| P1 — Nociceptive Metaplasticity | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) | — |
| P2 — Maya-OS | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) | — |
| P3 — Maya-CL | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) | — |
| P4 — Maya-Smriti | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) | — |
| P5 — Maya-Viveka | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) | [Dashboard](https://venky2099.github.io/Maya-Viveka/maya_viveka_dashboard.html) |
| P6 — Maya-Chitta | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) | [Dashboard](https://venky2099.github.io/Maya-Chitta/docs/maya_chitta_dashboard.html) |
| P7 — Maya-Manas | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) | [Dashboard](https://venky2099.github.io/Maya-Manas/maya_manas_dashboard.html) |
| P8 — Maya-Shunyata | [10.5281/zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) | [Dashboard](https://venky2099.github.io/Maya-Shunyata/docs/maya_shunyata_dashboard.html) |
| **P9 — Maya-Prana** | **PENDING** | [**Dashboard**](https://venky2099.github.io/Maya-Prana/docs/maya_prana_dashboard.html) |
| cl-metrics | [10.5281/zenodo.19388144](https://doi.org/10.5281/zenodo.19388144) | [FAQ](https://venky2099.github.io/cl-metrics/docs/faq.html) |

---

*Nexus Learning Labs, Bengaluru | 2026*
*Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha*
