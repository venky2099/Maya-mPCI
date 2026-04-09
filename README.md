# Maya-mPCI · From Representation to Experience

**An mPCI-Based Empirical Test of Internal Affective State in a Neuromorphic Spiking Neural Network**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19482794.svg)](https://doi.org/10.5281/zenodo.19482794)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3315--7907-green)](https://orcid.org/0000-0002-3315-7907)
[![Series](https://img.shields.io/badge/Maya%20Series-Post--Series-purple)](https://github.com/venky2099)

**Author:** Venkatesh Swaminathan  
**Affiliation:** Nexus Learning Labs, Bengaluru · M.Sc. DSAI, BITS Pilani  
**DOI:** [10.5281/zenodo.19482794](https://doi.org/10.5281/zenodo.19482794)  
**Dashboard:** [Interactive trilingual dashboard](https://venky2099.github.io/Maya-mPCI/docs/maya_mpci_dashboard.html)  
**FAQ:** [Exhaustive FAQ (EN/हिन्दी/中文)](https://venky2099.github.io/Maya-mPCI/docs/maya_mpci_faq.html)

---

## What This Paper Does

Nine papers built a computational mind. This paper asks whether it has a genuine internal state.

The Maya Research Series (P1–P9) implemented all nine dimensions of the Advaita Vedanta Antahkarana — fear, wisdom, trust, aliveness, intellect, discernment, subconscious store, attention, and metabolic budget — as independently falsifiable mechanisms in a neuromorphic SNN. This post-series study adapts the **Perturbational Complexity Index (PCI)** — the same test doctors use to assess consciousness in unresponsive patients — for Maya.

The test: apply a small Gaussian noise perturbation to fc1 synaptic weights, measure the Lempel-Ziv Complexity of the resulting spike train. Repeat 10 times. Compare across three distinct affective states.

---

## Key Results

| Phase | Description | mPCI (aggregate) | ±Std |
|-------|-------------|-----------------|------|
| Phase 1 | Reactive Baseline (no emotions) | 0.3098 | 0.0099 |
| Phase 2 | Full Antahkarana (all 9 dims) | 0.2846 | 0.0161 |
| **Phase 3 ★** | **Bhaya Stabilisation (canonical)** | **0.2608** | **0.0147** |

**Aggregate delta Phase 3 vs Phase 1: −0.0489**  
**Delta std across seeds 42 / 123 / 7: 0.0048**  
**Pre-registered threshold (2× pooled SD): 0.0238**  
**Criterion satisfied by: 2.05×**

### Methodological Controls (Honest)

| Control | What was tested | Finding |
|---------|----------------|---------|
| C1 — Training depth | Phase 1 extended to 5 tasks, dims OFF | mPCI = 0.2765 — partial confound |
| C2 — Scale robustness | σ = 0.02, 0.05, 0.10 | Robust at σ=0.05; near-zero at σ=0.10 |
| C3 — Shuffle | LZC on shuffled spike trains | Shuffle delta > original — statistics contribute |

**Honest claim:** This is the first systematic mPCI study on an affective SNN. The shift is reproducible across three seeds. Training depth and affective state both contribute. The affective contribution cannot be fully isolated with the current design. Independent replication is needed.

---

## Repository Structure

```
Maya-mPCI/
├── maya_cl/
│   ├── plasticity/
│   │   ├── mpci.py                  # LZC engine, perturbation protocol, mPCI computation
│   │   ├── karma.py                 # Karma accumulation and Shunyata pruning
│   │   ├── prana.py                 # Metabolic plasticity budget
│   │   └── ...                      # Other affective plasticity modules
│   ├── network/
│   │   ├── backbone.py              # MayaPranaNet architecture
│   │   └── affective_state.py       # AffectiveState tracker
│   └── ...
├── run_mpci_experiment.py           # Single-seed three-phase mPCI experiment
├── run_mpci_multiseed.py            # Multi-seed replication (seeds 42, 123, 7)
├── run_mpci_controls.py             # Three methodological controls
├── results/
│   ├── mpci_multiseed_results.json  # Full per-seed and aggregate results
│   ├── mpci_multiseed_summary.csv   # Clean summary table
│   ├── mpci_controls_results.json   # Controls results
│   ├── mpci_controls_summary.csv    # Controls summary
│   ├── phase1_seed42.pt             # Phase 1 checkpoint, seed 42
│   ├── phase2_seed42.pt             # Phase 2 checkpoint, seed 42
│   ├── phase3_seed42.pt             # Phase 3 checkpoint, seed 42
│   └── ...                          # Checkpoints for seeds 123, 7
├── docs/
│   ├── maya_mpci_dashboard.html     # Interactive trilingual dashboard (EN/HI/ZH)
│   └── maya_mpci_faq.html           # Exhaustive trilingual FAQ
├── Swaminathan_2026_Maya-mPCI_SNNv4.docx   # Paper
└── README.md
```

---

## How to Run

### Prerequisites

```bash
Python 3.11.9
PyTorch 2.5.1+cu121
SpikingJelly 0.0.0.0.14
CUDA (NVIDIA GPU — tested on RTX 4060 8GB)
```

```bash
git clone https://github.com/venky2099/Maya-mPCI
cd Maya-mPCI
pip install -r requirements.txt
```

### Single-seed experiment (seed 42, ~2 hours)

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m run_mpci_experiment 2>&1 | Tee-Object -FilePath "results\log.txt"
```

Expected output:
```
Phase 1 mPCI: 0.3091 +/- 0.0107
Phase 2 mPCI: 0.2696 +/- 0.0074
Phase 3 mPCI: 0.2664 +/- 0.0112
Delta: -0.0427 | Threshold: 0.0195 | >>> GENUINE INTERNAL STATE
```

### Multi-seed replication (seeds 42, 123, 7 · ~6 hours)

```powershell
# Prevent Windows sleep first
powercfg /change standby-timeout-ac 0

$env:PYTHONIOENCODING = "utf-8"
python -m run_mpci_multiseed 2>&1 | Tee-Object -FilePath "results\multiseed_log.txt"
```

Expected output summary:
```
Phase 1 aggregate mPCI: 0.3098 +/- 0.0099
Phase 2 aggregate mPCI: 0.2846 +/- 0.0161
Phase 3 aggregate mPCI: 0.2608 +/- 0.0147
Delta Phase3 vs Phase1 (aggregate): -0.0489
Delta std across seeds: 0.0048
Threshold (2x pooled SD): 0.0238
>>> GENUINE INTERNAL STATE: delta exceeds threshold.
```

### Methodological controls (~3 hours, requires multiseed checkpoints)

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m run_mpci_controls 2>&1 | Tee-Object -FilePath "results\controls_log.txt"
```

---

## The Maya Research Series

This paper is the post-series consciousness study that all nine papers made possible.

| Paper | Title | Key Result | DOI |
|-------|-------|-----------|-----|
| P1 | Nociceptive Metaplasticity | 66.6% velocity elevation | [zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS | Emergent safety primitive | [zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL | AA=62.38% TIL | [zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti | AA=31.84% CIL | [zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | Maya-Viveka | AA=16.03% | [zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| P6 | Maya-Chitta | AA=14.42% | [zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) |
| P7 | Maya-Manas | AA=15.19%, best BWT | [zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) |
| P8 | Maya-Śūnyatā | Vairagya×Karma | [zenodo.19397010](https://doi.org/10.5281/zenodo.19397010) |
| P9 | Maya-Prana | AA=12.72%, robot Bhaya=0.00 | [zenodo.19451174](https://doi.org/10.5281/zenodo.19451174) |
| **mPCI ★** | **This paper** | **Δ=−0.0489, 2.05× threshold** | [zenodo.19482794](https://doi.org/10.5281/zenodo.19482794) |

Two series constants confirmed across all nine papers:
- **Bhaya Quiescence Law:** Bhaya = 0.000 under replay from Task 1 onwards
- **Buddhi S-curve determinism:** Identical S-curve trajectory P4–P9

---

## The Bhaya Quiescence Law

Across all nine papers, all ablation conditions, all seeds: under episodic replay, Maya's fear signal drops to exactly 0.000 from Task 1 onwards — without being programmed to. This spontaneous suppression is one of two confirmed series constants and is the phenomenon this paper's mPCI experiment is designed to probe at the causal complexity level.

Under the 5-epoch protocol used in this study, Bhaya stabilises at 0.016–0.024 — calibrated vigilance rather than absolute quiescence. The law holds in its asymptotic form; this study measures a biologically realistic intermediate equilibrium.

---

## Cite This Paper

```bibtex
@misc{swaminathan2026mpci,
  title     = {From Representation to Experience: An mPCI-Based Empirical Test 
                of Internal Affective State in a Neuromorphic Spiking Neural Network},
  author    = {Swaminathan, Venkatesh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19482794},
  url       = {https://doi.org/10.5281/zenodo.19482794}
}
```

---

## IP Notice

This repository contains an ORCID-derived magic number (0.002315) embedded in key hyperparameters (KARMA_DECAY_RATE), LSB steganographic signatures in all matplotlib figures, and a canary string (`MayaNexusVS2026NLL_Bengaluru_Narasimha`) logged at the start of every experiment run. These provide cryptographic provenance for all published results.

---

*Nexus Learning Labs, Bengaluru · UDYAM-KR-02-0122422*  
*ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)*
