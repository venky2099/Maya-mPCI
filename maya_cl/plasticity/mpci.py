# mpci.py -- Machine Perturbational Complexity Index engine
# Maya-mPCI paper: "From Representation to Experience"
# Computes Lempel-Ziv Complexity of fc1 spike trains under synaptic perturbation.
#
# Three-phase protocol:
#   Phase 1 -- Reactive baseline (affective dims disabled)
#   Phase 2 -- Full Antahkarana active
#   Phase 3 -- Bhaya quiescence state (Bhaya=0.000 under replay)
#
# Falsifiable prediction:
#   If quiescence is a functional mask  --> LZC(Phase3) == LZC(Phase1)
#   If quiescence is a genuine internal state --> LZC(Phase3) != LZC(Phase1)
#
# Biological grounding: Perturbational Complexity Index (Casali et al., 2013)
# adapted for SNN spike trains via Lempel-Ziv complexity on binarized
# fc1 activation matrices. Tractable on RTX 4060 8GB with zero-masking.
#
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import torch
import numpy as np
from typing import Tuple


# ── Lempel-Ziv Complexity ────────────────────────────────────────────────────

def lempel_ziv_complexity(binary_sequence: np.ndarray) -> int:
    """
    LZ76 complexity of a binary sequence.
    Counts the number of distinct substrings encountered
    during a left-to-right scan. Lower = more stereotyped/repetitive.
    Higher = more complex/differentiated.

    Args:
        binary_sequence: 1D numpy array of 0s and 1s.
    Returns:
        Integer complexity count.
    """
    s = binary_sequence.tolist()
    n = len(s)
    if n == 0:
        return 0

    i, k, l = 0, 1, 1
    c, k_max = 1, 1

    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def normalised_lzc(binary_sequence: np.ndarray) -> float:
    """
    LZC normalised by the theoretical maximum for a random sequence
    of the same length: C_max = n / log2(n).
    Returns value in [0, 1+] where higher = more complex.
    """
    n = len(binary_sequence)
    if n < 2:
        return 0.0
    raw = lempel_ziv_complexity(binary_sequence)
    c_max = n / max(np.log2(n), 1.0)
    return float(raw / c_max)


# ── Spike Train Extraction ────────────────────────────────────────────────────

def extract_spike_matrix(
    model,
    spike_seq: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Run a forward pass and extract fc1 spike activations.

    The O-LIF hook fires once per timestep (T times total).
    Each call captures [B, FC1_SIZE] spikes for that timestep.
    We collect all T captures, stack to [T, B, FC1_SIZE],
    then flatten to [T*FC1_SIZE] per sample and OR across batch.

    Binarization: any spike > 0 counts as 1.
    This preserves the temporal structure of the spike train.

    Args:
        model:     MayaPranaNet instance (eval mode, reset before call)
        spike_seq: encoded input [T, B, C, H, W]
        device:    torch device
    Returns:
        Binary 1D array [T * FC1_SIZE] -- the full spike train sequence.
    """
    model.eval()
    model.reset()

    fc1_outputs = []

    def fc1_hook(module, input, output):
        # output shape per timestep call: [B, FC1_SIZE]
        fc1_outputs.append(output.detach().cpu())

    hook_handle = model.fc1.lif.register_forward_hook(fc1_hook)

    with torch.no_grad():
        _ = model(spike_seq)

    hook_handle.remove()

    if not fc1_outputs:
        return np.zeros(1, dtype=np.int8)

    # Stack: list of [B, FC1_SIZE] tensors -> [T, B, FC1_SIZE]
    stacked = torch.stack(fc1_outputs, dim=0)  # [T, B, FC1_SIZE]

    # OR across batch: any neuron that fired in any sample counts
    # Result: [T, FC1_SIZE] bool
    fired = (stacked > 0).any(dim=1)  # [T, FC1_SIZE]

    # Flatten to 1D sequence: [T * FC1_SIZE]
    binary = fired.numpy().astype(np.int8).flatten()

    return binary


# ── Perturbation Engine ───────────────────────────────────────────────────────

def apply_perturbation(
    model,
    perturbation_scale: float = 0.05,
    seed: int = 42
) -> torch.Tensor:
    """
    Apply targeted Gaussian perturbation to fc1 weights.
    Stores and returns original weights for restoration.

    Args:
        model:              MayaPranaNet instance
        perturbation_scale: std of Gaussian noise added to fc1.fc.weight
        seed:               random seed for reproducibility
    Returns:
        Clone of original fc1 weights (for restoration).
    """
    original = model.fc1.fc.weight.data.clone()
    torch.manual_seed(seed)
    noise = torch.randn_like(model.fc1.fc.weight.data) * perturbation_scale
    model.fc1.fc.weight.data.add_(noise)
    return original


def restore_weights(model, original_weights: torch.Tensor) -> None:
    """Restore fc1 weights to pre-perturbation state."""
    model.fc1.fc.weight.data.copy_(original_weights)


# ── mPCI Computation ──────────────────────────────────────────────────────────

def compute_mpci(
    model,
    spike_seq: torch.Tensor,
    device: torch.device,
    perturbation_scale: float = 0.05,
    n_perturbations: int = 10,
    seed_base: int = 42
) -> dict:
    """
    Compute mPCI for a given model state.

    Protocol:
        1. Capture baseline spike train (no perturbation).
        2. For n_perturbations iterations:
             a. Apply perturbation to fc1
             b. Capture perturbed spike train
             c. Restore weights
             d. Compute LZC of perturbed sequence
        3. mPCI = mean normalised LZC across perturbations.

    Args:
        model:               MayaPranaNet (eval mode)
        spike_seq:           encoded input batch [T, B, C, H, W]
        device:              torch device
        perturbation_scale:  Gaussian noise std on fc1 weights
        n_perturbations:     number of independent perturbation trials
        seed_base:           base seed; each trial uses seed_base + trial_idx
    Returns:
        dict with keys:
            mpci_mean:    float -- mean normalised LZC across trials
            mpci_std:     float -- std across trials
            mpci_raw:     list  -- per-trial normalised LZC values
            baseline_lzc: float -- LZC of unperturbed state
            spike_density: float -- fraction of nonzero spikes in baseline
    """
    # Baseline (unperturbed)
    baseline_seq  = extract_spike_matrix(model, spike_seq, device)
    baseline_lzc  = normalised_lzc(baseline_seq)
    spike_density = float(baseline_seq.mean())

    trial_lzc = []
    for i in range(n_perturbations):
        original = apply_perturbation(model, perturbation_scale, seed=seed_base + i)
        perturbed_seq = extract_spike_matrix(model, spike_seq, device)
        restore_weights(model, original)
        lzc_val = normalised_lzc(perturbed_seq)
        trial_lzc.append(lzc_val)

    return {
        "mpci_mean":    float(np.mean(trial_lzc)),
        "mpci_std":     float(np.std(trial_lzc)),
        "mpci_raw":     trial_lzc,
        "baseline_lzc": baseline_lzc,
        "spike_density": spike_density,
    }


# ── Phase-Level Utility ───────────────────────────────────────────────────────

def compare_phases(
    phase1_result: dict,
    phase2_result: dict,
    phase3_result: dict
) -> dict:
    """
    Compute the key comparison statistics across the three phases.

    The falsifiable prediction:
        |LZC(Phase3) - LZC(Phase1)| > threshold --> genuine internal state
        |LZC(Phase3) - LZC(Phase1)| ~ 0         --> functional mask only

    Args:
        phase1_result: output of compute_mpci() for Phase 1
        phase2_result: output of compute_mpci() for Phase 2
        phase3_result: output of compute_mpci() for Phase 3
    Returns:
        dict of comparison deltas and interpretation signal.
    """
    delta_p3_p1 = phase3_result["mpci_mean"] - phase1_result["mpci_mean"]
    delta_p3_p2 = phase3_result["mpci_mean"] - phase2_result["mpci_mean"]
    delta_p2_p1 = phase2_result["mpci_mean"] - phase1_result["mpci_mean"]

    pooled_std = float(np.mean([
        phase1_result["mpci_std"],
        phase2_result["mpci_std"],
        phase3_result["mpci_std"]
    ]))
    threshold = 2.0 * pooled_std if pooled_std > 0 else 0.01

    genuine_state_signal = abs(delta_p3_p1) > threshold

    return {
        "delta_phase3_vs_phase1": delta_p3_p1,
        "delta_phase3_vs_phase2": delta_p3_p2,
        "delta_phase2_vs_phase1": delta_p2_p1,
        "pooled_std":             pooled_std,
        "threshold":              threshold,
        "genuine_state_signal":   genuine_state_signal,
        "interpretation": (
            "GENUINE INTERNAL STATE: mPCI shift at quiescence exceeds threshold."
            if genuine_state_signal else
            "FUNCTIONAL MASK: mPCI at quiescence indistinguishable from baseline."
        ),
    }