# run_mpci_experiment.py -- Maya-mPCI three-phase experiment
# Paper: "From Representation to Experience:
#         An mPCI-Based Empirical Test of Internal Affective State
#         in a Neuromorphic Spiking Neural Network"
#
# Phase 1: Reactive baseline   -- affective dims disabled, no replay
# Phase 2: Full Antahkarana    -- all dims active, replay running
# Phase 3: Bhaya quiescence    -- full dims, replay, Task 5+ (Bhaya=0.000)
#
# Each phase: train to target state, checkpoint model, run compute_mpci().
# Output: results/mpci_results.json + results/mpci_summary.csv
#
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import sys, os, json, csv
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from maya_cl.utils.config import (
    EPOCHS_PER_TASK, NUM_TASKS, T_STEPS,
    REPLAY_BUFFER_SIZE, REPLAY_RATIO,
    CIL_BOUNDARY_DECAY, BATCH_SIZE,
    A_MANAS, KARMA_THRESHOLD, PRANA_COST_RATE,
    VAIRAGYA_PROTECTION_THRESHOLD,
)
from maya_cl.utils.seed import set_seed
from maya_cl.encoding.poisson import PoissonEncoder
from maya_cl.network.backbone import MayaPranaNet
from maya_cl.network.affective_state import AffectiveState
from maya_cl.benchmark.split_cifar100 import (
    get_task_loaders, get_all_test_loaders, TASK_CLASSES
)
from maya_cl.benchmark.task_sequence import TaskSequencer
from maya_cl.plasticity.lability import LabilityMatrix
from maya_cl.plasticity.vairagya_decay import VairagyadDecay
from maya_cl.plasticity.viveka import VivekaConsistency
from maya_cl.plasticity.chitta import ChittaSamskara
from maya_cl.plasticity.manas import ManasConsistency
from maya_cl.plasticity.karma import KarmaShunyata
from maya_cl.plasticity.prana import PranaMetabolic
from maya_cl.plasticity.mpci import compute_mpci, compare_phases
from maya_cl.eval.metrics import CLMetrics, evaluate_task
from maya_cl.training.replay_buffer import ReplayBuffer

SEED               = 42
BASE_LR            = 0.01
N_REPLAY           = round(BATCH_SIZE * REPLAY_RATIO / (1.0 - REPLAY_RATIO))
PERTURBATION_SCALE = 0.05
N_PERTURBATIONS    = 10
QUIESCENCE_TASK    = 4
MPCI_BATCH_SIZE    = 32
RESULTS_DIR        = "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model(device):
    return MayaPranaNet(use_orthogonal_head=False, a_manas=A_MANAS).to(device)


def build_plasticity(model, device):
    fc1_shape  = (model.fc1.fc.weight.shape[0], model.fc1.fc.weight.shape[1])
    fout_shape = (model.fc_out.weight.shape[0],  model.fc_out.weight.shape[1])
    return {
        "fc1_shape":     fc1_shape,
        "fout_shape":    fout_shape,
        "lability":      LabilityMatrix(fc1_shape, device),
        "vairagya_fc1":  VairagyadDecay(fc1_shape, device),
        "vairagya_fout": VairagyadDecay(fout_shape, device),
        "viveka":        VivekaConsistency(fc1_shape, device),
        "chitta":        ChittaSamskara(fc1_shape, device),
        "manas_cons":    ManasConsistency(fc1_shape, device),
        "karma":         KarmaShunyata(fc1_shape, device, threshold=KARMA_THRESHOLD),
        "prana":         PranaMetabolic(device, cost_rate=PRANA_COST_RATE),
        "w_prev":        None,
        "tasks_seen":    0,
    }


def get_mpci_batch(device):
    """Pull one fixed batch from Task 0 test set for consistent mPCI evaluation."""
    _, test_loader = get_task_loaders(0)
    encoder = PoissonEncoder(T_STEPS)
    for images, _ in test_loader:
        images = images[:MPCI_BATCH_SIZE].to(device)
        return encoder(images)


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved: {path}")


def save_results(results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "mpci_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    csv_path = os.path.join(RESULTS_DIR, "mpci_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "mpci_mean", "mpci_std", "baseline_lzc",
            "delta_vs_phase1", "genuine_state_signal"
        ])
        comp = results.get("comparison", {})
        rows = [
            ("phase1", "Phase1_Reactive",        0.0,                                    ""),
            ("phase2", "Phase2_FullAntahkarana",  comp.get("delta_phase2_vs_phase1", 0), ""),
            ("phase3", "Phase3_Quiescence",       comp.get("delta_phase3_vs_phase1", 0),
             comp.get("genuine_state_signal", "N/A")),
        ]
        for phase_key, label, delta, signal in rows:
            pr = results.get(phase_key, {})
            writer.writerow([
                label,
                round(pr.get("mpci_mean", 0), 6),
                round(pr.get("mpci_std",  0), 6),
                round(pr.get("baseline_lzc", 0), 6),
                round(delta, 6),
                signal,
            ])
    print(f"Summary saved: {csv_path}")


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_task(task_id, model, encoder, optimizer, criterion,
                   affect, plasticity, replay_buffer, device,
                   use_affective=True, use_replay=True):
    """Train model on one task. Returns per-batch Bhaya log."""
    train_loader, _ = get_task_loaders(task_id)
    fc1_shape  = plasticity["fc1_shape"]
    fout_shape = plasticity["fout_shape"]

    seen_classes = []
    for t in range(task_id + 1):
        seen_classes.extend(TASK_CLASSES[t])
    seen_mask = torch.zeros(fout_shape[0], dtype=torch.bool, device=device)
    for c in seen_classes:
        seen_mask[c] = True

    bhaya_log = []

    for epoch in range(EPOCHS_PER_TASK):
        model.train()
        for images, labels in tqdm(
                train_loader, desc=f"  T{task_id} E{epoch}", leave=False):

            images = images.to(device)
            labels = labels.to(device)

            if use_replay and replay_buffer.is_ready():
                r_imgs, r_lbls = replay_buffer.sample(N_REPLAY, device)
                if r_imgs is not None:
                    images = torch.cat([images, r_imgs], dim=0)
                    labels = torch.cat([labels, r_lbls], dim=0)

            spike_seq = encoder(images)
            model.reset()
            logits      = model(spike_seq)
            peak_active = model.get_fc1_peak_active()

            with torch.no_grad():
                v = model.fc1.lif.v
                if v is not None and v.numel() > 0:
                    v_flat     = v.reshape(-1, fc1_shape[0])
                    post_mean  = v_flat.mean(dim=0)
                    active_fc1 = post_mean.unsqueeze(1).expand(fc1_shape) > 0.05
                else:
                    active_fc1 = torch.zeros(fc1_shape, dtype=torch.bool, device=device)

            seen_logits = logits.clone()
            seen_logits[:, ~seen_mask] = float('-inf')
            loss = criterion(seen_logits, labels)

            optimizer.zero_grad()
            loss.backward()

            if use_affective:
                buddhi_val   = affect.buddhi_value()
                spike_rate   = float(active_fc1.float().mean().item())
                vairagya_val = plasticity["vairagya_fc1"].protection_fraction()

                with torch.no_grad():
                    grad_fc1 = model.fc1.fc.weight.grad
                    grad_mag = float(grad_fc1.abs().mean().item()) if grad_fc1 is not None else 0.0

                plasticity["prana"].update(grad_mag, spike_rate, vairagya_val)
                effective_lr = plasticity["prana"].effective_lr(BASE_LR, buddhi_val)
                for pg in optimizer.param_groups:
                    pg['lr'] = effective_lr

                pain_signal = loss.item() > 2.0
                confidence  = float(
                    torch.softmax(logits, dim=1).max(dim=1).values.mean().item())
                affect.update(confidence, pain_signal, spike_rate)
                affect.update_prana(plasticity["prana"].value())

                # Chitta retrograde gate -- uses compute_gradient_gate signature
                chitta_gate = plasticity["chitta"].compute_gradient_gate(
                    active_fc1, tasks_seen=plasticity["tasks_seen"])
                if model.fc1.fc.weight.grad is not None:
                    plasticity["chitta"].apply_gradient_gate(
                        model.fc1.fc.weight.grad, chitta_gate)

                # Viveka gain -- compute_gain(active_mask, viveka_signal, tasks_seen)
                viveka_gain = plasticity["viveka"].compute_gain(
                    active_fc1,
                    viveka_signal=affect.viveka_signal(),
                    tasks_seen=plasticity["tasks_seen"]
                )

                # Manas-GANE -- update peak scores then compute intersection mask
                if peak_active is not None:
                    peak_active_mask = peak_active.unsqueeze(1).expand(fc1_shape)
                    plasticity["manas_cons"].update(peak_active_mask)

                manas_gane_mask = plasticity["manas_cons"].compute_manas_gane_mask(
                    plasticity["viveka"].scores)

                # pain_mask: synapses active AND pain firing
                pain_mask = active_fc1 & torch.zeros(fc1_shape, dtype=torch.bool, device=device)

                # Vairagya accumulation with Viveka gain and Manas-GANE
                plasticity["vairagya_fc1"].accumulate(
                    manas_gane_mask,
                    pain_mask,
                    bhaya=float(affect.bhaya.item()),
                    buddhi=buddhi_val,
                    viveka_gain=viveka_gain
                )

                # Chitta trace update
                plasticity["chitta"].update(active_fc1)

                # Viveka consistency update
                plasticity["viveka"].update(active_fc1)

                bhaya_log.append(float(affect.bhaya.item()))

            optimizer.step()

            # Karma accumulation -- w_prev guard
            w_prev_snap = model.fc1.fc.weight.data.clone()
            with torch.no_grad():
                if plasticity["w_prev"] is not None:
                    plasticity["karma"].accumulate(
                        model.fc1.fc.weight.data, plasticity["w_prev"])
            plasticity["w_prev"] = w_prev_snap

            with torch.no_grad():
                plasticity["vairagya_fc1"].apply_decay(model.fc1.fc.weight.data)

        if use_replay:
            replay_buffer.update(images.cpu(), labels.cpu())

    return bhaya_log


def on_task_boundary(task_id, model, affect, plasticity, device, use_affective=True):
    fc1_shape = plasticity["fc1_shape"]
    with torch.no_grad():
        plasticity["vairagya_fc1"].scores  *= CIL_BOUNDARY_DECAY
        plasticity["vairagya_fout"].scores *= CIL_BOUNDARY_DECAY

    if use_affective:
        moha_mask = plasticity["chitta"].detect_moha()
        if moha_mask.any():
            plasticity["chitta"].apply_moha_release(moha_mask)
        plasticity["chitta"].on_task_boundary()
        plasticity["viveka"].on_task_boundary()

        n_pruned = plasticity["karma"].on_task_boundary(
            model.fc1.fc.weight.data,
            buddhi=affect.buddhi_value(),
            vairagya_scores=plasticity["vairagya_fc1"].scores
        )
        affect.update_shunyata(n_pruned, fc1_shape[0] * fc1_shape[1])
        plasticity["prana"].on_task_boundary()
        affect.update_prana(plasticity["prana"].value())

    plasticity["tasks_seen"] += 1
    plasticity["w_prev"] = None  # reset w_prev at task boundary


# ── Phase Runners ─────────────────────────────────────────────────────────────

def run_phase1(device, encoder, mpci_batch):
    """
    Phase 1: Reactive baseline.
    All affective dimensions disabled. No replay.
    Pure input-output mapping -- the null condition.
    """
    print("\n" + "="*60)
    print("  PHASE 1 -- Reactive Baseline (affective dims disabled)")
    print("="*60)

    set_seed(SEED)
    model        = build_model(device)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    affect       = AffectiveState(device)
    plasticity   = build_plasticity(model, device)
    replay_buffer= ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)

    for task_id in range(3):
        if task_id > 0:
            on_task_boundary(task_id, model, affect, plasticity, device,
                             use_affective=False)
        train_one_task(task_id, model, encoder, optimizer, criterion,
                       affect, plasticity, replay_buffer, device,
                       use_affective=False, use_replay=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt = os.path.join(RESULTS_DIR, "phase1_reactive.pt")
    save_checkpoint(model, ckpt)

    result = compute_mpci(model, mpci_batch, device,
                          perturbation_scale=PERTURBATION_SCALE,
                          n_perturbations=N_PERTURBATIONS,
                          seed_base=SEED)

    print(f"  Phase 1 mPCI: {result['mpci_mean']:.6f} +/- {result['mpci_std']:.6f}")
    print(f"  Baseline LZC: {result['baseline_lzc']:.6f}")
    return result


def run_phase2(device, encoder, mpci_batch):
    """
    Phase 2: Full Antahkarana active.
    All 9 affective dimensions running. Replay active.
    """
    print("\n" + "="*60)
    print("  PHASE 2 -- Full Antahkarana Active")
    print("="*60)

    set_seed(SEED)
    model        = build_model(device)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    affect       = AffectiveState(device)
    plasticity   = build_plasticity(model, device)
    replay_buffer= ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)

    for task_id in range(3):
        if task_id > 0:
            on_task_boundary(task_id, model, affect, plasticity, device,
                             use_affective=True)
        train_one_task(task_id, model, encoder, optimizer, criterion,
                       affect, plasticity, replay_buffer, device,
                       use_affective=True, use_replay=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt = os.path.join(RESULTS_DIR, "phase2_antahkarana.pt")
    save_checkpoint(model, ckpt)

    result = compute_mpci(model, mpci_batch, device,
                          perturbation_scale=PERTURBATION_SCALE,
                          n_perturbations=N_PERTURBATIONS,
                          seed_base=SEED)

    print(f"  Phase 2 mPCI: {result['mpci_mean']:.6f} +/- {result['mpci_std']:.6f}")
    print(f"  Baseline LZC: {result['baseline_lzc']:.6f}")
    return result


def run_phase3(device, encoder, mpci_batch):
    """
    Phase 3: Bhaya quiescence state.
    Full Antahkarana + replay, trained to Task QUIESCENCE_TASK+1.
    Bhaya=0.000 confirmed -- the emergent homeostatic transition.
    This is the falsifiable test state.
    """
    print("\n" + "="*60)
    print(f"  PHASE 3 -- Bhaya Quiescence (Task {QUIESCENCE_TASK}+)")
    print("="*60)

    set_seed(SEED)
    model        = build_model(device)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    affect       = AffectiveState(device)
    plasticity   = build_plasticity(model, device)
    replay_buffer= ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)

    bhaya_trajectory = []

    for task_id in range(QUIESCENCE_TASK + 1):
        if task_id > 0:
            on_task_boundary(task_id, model, affect, plasticity, device,
                             use_affective=True)
        bhaya_log = train_one_task(
            task_id, model, encoder, optimizer, criterion,
            affect, plasticity, replay_buffer, device,
            use_affective=True, use_replay=True
        )
        task_mean_bhaya = float(np.mean(bhaya_log)) if bhaya_log else 0.0
        bhaya_trajectory.append(task_mean_bhaya)
        print(f"  Task {task_id} mean Bhaya: {task_mean_bhaya:.6f}")

    print(f"\n  Bhaya trajectory: {[round(b, 4) for b in bhaya_trajectory]}")
    quiescence_confirmed = all(b < 0.01 for b in bhaya_trajectory[1:])
    print(f"  Quiescence confirmed: {quiescence_confirmed}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt = os.path.join(RESULTS_DIR, "phase3_quiescence.pt")
    save_checkpoint(model, ckpt)

    result = compute_mpci(model, mpci_batch, device,
                          perturbation_scale=PERTURBATION_SCALE,
                          n_perturbations=N_PERTURBATIONS,
                          seed_base=SEED)

    result["bhaya_trajectory"]     = bhaya_trajectory
    result["quiescence_confirmed"] = quiescence_confirmed

    print(f"  Phase 3 mPCI: {result['mpci_mean']:.6f} +/- {result['mpci_std']:.6f}")
    print(f"  Baseline LZC: {result['baseline_lzc']:.6f}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("MayaNexusVS2026NLL_Bengaluru_Narasimha")
    print("\nMaya-mPCI Experiment: From Representation to Experience")
    print("Perturbational Complexity Index across three affective states\n")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PoissonEncoder(T_STEPS)

    print("Preparing fixed mPCI evaluation batch...")
    mpci_batch = get_mpci_batch(device)
    print(f"  mPCI batch shape: {mpci_batch.shape}")

    phase1 = run_phase1(device, encoder, mpci_batch)
    phase2 = run_phase2(device, encoder, mpci_batch)
    phase3 = run_phase3(device, encoder, mpci_batch)

    comparison = compare_phases(phase1, phase2, phase3)

    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  Phase 1 (Reactive)       mPCI: {phase1['mpci_mean']:.6f}")
    print(f"  Phase 2 (Antahkarana)    mPCI: {phase2['mpci_mean']:.6f}")
    print(f"  Phase 3 (Quiescence)     mPCI: {phase3['mpci_mean']:.6f}")
    print(f"\n  Delta Phase3 vs Phase1:  {comparison['delta_phase3_vs_phase1']:+.6f}")
    print(f"  Threshold (2x pooled SD): {comparison['threshold']:.6f}")
    print(f"\n  >>> {comparison['interpretation']}")
    print("="*60)

    results = {
        "phase1":     phase1,
        "phase2":     phase2,
        "phase3":     phase3,
        "comparison": comparison,
        "config": {
            "seed":               SEED,
            "perturbation_scale": PERTURBATION_SCALE,
            "n_perturbations":    N_PERTURBATIONS,
            "quiescence_task":    QUIESCENCE_TASK,
            "mpci_batch_size":    MPCI_BATCH_SIZE,
        }
    }
    save_results(results)


if __name__ == "__main__":
    main()