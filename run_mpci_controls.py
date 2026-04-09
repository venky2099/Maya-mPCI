# run_mpci_controls.py -- Three methodological controls for Maya-mPCI
#
# Control 1: Training depth matched baseline
#   Phase 1 extended to 5 tasks (matching Phase 3) with affective dims OFF
#   Rules out training depth as confound for the mPCI shift
#
# Control 2: Perturbation scale robustness
#   Runs Phase 1 and Phase 3 at sigma = 0.02, 0.05, 0.10
#   Confirms shift is not artefact of one specific perturbation magnitude
#
# Control 3: Shuffle control
#   Computes LZC on randomly shuffled spike trains from Phase 1 and Phase 3
#   Confirms structural organisation -- not just statistics -- drives the shift
#
# All controls run across seeds 42, 123, 7
# Output: results/mpci_controls_results.json + results/mpci_controls_summary.csv
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
from maya_cl.plasticity.lability import LabilityMatrix
from maya_cl.plasticity.vairagya_decay import VairagyadDecay
from maya_cl.plasticity.viveka import VivekaConsistency
from maya_cl.plasticity.chitta import ChittaSamskara
from maya_cl.plasticity.manas import ManasConsistency
from maya_cl.plasticity.karma import KarmaShunyata
from maya_cl.plasticity.prana import PranaMetabolic
from maya_cl.plasticity.mpci import (
    compute_mpci, extract_spike_matrix,
    lempel_ziv_complexity, normalised_lzc
)
from maya_cl.training.replay_buffer import ReplayBuffer

SEEDS              = [42, 123, 7]
BASE_LR            = 0.01
N_REPLAY           = round(BATCH_SIZE * REPLAY_RATIO / (1.0 - REPLAY_RATIO))
N_PERTURBATIONS    = 10
QUIESCENCE_TASK    = 4
MPCI_BATCH_SIZE    = 32
PERTURBATION_SCALES = [0.02, 0.05, 0.10]
RESULTS_DIR        = "results"


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def get_mpci_batch(device, seed):
    set_seed(seed)
    _, test_loader = get_task_loaders(0)
    encoder = PoissonEncoder(T_STEPS)
    for images, _ in test_loader:
        images = images[:MPCI_BATCH_SIZE].to(device)
        return encoder(images)


def train_one_task(task_id, model, encoder, optimizer, criterion,
                   affect, plasticity, replay_buffer, device,
                   use_affective=True, use_replay=True):
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

                chitta_gate = plasticity["chitta"].compute_gradient_gate(
                    active_fc1, tasks_seen=plasticity["tasks_seen"])
                if model.fc1.fc.weight.grad is not None:
                    plasticity["chitta"].apply_gradient_gate(
                        model.fc1.fc.weight.grad, chitta_gate)

                viveka_gain = plasticity["viveka"].compute_gain(
                    active_fc1,
                    viveka_signal=affect.viveka_signal(),
                    tasks_seen=plasticity["tasks_seen"]
                )

                if peak_active is not None:
                    peak_active_mask = peak_active.unsqueeze(1).expand(fc1_shape)
                    plasticity["manas_cons"].update(peak_active_mask)

                manas_gane_mask = plasticity["manas_cons"].compute_manas_gane_mask(
                    plasticity["viveka"].scores)

                pain_mask = active_fc1 & torch.zeros(
                    fc1_shape, dtype=torch.bool, device=device)

                plasticity["vairagya_fc1"].accumulate(
                    manas_gane_mask, pain_mask,
                    bhaya=float(affect.bhaya.item()),
                    buddhi=buddhi_val,
                    viveka_gain=viveka_gain
                )

                plasticity["chitta"].update(active_fc1)
                plasticity["viveka"].update(active_fc1)
                bhaya_log.append(float(affect.bhaya.item()))

            optimizer.step()

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
    plasticity["w_prev"] = None


# ── Control 1: Training depth matched baseline ────────────────────────────────

def run_phase1_extended(device, encoder, mpci_batch, seed, sigma=0.05):
    """
    Phase 1 extended to QUIESCENCE_TASK+1 tasks with affective dims OFF.
    Matches the training depth of Phase 3 exactly.
    If mPCI here is similar to Phase 1 original (3 tasks), training depth
    is not driving the Phase 3 shift.
    """
    set_seed(seed)
    model        = build_model(device)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    affect       = AffectiveState(device)
    plasticity   = build_plasticity(model, device)
    replay_buffer= ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)

    for task_id in range(QUIESCENCE_TASK + 1):
        if task_id > 0:
            on_task_boundary(task_id, model, affect, plasticity, device,
                             use_affective=False)
        train_one_task(task_id, model, encoder, optimizer, criterion,
                       affect, plasticity, replay_buffer, device,
                       use_affective=False, use_replay=False)

    result = compute_mpci(model, mpci_batch, device,
                          perturbation_scale=sigma,
                          n_perturbations=N_PERTURBATIONS,
                          seed_base=seed)
    result["control"] = "phase1_extended"
    result["n_tasks"]  = QUIESCENCE_TASK + 1
    result["sigma"]    = sigma
    return result


# ── Control 2: Perturbation scale robustness ──────────────────────────────────

def run_scale_robustness(model_p1, model_p3, mpci_batch, device, seed):
    """
    Run mPCI at three perturbation scales on pre-trained Phase 1 and Phase 3 models.
    Models are loaded from checkpoints saved during multiseed run.
    If shift is consistent across scales, result is robust.
    """
    results = {}
    for sigma in PERTURBATION_SCALES:
        r1 = compute_mpci(model_p1, mpci_batch, device,
                          perturbation_scale=sigma,
                          n_perturbations=N_PERTURBATIONS,
                          seed_base=seed)
        r3 = compute_mpci(model_p3, mpci_batch, device,
                          perturbation_scale=sigma,
                          n_perturbations=N_PERTURBATIONS,
                          seed_base=seed)
        delta = r3["mpci_mean"] - r1["mpci_mean"]
        results[f"sigma_{sigma}"] = {
            "sigma":        sigma,
            "phase1_mpci":  r1["mpci_mean"],
            "phase3_mpci":  r3["mpci_mean"],
            "delta":        delta,
            "phase1_std":   r1["mpci_std"],
            "phase3_std":   r3["mpci_std"],
        }
        print(f"    sigma={sigma}: P1={r1['mpci_mean']:.4f} P3={r3['mpci_mean']:.4f} delta={delta:+.4f}")
    return results


# ── Control 3: Shuffle control ────────────────────────────────────────────────

def compute_shuffle_lzc(model, spike_seq, device, n_shuffles=10, seed_base=42):
    """
    Extract spike train from model, randomly shuffle bits, compute LZC.
    If shuffled LZC is similar across phases, structural organisation
    -- not just spike statistics -- drives the observed mPCI differences.
    """
    binary = extract_spike_matrix(model, spike_seq, device)
    rng    = np.random.default_rng(seed_base)

    shuffle_lzc = []
    for i in range(n_shuffles):
        shuffled = binary.copy()
        rng.shuffle(shuffled)
        shuffle_lzc.append(normalised_lzc(shuffled))

    return {
        "original_lzc":  normalised_lzc(binary),
        "shuffle_mean":  float(np.mean(shuffle_lzc)),
        "shuffle_std":   float(np.std(shuffle_lzc)),
        "shuffle_raw":   shuffle_lzc,
        "spike_density": float(binary.mean()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("MayaNexusVS2026NLL_Bengaluru_Narasimha")
    print("\nMaya-mPCI Controls Experiment")
    print("Control 1: Training depth matched baseline")
    print("Control 2: Perturbation scale robustness")
    print("Control 3: Shuffle control")
    print(f"Seeds: {SEEDS}\n")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PoissonEncoder(T_STEPS)

    all_results = {
        "control1_depth_matched": {},
        "control2_scale_robustness": {},
        "control3_shuffle": {},
    }

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        mpci_batch = get_mpci_batch(device, seed)

        # ── Control 1 ─────────────────────────────────────────────────────────
        print(f"\n  Control 1: Phase 1 extended to {QUIESCENCE_TASK+1} tasks (seed={seed})")
        r_ext = run_phase1_extended(device, encoder, mpci_batch, seed)
        all_results["control1_depth_matched"][str(seed)] = r_ext
        print(f"  Phase 1-extended mPCI: {r_ext['mpci_mean']:.6f} +/- {r_ext['mpci_std']:.6f}")
        print(f"  (Compare to Phase 1 original ~0.3098, Phase 3 ~0.2608)")
        print(f"  If Phase 1-extended is close to Phase 1-original, depth confound is ruled out.")

        # ── Control 2 ─────────────────────────────────────────────────────────
        print(f"\n  Control 2: Perturbation scale robustness (seed={seed})")
        p1_ckpt = os.path.join(RESULTS_DIR, f"phase1_seed{seed}.pt")
        p3_ckpt = os.path.join(RESULTS_DIR, f"phase3_seed{seed}.pt")

        if os.path.exists(p1_ckpt) and os.path.exists(p3_ckpt):
            model_p1 = build_model(device)
            model_p1.load_state_dict(torch.load(p1_ckpt, map_location=device,
                                                weights_only=False))
            model_p1.eval()

            model_p3 = build_model(device)
            model_p3.load_state_dict(torch.load(p3_ckpt, map_location=device,
                                                weights_only=False))
            model_p3.eval()

            scale_results = run_scale_robustness(model_p1, model_p3, mpci_batch, device, seed)
            all_results["control2_scale_robustness"][str(seed)] = scale_results

            deltas = [scale_results[k]["delta"] for k in scale_results]
            consistent = all(d < -0.01 for d in deltas)
            print(f"  Shift consistent across scales: {consistent}")
            print(f"  Deltas: {[round(d,4) for d in deltas]}")
        else:
            print(f"  Checkpoints not found for seed {seed}. Run run_mpci_multiseed.py first.")
            all_results["control2_scale_robustness"][str(seed)] = {"error": "checkpoints not found"}

        # ── Control 3 ─────────────────────────────────────────────────────────
        print(f"\n  Control 3: Shuffle control (seed={seed})")
        if os.path.exists(p1_ckpt) and os.path.exists(p3_ckpt):
            shuffle_p1 = compute_shuffle_lzc(model_p1, mpci_batch, device,
                                             n_shuffles=10, seed_base=seed)
            shuffle_p3 = compute_shuffle_lzc(model_p3, mpci_batch, device,
                                             n_shuffles=10, seed_base=seed)

            shuffle_delta = shuffle_p3["shuffle_mean"] - shuffle_p1["shuffle_mean"]
            all_results["control3_shuffle"][str(seed)] = {
                "phase1": shuffle_p1,
                "phase3": shuffle_p3,
                "shuffle_delta": shuffle_delta,
            }
            print(f"  Phase 1 original LZC:  {shuffle_p1['original_lzc']:.6f}")
            print(f"  Phase 1 shuffled LZC:  {shuffle_p1['shuffle_mean']:.6f} +/- {shuffle_p1['shuffle_std']:.6f}")
            print(f"  Phase 3 original LZC:  {shuffle_p3['original_lzc']:.6f}")
            print(f"  Phase 3 shuffled LZC:  {shuffle_p3['shuffle_mean']:.6f} +/- {shuffle_p3['shuffle_std']:.6f}")
            print(f"  Shuffle delta (P3-P1): {shuffle_delta:+.6f}")
            print(f"  Interpretation: if shuffle delta << original delta, structure drives the shift.")
        else:
            print(f"  Skipped (checkpoints not found).")
            all_results["control3_shuffle"][str(seed)] = {"error": "checkpoints not found"}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONTROLS SUMMARY")
    print(f"{'='*60}")

    print("\n  Control 1 -- Training depth:")
    depth_mpci = [all_results["control1_depth_matched"][str(s)]["mpci_mean"] for s in SEEDS]
    print(f"  Phase 1-extended mPCI across seeds: {[round(x,4) for x in depth_mpci]}")
    print(f"  Mean: {np.mean(depth_mpci):.4f} +/- {np.std(depth_mpci):.4f}")
    print(f"  Phase 1-original mean was ~0.3098 | Phase 3 mean was ~0.2608")
    depth_confound_ruled_out = abs(np.mean(depth_mpci) - 0.3098) < 0.02
    print(f"  Depth confound ruled out: {depth_confound_ruled_out}")

    print("\n  Control 2 -- Scale robustness:")
    for seed in SEEDS:
        sr = all_results["control2_scale_robustness"].get(str(seed), {})
        if "error" not in sr:
            deltas = [sr[k]["delta"] for k in sr]
            print(f"  Seed {seed} deltas: {[round(d,4) for d in deltas]}")

    print("\n  Control 3 -- Shuffle:")
    for seed in SEEDS:
        sc = all_results["control3_shuffle"].get(str(seed), {})
        if "error" not in sc:
            orig_delta = (sc["phase3"]["original_lzc"] - sc["phase1"]["original_lzc"])
            shuf_delta = sc["shuffle_delta"]
            print(f"  Seed {seed}: original delta={orig_delta:+.4f} | shuffle delta={shuf_delta:+.4f}")

    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "mpci_controls_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {json_path}")

    csv_path = os.path.join(RESULTS_DIR, "mpci_controls_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["control", "seed", "metric", "value"])
        for seed in SEEDS:
            r = all_results["control1_depth_matched"].get(str(seed), {})
            if r:
                writer.writerow(["depth_matched", seed, "mpci_mean", round(r.get("mpci_mean",0), 6)])
                writer.writerow(["depth_matched", seed, "mpci_std",  round(r.get("mpci_std",0),  6)])
            sr = all_results["control2_scale_robustness"].get(str(seed), {})
            if sr and "error" not in sr:
                for k, v in sr.items():
                    writer.writerow(["scale_robustness", seed, f"delta_{v['sigma']}", round(v["delta"], 6)])
            sc = all_results["control3_shuffle"].get(str(seed), {})
            if sc and "error" not in sc:
                writer.writerow(["shuffle", seed, "original_delta",
                    round(sc["phase3"]["original_lzc"] - sc["phase1"]["original_lzc"], 6)])
                writer.writerow(["shuffle", seed, "shuffle_delta",
                    round(sc["shuffle_delta"], 6)])
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
