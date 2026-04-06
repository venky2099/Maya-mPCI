# run_ablation_prana.py -- Maya-Prana Paper 9 ablation study
# Split-CIFAR-100 CIL, 6 conditions
#
# A: P8 baseline (no Prana)              -- reference ~15.19% AA
# B: Prana only, no other affective dims -- unregulated budget, expected collapse
# C: Fixed Prana (constant 1.0)          -- structural check, should match A
# D: Full Maya-Prana, calibrated         -- canonical starred
# E: Aggressive depletion rate           -- learning starvation risk
# F: Prana without Buddhi modulation     -- tests mature regulation
#
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm

from maya_cl.utils.config import (
    EPOCHS_PER_TASK, NUM_TASKS, T_STEPS,
    REPLAY_BUFFER_SIZE, REPLAY_RATIO,
    CIL_BOUNDARY_DECAY, BATCH_SIZE,
    REPLAY_PAIN_EXEMPT, A_MANAS,
    KARMA_THRESHOLD, PRANA_COST_RATE, PRANA_AGGRESSIVE_COST,
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
from maya_cl.eval.metrics import CLMetrics, evaluate_task
from maya_cl.eval.logger import RunLogger
from maya_cl.training.replay_buffer import ReplayBuffer

N_REPLAY = round(BATCH_SIZE * REPLAY_RATIO / (1.0 - REPLAY_RATIO))
BASE_LR  = 0.01

CONDITIONS = {
    'A_baseline': {
        'use_prana':       False,
        'use_chitta':      True,
        'use_viveka':      True,
        'use_manas_gane':  True,
        'buddhi_gate':     True,
        'prana_cost_rate': PRANA_COST_RATE,
        'fixed_prana':     False,
        'description':     'P8 Maya-Shunyata baseline -- no Prana',
    },
    'B_prana_only': {
        'use_prana':       True,
        'use_chitta':      False,
        'use_viveka':      False,
        'use_manas_gane':  False,
        'buddhi_gate':     True,
        'prana_cost_rate': PRANA_COST_RATE,
        'fixed_prana':     False,
        'description':     'Prana only -- no Chitta, Viveka, Manas-GANE',
    },
    'C_fixed_prana': {
        'use_prana':       True,
        'use_chitta':      True,
        'use_viveka':      True,
        'use_manas_gane':  True,
        'buddhi_gate':     True,
        'prana_cost_rate': PRANA_COST_RATE,
        'fixed_prana':     True,
        'description':     'Fixed Prana=1.0 constant -- structural check, expected ~A',
    },
    'D_canonical': {
        'use_prana':       True,
        'use_chitta':      True,
        'use_viveka':      True,
        'use_manas_gane':  True,
        'buddhi_gate':     True,
        'prana_cost_rate': PRANA_COST_RATE,
        'fixed_prana':     False,
        'description':     'Full Maya-Prana canonical -- starred',
    },
    'E_aggressive_depletion': {
        'use_prana':       True,
        'use_chitta':      True,
        'use_viveka':      True,
        'use_manas_gane':  True,
        'buddhi_gate':     True,
        'prana_cost_rate': PRANA_AGGRESSIVE_COST,
        'fixed_prana':     False,
        'description':     f'Aggressive depletion rate={PRANA_AGGRESSIVE_COST} -- starvation risk',
    },
    'F_no_buddhi_gate': {
        'use_prana':       True,
        'use_chitta':      True,
        'use_viveka':      True,
        'use_manas_gane':  True,
        'buddhi_gate':     False,
        'prana_cost_rate': PRANA_COST_RATE,
        'fixed_prana':     False,
        'description':     'Prana without Buddhi modulation -- tests mature regulation',
    },
}


def run_condition(condition_name: str, seed: int = 42) -> dict:
    print("MayaNexusVS2026NLL_Bengaluru_Narasimha")
    assert condition_name in CONDITIONS
    cfg = CONDITIONS[condition_name]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Condition : {condition_name}")
    print(f"  {cfg['description']}")
    print(f"  Seed      : {seed}")
    print(f"{'='*60}")

    model         = MayaPranaNet(use_orthogonal_head=False, a_manas=A_MANAS).to(device)
    encoder       = PoissonEncoder(T_STEPS)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    affect        = AffectiveState(device)
    sequencer     = TaskSequencer()
    metrics       = CLMetrics(NUM_TASKS)
    logger        = RunLogger(f"ablation_{condition_name}")
    test_loaders  = get_all_test_loaders()
    replay_buffer = ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)

    fc1_shape  = (model.fc1.fc.weight.shape[0], model.fc1.fc.weight.shape[1])
    fout_shape = (model.fc_out.weight.shape[0], model.fc_out.weight.shape[1])

    lability_fc1  = LabilityMatrix(fc1_shape,  device)
    vairagya_fc1  = VairagyadDecay(fc1_shape,  device)
    vairagya_fout = VairagyadDecay(fout_shape, device)
    viveka        = VivekaConsistency(fc1_shape, device)
    chitta        = ChittaSamskara(fc1_shape, device)
    manas_cons    = ManasConsistency(fc1_shape, device)
    karma         = KarmaShunyata(fc1_shape, device, threshold=KARMA_THRESHOLD)
    prana         = PranaMetabolic(device, cost_rate=cfg['prana_cost_rate'])

    w_prev     = model.fc1.fc.weight.data.clone()
    prev_loss  = None
    tasks_seen = 0

    for task_id in range(NUM_TASKS):
        train_loader, _ = get_task_loaders(task_id)
        sequencer.current_task = task_id

        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(TASK_CLASSES[t])

        seen_mask = torch.zeros(fout_shape[0], dtype=torch.bool, device=device)
        for c in seen_classes:
            seen_mask[c] = True

        if task_id > 0:
            with torch.no_grad():
                vairagya_fc1.scores  *= CIL_BOUNDARY_DECAY
                vairagya_fout.scores *= CIL_BOUNDARY_DECAY

            if cfg['use_chitta']:
                moha_mask = chitta.detect_moha()
                if moha_mask.any():
                    chitta.apply_moha_release(moha_mask)
                chitta.on_task_boundary()

            if cfg['use_viveka']:
                viveka.on_task_boundary()

            n_pruned = karma.on_task_boundary(
                model.fc1.fc.weight.data,
                buddhi=affect.buddhi_value(),
                vairagya_scores=vairagya_fc1.scores)
            affect.update_shunyata(n_pruned, fc1_shape[0] * fc1_shape[1])

            if cfg['use_prana']:
                prana.on_task_boundary()
                affect.update_prana(prana.value())

            tasks_seen += 1

        print(f"\nTask {task_id} | classes {TASK_CLASSES[task_id]}")

        for epoch in range(EPOCHS_PER_TASK):
            epoch_loss = 0.0
            model.train()

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  T{task_id} E{epoch}", leave=False)):

                images = images.to(device)
                labels = labels.to(device)

                is_replay_batch = False
                if replay_buffer.is_ready():
                    r_imgs, r_lbls = replay_buffer.sample(N_REPLAY, device)
                    if r_imgs is not None:
                        images = torch.cat([images, r_imgs], dim=0)
                        labels = torch.cat([labels, r_lbls], dim=0)
                        is_replay_batch = True

                spike_seq = encoder(images)
                model.reset()
                logits      = model(spike_seq)
                peak_active = model.get_fc1_peak_active()

                # active_fc1 [FC1_SIZE, in_features] -- membrane voltage proxy
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

                buddhi_val   = affect.buddhi_value()
                spike_rate   = float(active_fc1.float().mean().item())
                vairagya_val = vairagya_fc1.protection_fraction()

                # Prana gating
                if cfg['use_prana'] and not cfg['fixed_prana']:
                    grad_mag = 0.0
                    if model.fc1.fc.weight.grad is not None:
                        grad_mag = float(model.fc1.fc.weight.grad.abs().mean().item())
                    prana.update(grad_mag, spike_rate, vairagya_val)
                    affect.update_prana(prana.value())
                    gate_buddhi = buddhi_val if cfg['buddhi_gate'] else 0.5
                    eff_lr = prana.effective_lr(BASE_LR, gate_buddhi)
                elif cfg['use_prana'] and cfg['fixed_prana']:
                    eff_lr = BASE_LR * (0.5 + buddhi_val * 0.5)
                else:
                    eff_lr = BASE_LR

                for pg in optimizer.param_groups:
                    pg['lr'] = eff_lr

                if cfg['use_chitta']:
                    chitta_gate      = chitta.compute_gradient_gate(active_fc1, tasks_seen)
                    retrograde_fired = chitta_gate.mean().item() < 1.0
                    if model.fc1.fc.weight.grad is not None:
                        chitta.apply_gradient_gate(model.fc1.fc.weight.grad, chitta_gate)
                        affect.update_chitta(
                            True, float((1.0 - chitta_gate).mean().item()))
                else:
                    retrograde_fired = False

                manas_gane_mask = torch.zeros(fc1_shape, dtype=torch.bool, device=device)
                if cfg['use_manas_gane'] and peak_active is not None:
                    peak_2d         = peak_active.unsqueeze(1).expand(fc1_shape)
                    manas_gane_mask = active_fc1 & peak_2d

                optimizer.step()

                with torch.no_grad():
                    w_current = model.fc1.fc.weight.data
                    karma.accumulate(w_current, w_prev)
                    karma.apply_mask(model.fc1.fc.weight.data)
                    w_prev = w_current.clone()

                epoch_loss += loss.item()

                with torch.no_grad():
                    cur_loss = loss.item()
                    conf     = sequencer.update_confidence(logits)

                    if REPLAY_PAIN_EXEMPT and is_replay_batch:
                        pain = False
                    else:
                        pain = sequencer.check_pain_signal(cur_loss, prev_loss, conf)
                        prev_loss = cur_loss

                    affect.update(conf, pain, spike_rate)
                    affect.update_manas(peak_active)

                    bhaya_val  = affect.bhaya.item()
                    buddhi_val = affect.buddhi_value()

                    if cfg['use_viveka']:
                        viveka_gain = viveka.compute_gain(
                            active_fc1, affect.viveka_signal(), tasks_seen)
                        viveka.update(active_fc1)
                    else:
                        viveka_gain = torch.ones(fc1_shape, device=device)

                    pain_fc1 = active_fc1 if pain else torch.zeros(
                        fc1_shape, dtype=torch.bool, device=device)
                    if pain:
                        lability_fc1.inject_pain(active_fc1)
                    lability_fc1.decay()

                    if cfg['use_chitta']:
                        chitta.update(active_fc1)

                    manas_viveka_gain = viveka_gain.clone()
                    if cfg['use_manas_gane'] and manas_gane_mask.any():
                        manas_viveka_gain[manas_gane_mask] *= 2.0

                    vairagya_fc1.accumulate(
                        active_fc1, pain_fc1,
                        bhaya=bhaya_val, buddhi=buddhi_val,
                        viveka_gain=manas_viveka_gain)
                    vairagya_fc1.apply_decay(model.fc1.fc.weight.data)

                    logit_mag   = logits.detach().abs().mean(dim=0)
                    active_fout = (
                        logit_mag.unsqueeze(1).expand(fout_shape) > logit_mag.mean()
                    )
                    active_fout = active_fout & seen_mask.unsqueeze(1)
                    pain_fout   = active_fout if pain else torch.zeros(
                        fout_shape, dtype=torch.bool, device=device)
                    vairagya_fout.accumulate(
                        active_fout, pain_fout,
                        bhaya=bhaya_val, buddhi=buddhi_val)

                    if cfg['use_manas_gane'] and peak_active is not None:
                        peak_2d_bool   = peak_active.unsqueeze(1).expand(fc1_shape)
                        manas_peak_fc1 = active_fc1 & peak_2d_bool
                        manas_cons.update(manas_peak_fc1)

                manas_peak_fraction = float(peak_active.float().mean().item()) if peak_active is not None else 0.0

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict(),
                    samskara_mean=chitta.mean_samskara() if cfg['use_chitta'] else 0.0,
                    moha_fraction=chitta.moha_fraction() if cfg['use_chitta'] else 0.0,
                    retrograde_fired=bool(retrograde_fired),
                    manas_peak_fraction=manas_peak_fraction,
                    karma_mean=karma.karma_mean(),
                    pruned_fraction=karma.pruned_fraction(),
                    shunyata_events=karma.total_pruned(),
                    prana_value=prana.value() if cfg['use_prana'] else 1.0,
                    effective_lr=eff_lr,
                )

            with torch.no_grad():
                for buf_imgs, buf_lbls in train_loader:
                    replay_buffer.update(buf_imgs, buf_lbls)
                    break

            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {affect.bhaya.item():.3f} | "
                  f"Buddhi: {affect.buddhi_value():.3f} | "
                  f"Prana: {prana.value():.4f} | "
                  f"EffLR: {eff_lr:.6f} | "
                  f"Karma: {karma.karma_mean():.4f} | "
                  f"Pruned: {karma.pruned_fraction()*100:.2f}%")

        print(f"  Evaluating after Task {task_id} [CIL]...")
        acc_dict = {}
        for t in range(NUM_TASKS):
            acc = evaluate_task(
                model, test_loaders[t], device, encoder, T_STEPS,
                task_classes=None)
            metrics.update(trained_up_to=task_id, task_id=t, accuracy=acc)
            acc_dict[f"task_{t}"] = round(acc * 100, 2)
            print(f"    Task {t}: {acc*100:.2f}%")

        logger.log_task_summary(
            task_id, acc_dict, metrics.summary(),
            karma.summary(),
            {"mean": prana.mean_history(), "min": prana.min_history()})

    metrics.print_matrix()
    final = metrics.summary()
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name} | seed={seed}")
    print(f"  AA  : {final['AA']}%")
    print(f"  BWT : {final['BWT']}%")
    print(f"  FWT : {final['FWT']}%")
    print(f"  Total pruned: {karma.total_pruned()} synapses "
          f"({karma.pruned_fraction()*100:.2f}%)")
    print(f"  Prana final: {prana.value():.4f}")
    print(f"{'='*60}")
    logger.log_final(final)
    logger.close()
    return final


if __name__ == "__main__":
    results = {}
    for cond in CONDITIONS:
        results[cond] = run_condition(cond, seed=42)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY -- Maya-Prana Paper 9")
    print(f"{'='*60}")
    for cond, r in results.items():
        print(f"  {cond:30s} AA={r['AA']}% BWT={r['BWT']}%")
    print(f"{'='*60}")