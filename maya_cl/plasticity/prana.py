# prana.py -- Prana metabolic plasticity budget (Paper 9 core contribution)
# Biological ground: astrocyte-neuron lactate shuttle (ANLS)
# Astrocytes supply lactate to active neurons. Under sustained learning load,
# metabolic demand exceeds supply -- plasticity degrades. Prana is that supply.
#
# Mechanism:
#   prana -= PRANA_COST_RATE * gradient_magnitude * activity_level  (depletion)
#   prana += PRANA_RECOVERY_RATE * (1.0 - activity_level) * vairagya  (recovery)
#   prana = clip(prana, PRANA_MIN, 1.0)
#
#   effective_lr = base_lr * prana * (0.5 + buddhi * 0.5)  (Buddhi-modulated gate)
#
# Vedantic ground: Prana is the first vital force. Without it, nothing moves,
# nothing learns, nothing grows. Every other Antahkarana dimension operates
# within the budget Prana provides.
#
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import torch
from maya_cl.utils.config import (
    PRANA_INIT,
    PRANA_COST_RATE,
    PRANA_RECOVERY_RATE,
    PRANA_MIN,
    PRANA_RECOVERY_THRESHOLD,
)


class PranaMetabolic:
    """
    Scalar metabolic plasticity budget.

    Depletes when the network is learning hard (high gradient magnitude, high activity).
    Recovers when the network is at rest (low activity), modulated by Vairagya.
    Gates the optimizer learning rate each batch via effective_lr().

    This is not a weight mask and not a gradient gate.
    Prana is an external metabolic constraint on how much plasticity the system
    can sustain -- analogous to astrocytic lactate supply to active neurons.
    """

    def __init__(self, device: torch.device,
                 cost_rate: float = PRANA_COST_RATE,
                 recovery_rate: float = PRANA_RECOVERY_RATE):
        self.device        = device
        self.cost_rate     = cost_rate
        self.recovery_rate = recovery_rate
        self.prana         = PRANA_INIT
        self._history      = []           # per-batch prana values for logging

    def update(self,
               gradient_magnitude: float,
               activity_level: float,
               vairagya: float) -> None:
        """
        Called once per batch after loss.backward(), before optimizer.step().

        Args:
            gradient_magnitude: mean absolute gradient of fc1 weights this batch
            activity_level:     mean spike rate of fc1 this batch (0.0 -- 1.0)
            vairagya:           current Vairagya protection fraction (0.0 -- 1.0)
                                High Vairagya = earned detachment = faster recovery.
                                Philosophically: a mind with Vairagya recovers its
                                Prana more efficiently -- it does not waste energy
                                on attachment to outcomes.
        """
        with torch.no_grad():
            # Depletion: proportional to learning load
            depletion = self.cost_rate * gradient_magnitude * activity_level
            self.prana -= depletion

            # Recovery: fires only when activity is low (rest state)
            # Vairagya modulates recovery efficiency
            if activity_level < PRANA_RECOVERY_THRESHOLD:
                recovery = self.recovery_rate * (1.0 - activity_level) * (0.5 + vairagya * 0.5)
                self.prana += recovery

            # Clamp to [PRANA_MIN, 1.0] -- never fully starved, never over-full
            self.prana = float(max(PRANA_MIN, min(1.0, self.prana)))
            self._history.append(self.prana)

    def effective_lr(self, base_lr: float, buddhi: float) -> float:
        """
        Returns the Prana-and-Buddhi-modulated learning rate for this batch.

        effective_lr = base_lr * prana * (0.5 + buddhi * 0.5)

        Buddhi modulation mirrors the P8 pruning threshold formula:
        young Maya (low Buddhi) uses energy impulsively (0.5x base).
        mature Maya (high Buddhi) uses energy wisely (up to 1.0x base).
        At full Prana and full Buddhi: effective_lr = base_lr (no penalty).
        At PRANA_MIN and zero Buddhi: effective_lr = base_lr * 0.05 * 0.5.
        """
        return base_lr * self.prana * (0.5 + buddhi * 0.5)

    def on_task_boundary(self) -> None:
        """
        Called at each task boundary.
        Partial Prana recovery at rest -- analogous to sleep consolidation.
        Does not fully restore -- sustained load across tasks leaves a residue.
        """
        self.prana = float(min(1.0, self.prana + 0.3 * (1.0 - self.prana)))
        self._history.clear()

    def value(self) -> float:
        return self.prana

    def mean_history(self) -> float:
        if not self._history:
            return self.prana
        return float(sum(self._history) / len(self._history))

    def min_history(self) -> float:
        if not self._history:
            return self.prana
        return float(min(self._history))