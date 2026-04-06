# affective_state.py -- Maya-Prana (Paper 9)
# Extends P8 with prana tracking.
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import torch
from maya_cl.utils.config import (
    TAU_SHRADDHA, TAU_BHAYA, TAU_VAIRAGYA, TAU_SPANDA,
    TAU_VIVEKA, TAU_BUDDHI,
)


class AffectiveState:
    def __init__(self, device: torch.device):
        self.device   = device
        self.shraddha = torch.tensor(0.5,  device=device)
        self.bhaya    = torch.tensor(0.0,  device=device)
        self.vairagya = torch.tensor(0.3,  device=device)
        self.spanda   = torch.tensor(0.4,  device=device)
        self.viveka   = torch.tensor(0.5,  device=device)
        self.buddhi   = torch.tensor(0.85, device=device)
        self.chitta   = torch.tensor(0.0,  device=device)
        self.manas    = torch.tensor(0.0,  device=device)
        self.shunyata = torch.tensor(0.0,  device=device)

        # Prana -- P9 new dimension.
        # Scalar metabolic plasticity budget. Tracked here for logging/dashboard.
        # Actual depletion/recovery logic lives in PranaMetabolic (plasticity/prana.py).
        # This value is updated by the training loop via update_prana().
        self.prana = torch.tensor(1.0, device=device)

        self._experience_count = 0
        self._tau_buddhi       = TAU_BUDDHI

    def update(self, confidence: float, pain: bool,
               spike_rate: float) -> None:
        c = torch.tensor(confidence, device=self.device)
        p = torch.tensor(1.0 if pain else 0.0, device=self.device)

        alpha_s  = 1.0 / TAU_SHRADDHA
        alpha_b  = 1.0 / TAU_BHAYA
        alpha_v  = 1.0 / TAU_VAIRAGYA
        alpha_p  = 1.0 / TAU_SPANDA
        alpha_vk = 1.0 / TAU_VIVEKA

        self.shraddha = torch.clamp(
            self.shraddha * (1 - alpha_s) + c * alpha_s, 0.0, 1.0)
        self.bhaya = torch.clamp(
            self.bhaya    * (1 - alpha_b) + p * alpha_b, 0.0, 1.0)
        self.vairagya = torch.clamp(
            self.vairagya * (1 - alpha_v) + c * alpha_v, 0.0, 1.0)
        self.spanda = torch.clamp(
            self.spanda   * (1 - alpha_p) +
            torch.tensor(spike_rate, device=self.device) * alpha_p,
            0.0, 1.0)
        self.viveka = torch.clamp(
            self.viveka   * (1 - alpha_vk) + c * alpha_vk, 0.0, 1.0)

        self._experience_count += 1
        exp_factor = 1.0 - torch.exp(
            torch.tensor(-self._experience_count / self._tau_buddhi,
                         device=self.device))
        self.buddhi = torch.clamp(
            exp_factor * (1.0 - self.bhaya), 0.0, 1.0)

        self.chitta   = torch.clamp(self.chitta   * 0.995, 0.0, 1.0)
        self.manas    = torch.clamp(self.manas    * 0.995, 0.0, 1.0)
        self.shunyata = torch.clamp(self.shunyata * 0.990, 0.0, 1.0)

    def update_chitta(self, retrograde_fired: bool,
                      release_fraction: float) -> None:
        if retrograde_fired:
            self.chitta = torch.clamp(
                self.chitta + torch.tensor(
                    release_fraction * 0.1, device=self.device),
                0.0, 1.0)

    def update_manas(self, peak_active: torch.Tensor) -> None:
        if peak_active is not None:
            fraction = float(peak_active.float().mean().item())
            self.manas = torch.clamp(
                self.manas + torch.tensor(fraction * 0.05, device=self.device),
                0.0, 1.0)

    def update_shunyata(self, n_pruned: int, total_synapses: int) -> None:
        if n_pruned > 0 and total_synapses > 0:
            release_fraction = n_pruned / total_synapses
            self.shunyata = torch.clamp(
                self.shunyata + torch.tensor(
                    release_fraction * 2.0, device=self.device),
                0.0, 1.0)

    def update_prana(self, prana_value: float) -> None:
        """Called by training loop after PranaMetabolic.update() each batch."""
        self.prana = torch.tensor(
            float(prana_value), device=self.device).clamp(0.0, 1.0)

    def reset_experience(self) -> None:
        self._experience_count = 0

    def viveka_signal(self) -> float:
        return float(self.viveka.item())

    def buddhi_value(self) -> float:
        return float(self.buddhi.item())

    def chitta_value(self) -> float:
        return float(self.chitta.item())

    def manas_value(self) -> float:
        return float(self.manas.item())

    def shunyata_value(self) -> float:
        return float(self.shunyata.item())

    def prana_value(self) -> float:
        return float(self.prana.item())

    def as_dict(self) -> dict:
        return {
            "shraddha": float(self.shraddha.item()),
            "bhaya":    float(self.bhaya.item()),
            "vairagya": float(self.vairagya.item()),
            "spanda":   float(self.spanda.item()),
            "viveka":   float(self.viveka.item()),
            "buddhi":   float(self.buddhi.item()),
            "chitta":   float(self.chitta.item()),
            "manas":    float(self.manas.item()),
            "shunyata": float(self.shunyata.item()),
            "prana":    float(self.prana.item()),
        }