from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class StructuralDefectSampler:
    """
    Graphene structural defect sampler.

    Returns a defect metadata dictionary with canonical tensors:
      - vacancy_mask: (2, H, W) bool, sublattice [A, B]
      - onsite_ab_delta: (2, H, W) float32
      - bond_mod: (3, H, W) float32

    Bond directions are defined from A(i,j) to B neighbors:
      k=0 -> B(i, j)
      k=1 -> B(i-1, j)
      k=2 -> B(i, j-1)
    """

    def __init__(self, config: Optional[Dict[str, Any]]):
        self.cfg = config or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.family = str(self.cfg.get("family", "mixed"))
        self.mixed_cfg = self.cfg.get("mixed", {})
        self.vacancy_cfg = self.cfg.get("vacancy", {})
        self.bond_cfg = self.cfg.get("bond_disorder", {})
        self.sub_cfg = self.cfg.get("sublattice_selective", {})

    def sample_graphene(self, H: int, W: int, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))

        vacancy_mask = torch.zeros((2, H, W), dtype=torch.bool)
        onsite_ab_delta = torch.zeros((2, H, W), dtype=torch.float32)
        bond_mod = torch.zeros((3, H, W), dtype=torch.float32)

        if not self.enabled:
            return {
                "vacancy_mask": vacancy_mask,
                "onsite_ab_delta": onsite_ab_delta,
                "bond_mod": bond_mod,
            }

        if self.family == "mixed":
            weights_cfg = dict(self.mixed_cfg.get("weights", {}))
            weights = {
                "vacancy": float(weights_cfg.get("vacancy", 0.0)),
                "bond_disorder": float(weights_cfg.get("bond_disorder", 0.0)),
                "sublattice_selective": float(weights_cfg.get("sublattice_selective", 0.0)),
            }
            if weights["vacancy"] > 0:
                vacancy_mask |= self._sample_vacancy(H, W, gen)
            if weights["bond_disorder"] > 0:
                bond_mod += self._sample_bond_disorder(H, W, gen)
            if weights["sublattice_selective"] > 0:
                onsite_ab_delta += self._sample_sublattice_selective(H, W, gen)
        elif self.family == "vacancy":
            vacancy_mask |= self._sample_vacancy(H, W, gen)
        elif self.family == "bond_disorder":
            bond_mod += self._sample_bond_disorder(H, W, gen)
        elif self.family == "sublattice_selective":
            onsite_ab_delta += self._sample_sublattice_selective(H, W, gen)
        else:
            raise ValueError(f"Unknown structural defect family: {self.family}")

        return {
            "vacancy_mask": vacancy_mask,
            "onsite_ab_delta": onsite_ab_delta.to(torch.float32),
            "bond_mod": bond_mod.to(torch.float32),
        }

    def _sample_vacancy(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        c_min, c_max = self.vacancy_cfg.get("concentration_range", [0.0, 0.0])
        concentration = float(c_min) + (float(c_max) - float(c_min)) * float(torch.rand(1, generator=gen).item())
        concentration = max(0.0, min(1.0, concentration))
        p_ab = float(self.vacancy_cfg.get("ab_balance", 0.5))
        p_ab = max(0.0, min(1.0, p_ab))

        n_sites = 2 * H * W
        n_vac = int(round(concentration * n_sites))
        if n_vac <= 0:
            return torch.zeros((2, H, W), dtype=torch.bool)

        mask = torch.zeros((2, H, W), dtype=torch.bool)
        flat_idx = torch.randperm(n_sites, generator=gen)[:n_vac]
        sub_idx = flat_idx // (H * W)
        rem = flat_idx % (H * W)
        xs = rem // W
        ys = rem % W
        mask[sub_idx.long(), xs.long(), ys.long()] = True

        if self.vacancy_cfg.get("sublattice_bias", False):
            # Optional bias redraw preserving total vacancy count.
            target_a = int(round(p_ab * n_vac))
            target_b = n_vac - target_a
            mask = torch.zeros((2, H, W), dtype=torch.bool)
            a_idx = torch.randperm(H * W, generator=gen)[: max(0, min(H * W, target_a))]
            b_idx = torch.randperm(H * W, generator=gen)[: max(0, min(H * W, target_b))]
            ax, ay = (a_idx // W).long(), (a_idx % W).long()
            bx, by = (b_idx // W).long(), (b_idx % W).long()
            mask[0, ax, ay] = True
            mask[1, bx, by] = True
        return mask

    def _sample_bond_disorder(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        d_min, d_max = self.bond_cfg.get("delta_range", [0.0, 0.0])
        missing_prob = float(self.bond_cfg.get("missing_bond_prob", 0.0))
        p_apply = float(self.bond_cfg.get("apply_prob", 1.0))
        if float(torch.rand(1, generator=gen).item()) > p_apply:
            return torch.zeros((3, H, W), dtype=torch.float32)

        delta = float(d_min) + (float(d_max) - float(d_min)) * float(torch.rand(1, generator=gen).item())
        # Symmetric zero-mean bond fluctuation; scale by sampled amplitude.
        mod = (2.0 * torch.rand((3, H, W), generator=gen) - 1.0) * delta
        if missing_prob > 0.0:
            missing = torch.rand((3, H, W), generator=gen) < missing_prob
            mod[missing] = -1.0
        return mod.to(torch.float32)

    def _sample_sublattice_selective(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        amp_min, amp_max = self.sub_cfg.get("amplitude_range", [0.0, 0.0])
        mode = str(self.sub_cfg.get("mode", "ab_opposite"))
        amp = float(amp_min) + (float(amp_max) - float(amp_min)) * float(torch.rand(1, generator=gen).item())
        sign = -1.0 if float(torch.rand(1, generator=gen).item()) < 0.5 else 1.0
        amp = amp * sign
        out = torch.zeros((2, H, W), dtype=torch.float32)
        if mode == "ab_opposite":
            out[0].fill_(amp)
            out[1].fill_(-amp)
        elif mode == "a_only":
            out[0].fill_(amp)
        elif mode == "b_only":
            out[1].fill_(amp)
        else:
            raise ValueError(f"Unknown sublattice_selective.mode: {mode}")
        return out

