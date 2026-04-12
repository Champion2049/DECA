import copy

import torch
import torch.nn as nn


class ResidualCodeHead(nn.Module):
    """Tiny residual corrector for DECA coarse codes.

    Input: concat(exp[50], jaw_pose[3], cam[3])
    Output: residuals for same components (56-dim)
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(56, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 56),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_head_input(codedict: dict) -> torch.Tensor:
    exp = codedict["exp"]
    jaw = codedict["pose"][:, 3:6]
    cam = codedict["cam"]
    return torch.cat([exp, jaw, cam], dim=1)


def apply_residual_correction(codedict: dict, residual: torch.Tensor) -> dict:
    """Apply bounded residuals to DECA code dictionary.

    The tanh scaling keeps corrections conservative and stable.
    """

    out = copy.copy(codedict)
    out["exp"] = codedict["exp"] + 0.10 * torch.tanh(residual[:, :50])
    pose = codedict["pose"].clone()
    pose[:, 3:6] = pose[:, 3:6] + 0.05 * torch.tanh(residual[:, 50:53])
    out["pose"] = pose
    out["cam"] = codedict["cam"] + 0.05 * torch.tanh(residual[:, 53:56])
    return out
