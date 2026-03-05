import torch
import torch.nn.functional as F


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr[:, 0], ycbcr[:, 1] - 0.5, ycbcr[:, 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.clamp(torch.stack([r, g, b], dim=1), 0.0, 1.0)
