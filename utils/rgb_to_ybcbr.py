import torch
import torch.nn.functional as F


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """
    RGB [0,1] → YCbCr [0,1]
    Standard ITU-R BT.601 conversion -> Standard convsersion metrics
    """
    y = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    cb = -0.168736 * img[:, 0] - 0.331264 * img[:, 1] + 0.500 * img[:, 2] + 0.5
    cr = 0.500 * img[:, 0] - 0.418688 * img[:, 1] - 0.081312 * img[:, 2] + 0.5
    return torch.stack([y, cb, cr], dim=1)
