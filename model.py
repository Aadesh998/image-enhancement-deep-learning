import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rgb_to_ycbcr, ycbcr_to_rgb


class DCENet(nn.Module):
    def __init__(
        self, in_channels=3, num_filters=32, kernel_size=3, stride=1, padding=1
    ):
        super(DCENet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size, stride, padding)

        self.conv5 = nn.Conv2d(
            num_filters * 2, num_filters, kernel_size, stride, padding
        )
        self.conv6 = nn.Conv2d(
            num_filters * 2, num_filters, kernel_size, stride, padding
        )
        self.conv7 = nn.Conv2d(num_filters * 2, 24, kernel_size, stride, padding)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))

        int_con1 = torch.cat([conv4, conv3], dim=1)
        conv5 = F.relu(self.conv5(int_con1))

        int_con2 = torch.cat([conv5, conv2], dim=1)
        conv6 = F.relu(self.conv6(int_con2))

        int_con3 = torch.cat([conv6, conv1], dim=1)
        out = torch.tanh(self.conv7(int_con3))

        return out


# Loss Functions
def color_constancy_loss(x, color_space="RGB"):
    if color_space == "YCbCr":
        mean_uv = x[:, 1:, :, :].mean(dim=(2, 3))
        loss = torch.mean((mean_uv - 0.5) ** 2)
        return loss

    mean_rgb = x.mean(dim=(2, 3), keepdim=True)
    mean_r, mean_g, mean_b = (
        mean_rgb[:, 0, :, :],
        mean_rgb[:, 1, :, :],
        mean_rgb[:, 2, :, :],
    )
    diff_rg = (mean_r - mean_g) ** 2
    diff_rb = (mean_r - mean_b) ** 2
    diff_gb = (mean_g - mean_b) ** 2
    loss = torch.sqrt(diff_rg + diff_rb + diff_gb)
    return loss.mean()


def exposure_loss(
    x, mean_val=0.58, color_space="YCbCr"
):  # lower mean_val slightly for moon dark bg
    if color_space == "YCbCr":
        # Use Y channel only
        gray = x[:, 0:1, :, :]
    else:
        gray = x.mean(dim=1, keepdim=True)

    mean = F.avg_pool2d(gray, kernel_size=16, stride=16)
    return ((mean - mean_val) ** 2).mean()


def illumination_smoothness_loss(r):  # r is B×24×H×W
    # Average the 24 channels → illumination-like map
    illum_map = r.mean(dim=1, keepdim=True)  # B×1×H×W

    h_tv = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :]).sum()
    w_tv = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1]).sum()

    count_h = (illum_map.shape[2] - 1) * illum_map.shape[3]
    count_w = illum_map.shape[2] * (illum_map.shape[3] - 1)
    return 2 * (h_tv / count_h + w_tv / count_w) / illum_map.shape[0]


class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        self.left_kernel = torch.tensor(
            [[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float32
        )
        self.right_kernel = torch.tensor(
            [[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]], dtype=torch.float32
        )
        self.up_kernel = torch.tensor(
            [[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32
        )
        self.down_kernel = torch.tensor(
            [[[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]], dtype=torch.float32
        )

    def forward(self, enhanced, input_img, color_space="YCbCr"):
        if color_space == "YCbCr":
            enhanced_gray = enhanced[:, 0:1, :, :]
            input_gray = input_img[:, 0:1, :, :]
        elif enhanced.shape[1] == 3:
            enhanced_gray = (
                0.299 * enhanced[:, 0:1, :, :]
                + 0.587 * enhanced[:, 1:2, :, :]
                + 0.114 * enhanced[:, 2:3, :, :]
            )
            input_gray = (
                0.299 * input_img[:, 0:1, :, :]
                + 0.587 * input_img[:, 1:2, :, :]
                + 0.114 * input_img[:, 2:3, :, :]
            )
        else:
            enhanced_gray = (
                0.299 * enhanced[:, 0] + 0.587 * enhanced[:, 1] + 0.114 * enhanced[:, 2]
            )
            input_gray = (
                0.299 * input_img[:, 0]
                + 0.587 * input_img[:, 1]
                + 0.114 * input_img[:, 2]
            )
            enhanced_gray = enhanced_gray.unsqueeze(1)
            input_gray = input_gray.unsqueeze(1)

        y_true_pool = F.avg_pool2d(input_gray, kernel_size=4, stride=4)
        y_pred_pool = F.avg_pool2d(enhanced_gray, kernel_size=4, stride=4)

        device = input_img.device
        kernels = [
            self.left_kernel,
            self.right_kernel,
            self.up_kernel,
            self.down_kernel,
        ]
        kernels = [k.to(device) for k in kernels]

        d_orig = [F.conv2d(y_true_pool, k, padding=1) for k in kernels]
        d_pred = [F.conv2d(y_pred_pool, k, padding=1) for k in kernels]

        diffs = [(a - b).pow(2) for a, b in zip(d_orig, d_pred)]
        return sum([d.mean() for d in diffs])


class ZeroDCE(nn.Module):
    def __init__(self, cfg):
        super(ZeroDCE, self).__init__()
        self.color_space = getattr(cfg, "color_space", "RGB")
        self.dce_model = DCENet(
            in_channels=cfg.network.in_channels,
            num_filters=cfg.network.num_filters,
            kernel_size=cfg.network.kernel_size,
            stride=cfg.network.stride,
            padding=cfg.network.padding,
        )
        self.spatial_loss_fn = SpatialConsistencyLoss()
        self.weights = cfg.loss_weights

    def get_enhanced_image_ycbcr(
        self, x_rgb: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:

        x_ycbcr = rgb_to_ycbcr(x_rgb)
        y = x_ycbcr[:, 0:1, :, :]

        enhanced_y = y.clone()
        for i in range(0, 24, 3):
            r_i = r[:, i : i + 3, :, :]
            alpha = r_i.mean(dim=1, keepdim=True)
            enhanced_y = enhanced_y + alpha * (enhanced_y.pow(2) - enhanced_y)

        enhanced_y = torch.clamp(enhanced_y, 0.0, 1.0)

        enhanced_ycbcr = torch.cat([enhanced_y, x_ycbcr[:, 1:]], dim=1)
        enhanced_rgb = ycbcr_to_rgb(enhanced_ycbcr)
        return enhanced_rgb

    def chroma_preserve_loss(self, enhanced_ycbcr, input_ycbcr):
        diff = (enhanced_ycbcr[:, 1:] - input_ycbcr[:, 1:]).abs().mean()
        return diff

    def forward(self, x):
        r = self.dce_model(x)
        enhanced = self.get_enhanced_image_ycbcr(x, r)
        return enhanced, r

    def compute_losses(self, x, r, enhanced):
        x_ycbcr = rgb_to_ycbcr(x)
        enhanced_ycbcr = rgb_to_ycbcr(enhanced)

        loss_illum = (
            self.weights.illumination_smoothness * illumination_smoothness_loss(r)
        )
        loss_spatial = self.weights.spatial_constancy * self.spatial_loss_fn(
            enhanced, x, "YCbCr"
        )
        loss_color = self.weights.color_constancy * color_constancy_loss(
            enhanced, "YCbCr"
        )
        loss_expo = self.weights.exposure * exposure_loss(
            enhanced_ycbcr, self.weights.exposure_mean_val, "YCbCr"
        )

        loss_chroma = 3.0 * self.chroma_preserve_loss(enhanced_ycbcr, x_ycbcr)  # new!

        total = loss_illum + loss_spatial + loss_color + loss_expo + loss_chroma

        return {
            "total_loss": total,
            "loss_illum": loss_illum,
            "loss_spatial": loss_spatial,
            "loss_color": loss_color,
            "loss_expo": loss_expo,
            "loss_chroma": loss_chroma,
        }
