import torch
import torch.nn as nn
import torch.nn.functional as F

class DCENet(nn.Module):
    def __init__(self, in_channels=3, num_filters=32, kernel_size=3, stride=1, padding=1):
        super(DCENet, self).__init__()
        
        self.conv1 = self.dwise_conv(in_channels, num_filters)
        self.conv2 = self.dwise_conv(num_filters, num_filters)
        self.conv3 = self.dwise_conv(num_filters, num_filters)
        self.conv4 = self.dwise_conv(num_filters, num_filters)
        
        # After concat(conv4, conv3) -> 64 channels
        self.conv5 = self.dwise_conv(num_filters * 2, num_filters)
        
        # After concat(conv5, conv2) -> 64 channels
        self.conv6 = self.dwise_conv(num_filters * 2, num_filters)
        
        # After concat(conv6, conv1) -> 64 channels
        self.conv7 = self.dwise_conv(num_filters * 2, 24)

    def dwise_conv(self, in_channels, num_filters, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size, stride, padding, groups=in_channels),
            nn.Conv2d(num_filters, num_filters, kernel_size, stride, padding, groups=in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        int_con1 = torch.cat([conv4, conv3], dim=1)
        conv5 = self.conv5(int_con1)

        int_con2 = torch.cat([conv5, conv2], dim=1)
        conv6 = self.conv6(int_con2)

        int_con3 = torch.cat([conv6, conv1], dim=1)
        out = torch.tanh(self.conv7(int_con3))

        return out

# Loss Functions
def color_constancy_loss(x, color_space="RGB"):
    if color_space == "YUV":
        # In YUV, we want U and V to be close to 0.5 (neutral)
        mean_uv = x[:, 1:, :, :].mean(dim=(2, 3))
        loss = torch.mean((mean_uv - 0.5) ** 2)
        return loss
        
    mean_rgb = x.mean(dim=(2, 3), keepdim=True)
    mean_r, mean_g, mean_b = mean_rgb[:, 0, :, :], mean_rgb[:, 1, :, :], mean_rgb[:, 2, :, :]
    diff_rg = (mean_r - mean_g) ** 2
    diff_rb = (mean_r - mean_b) ** 2
    diff_gb = (mean_g - mean_b) ** 2
    loss = torch.sqrt(diff_rg + diff_rb + diff_gb)
    return loss.mean()

def exposure_loss(x, mean_val=0.6, color_space="RGB"):
    if color_space == "YUV":
        # In YUV, exposure is mainly about the Y channel
        x = x[:, 0:1, :, :]
    else:
        x = x.mean(dim=1, keepdim=True)
        
    mean = F.avg_pool2d(x, kernel_size=16, stride=16, padding=0)
    return ((mean - mean_val) ** 2).mean()

def illumination_smoothness_loss(x):
    batch_size = x.shape[0]
    h_x = x.shape[2]
    w_x = x.shape[3]
    count_h = (x.shape[2] - 1) * x.shape[3]
    count_w = x.shape[2] * (x.shape[3] - 1)
    
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
    
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        self.left_kernel = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
        self.right_kernel = torch.tensor([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]], dtype=torch.float32)
        self.up_kernel = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
        self.down_kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]], dtype=torch.float32)

    def forward(self, y_true, y_pred, color_space="RGB"):
        if color_space == "YUV":
            y_true_gray = y_true[:, 0:1, :, :]
            y_pred_gray = y_pred[:, 0:1, :, :]
        elif y_true.shape[1] == 3:
            y_true_gray = 0.299 * y_true[:, 0:1, :, :] + 0.587 * y_true[:, 1:2, :, :] + 0.114 * y_true[:, 2:3, :, :]
            y_pred_gray = 0.299 * y_pred[:, 0:1, :, :] + 0.587 * y_pred[:, 1:2, :, :] + 0.114 * y_pred[:, 2:3, :, :]
        else:
            y_true_gray = y_true
            y_pred_gray = y_pred
            
        y_true_pool = F.avg_pool2d(y_true_gray, kernel_size=4, stride=4)
        y_pred_pool = F.avg_pool2d(y_pred_gray, kernel_size=4, stride=4)
        
        device = y_true.device
        kernels = [self.left_kernel, self.right_kernel, self.up_kernel, self.down_kernel]
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
            padding=cfg.network.padding
        )
        self.spatial_loss_fn = SpatialConsistencyLoss()
        self.weights = cfg.loss_weights

    def get_enhanced_image(self, x, r):
        for i in range(0, 24, 3):
            r_i = r[:, i:i+3, :, :]
            x = x + r_i * (torch.pow(x, 2) - x)
        return x

    def forward(self, x):
        r = self.dce_model(x)
        enhanced = self.get_enhanced_image(x, r)
        return enhanced, r

    def compute_losses(self, x, r, enhanced):
        loss_illumination = self.weights.illumination_smoothness * illumination_smoothness_loss(r)
        loss_spatial = self.weights.spatial_constancy * self.spatial_loss_fn(enhanced, x, self.color_space)
        loss_color = self.weights.color_constancy * color_constancy_loss(enhanced, self.color_space)
        loss_exposure = self.weights.exposure * exposure_loss(enhanced, self.weights.exposure_mean_val, self.color_space)
        
        total_loss = loss_illumination + loss_spatial + loss_color + loss_exposure
        return {
            'total_loss': total_loss,
            'loss_illumination': loss_illumination,
            'loss_spatial': loss_spatial,
            'loss_color': loss_color,
            'loss_exposure': loss_exposure
        }
