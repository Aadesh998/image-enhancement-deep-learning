import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import wandb
import os
import logging
from model import ZeroDCE
from dataset import get_dataloaders
from torchvision.utils import make_grid
from metrics import calculate_ssim, calculate_psnr
from post_processing import apply_post_processing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

log = logging.getLogger(__name__)

def yuv_to_rgb(yuv_tensor):
    # Assume yuv_tensor is (B, 3, H, W) and in [0, 1]
    # Y, Cb, Cr = yuv_tensor[:, 0, :, :], yuv_tensor[:, 1, :, :], yuv_tensor[:, 2, :, :]
    
    y = yuv_tensor[:, 0:1, :, :]
    cb = yuv_tensor[:, 1:2, :, :]
    cr = yuv_tensor[:, 2:3, :, :]
    
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    
    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, 0, 1)

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    wandb.login(key="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")
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes
    )
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    if train_loader is None:
        log.error("Failed to load data. Exiting.")
        return

    model = ZeroDCE(cfg.model).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    log.info("Starting training...")
    
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss_total = 0.0
        
        for batch_idx, (low_images, high_images) in enumerate(train_loader):
            low_images = low_images.to(device)
            # high_images = high_images.to(device) # Not used in training loss currently
            
            optimizer.zero_grad()
            enhanced, r = model(low_images)
            losses = model.compute_losses(low_images, r, enhanced)
            
            losses['total_loss'].backward()
            optimizer.step()
            
            train_loss_total += losses['total_loss'].item()
            
            if batch_idx % cfg.wandb.log_freq == 0:
                wandb.log({
                    "batch/train_loss": losses['total_loss'].item(),
                    "batch/illumination": losses['loss_illumination'].item(),
                    "batch/spatial": losses['loss_spatial'].item(),
                    "batch/color": losses['loss_color'].item(),
                    "batch/exposure": losses['loss_exposure'].item()
                })
        
        avg_train_loss = train_loss_total / len(train_loader)
        
        model.eval()
        val_loss_total = 0.0
        val_ssim_total = 0.0
        val_psnr_total = 0.0
        val_ssim_pp_total = 0.0
        val_psnr_pp_total = 0.0
        val_images_log = []
        
        with torch.no_grad():
            for batch_idx, (low_images, high_images) in enumerate(val_loader):
                low_images = low_images.to(device)
                high_images = high_images.to(device)
                
                enhanced, r = model(low_images)
                losses = model.compute_losses(low_images, r, enhanced)
                val_loss_total += losses['total_loss'].item()
                
                # Apply Post-Processing
                enhanced_pp = apply_post_processing(enhanced.clone())
                
                # Calculate Metrics against High Quality Reference
                if cfg.metrics.calculate_ssim:
                    val_ssim_total += calculate_ssim(enhanced, high_images).item()
                    val_ssim_pp_total += calculate_ssim(enhanced_pp, high_images).item()
                if cfg.metrics.calculate_psnr:
                    val_psnr_total += calculate_psnr(enhanced, high_images).item()
                    val_psnr_pp_total += calculate_psnr(enhanced_pp, high_images).item()

                if batch_idx == 0:
                    n_images = min(low_images.size(0), 4)
                    log_low = low_images[:n_images]
                    log_high = high_images[:n_images]
                    log_enhanced = enhanced[:n_images]
                    log_enhanced_pp = enhanced_pp[:n_images]
                    
                    if cfg.data.color_space == "YUV":
                        log_low = yuv_to_rgb(log_low)
                        log_high = yuv_to_rgb(log_high)
                        log_enhanced = yuv_to_rgb(log_enhanced)
                        log_enhanced_pp = yuv_to_rgb(log_enhanced_pp)
                        
                    orig_grid = make_grid(log_low, nrow=n_images)
                    high_grid = make_grid(log_high, nrow=n_images)
                    enh_grid = make_grid(log_enhanced, nrow=n_images)
                    enh_pp_grid = make_grid(log_enhanced_pp, nrow=n_images)
                    
                    val_images_log = [
                        wandb.Image(orig_grid, caption="Low Light (Input)"), 
                        wandb.Image(high_grid, caption="Normal Light (Reference)"),
                        wandb.Image(enh_grid, caption="Enhanced (Output)"),
                        wandb.Image(enh_pp_grid, caption="Enhanced + Post-Processed")
                    ]

        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_ssim = val_ssim_total / len(val_loader) if cfg.metrics.calculate_ssim else 0.0
        avg_val_psnr = val_psnr_total / len(val_loader) if cfg.metrics.calculate_psnr else 0.0
        avg_val_ssim_pp = val_ssim_pp_total / len(val_loader) if cfg.metrics.calculate_ssim else 0.0
        avg_val_psnr_pp = val_psnr_pp_total / len(val_loader) if cfg.metrics.calculate_psnr else 0.0
        
        log_msg = f"Epoch {epoch+1}/{cfg.training.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        if cfg.metrics.calculate_ssim:
            log_msg += f" | Val SSIM: {avg_val_ssim:.4f} (PP: {avg_val_ssim_pp:.4f})"
        if cfg.metrics.calculate_psnr:
            log_msg += f" | Val PSNR: {avg_val_psnr:.4f} (PP: {avg_val_psnr_pp:.4f})"
        log.info(log_msg)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/ssim": avg_val_ssim,
            "val/psnr": avg_val_psnr,
            "val/ssim_pp": avg_val_ssim_pp,
            "val/psnr_pp": avg_val_psnr_pp,
            "val/examples": val_images_log
        })
        
        if cfg.wandb.save_model and (epoch + 1) % 10 == 0:
            os.makedirs(cfg.training.save_dir, exist_ok=True)
            save_path = os.path.join(cfg.training.save_dir, f"zerodce_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

    # Test Loop
    log.info("Starting testing...")
    model.eval()
    test_loss_total = 0.0
    test_ssim_total = 0.0
    test_psnr_total = 0.0
    test_ssim_pp_total = 0.0
    test_psnr_pp_total = 0.0
    
    with torch.no_grad():
        for batch_idx, (low_images, high_images) in enumerate(test_loader):
            low_images = low_images.to(device)
            high_images = high_images.to(device)
            
            enhanced, r = model(low_images)
            losses = model.compute_losses(low_images, r, enhanced)
            test_loss_total += losses['total_loss'].item()
            
            # Apply Post-Processing
            enhanced_pp = apply_post_processing(enhanced.clone())
            
            if cfg.metrics.calculate_ssim:
                test_ssim_total += calculate_ssim(enhanced, high_images).item()
                test_ssim_pp_total += calculate_ssim(enhanced_pp, high_images).item()
            if cfg.metrics.calculate_psnr:
                test_psnr_total += calculate_psnr(enhanced, high_images).item()
                test_psnr_pp_total += calculate_psnr(enhanced_pp, high_images).item()
    
    avg_test_loss = test_loss_total / len(test_loader)
    avg_test_ssim = test_ssim_total / len(test_loader) if cfg.metrics.calculate_ssim else 0.0
    avg_test_psnr = test_psnr_total / len(test_loader) if cfg.metrics.calculate_psnr else 0.0
    avg_test_ssim_pp = test_ssim_pp_total / len(test_loader) if cfg.metrics.calculate_ssim else 0.0
    avg_test_psnr_pp = test_psnr_pp_total / len(test_loader) if cfg.metrics.calculate_psnr else 0.0
    
    log.info(f"Test Results | Loss: {avg_test_loss:.4f} | SSIM: {avg_test_ssim:.4f} (PP: {avg_test_ssim_pp:.4f}) | PSNR: {avg_test_psnr:.4f} (PP: {avg_test_psnr_pp:.4f})")
    wandb.log({
        "test/loss": avg_test_loss,
        "test/ssim": avg_test_ssim,
        "test/psnr": avg_test_psnr,
        "test/ssim_pp": avg_test_ssim_pp,
        "test/psnr_pp": avg_test_psnr_pp
    })

    log.info("Training and Testing complete.")
    wandb.finish()

if __name__ == "__main__":
    train()
