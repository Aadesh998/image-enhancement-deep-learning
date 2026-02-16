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
    wandb.login(api_key=cfg.wandb.api_key)
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
    
    train_loader, val_loader = get_dataloaders(cfg)
    if train_loader is None:
        log.error("Failed to load data. Exiting.")
        return

    model = ZeroDCE(cfg.model).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    log.info("Starting training...")
    
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss_total = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            enhanced, r = model(images)
            losses = model.compute_losses(images, r, enhanced)
            
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
        val_images_log = []
        
        with torch.no_grad():
            for batch_idx, images in enumerate(val_loader):
                images = images.to(device)
                enhanced, r = model(images)
                losses = model.compute_losses(images, r, enhanced)
                val_loss_total += losses['total_loss'].item()
                
                if batch_idx == 0:
                    n_images = min(images.size(0), 4)
                    log_images = images[:n_images]
                    log_enhanced = enhanced[:n_images]
                    
                    if cfg.data.color_space == "YUV":
                        log_images = yuv_to_rgb(log_images)
                        log_enhanced = yuv_to_rgb(log_enhanced)
                        
                    orig_grid = make_grid(log_images, nrow=n_images)
                    enh_grid = make_grid(log_enhanced, nrow=n_images)
                    val_images_log = [wandb.Image(orig_grid, caption="Original"), wandb.Image(enh_grid, caption="Enhanced")]

        avg_val_loss = val_loss_total / len(val_loader)
        
        log.info(f"Epoch {epoch+1}/{cfg.training.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/examples": val_images_log
        })
        
        if cfg.wandb.save_model and (epoch + 1) % 10 == 0:
            os.makedirs(cfg.training.save_dir, exist_ok=True)
            save_path = os.path.join(cfg.training.save_dir, f"zerodce_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

    log.info("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    train()
