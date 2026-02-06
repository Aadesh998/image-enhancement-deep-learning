import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MoonFrameDataset(Dataset):
    def __init__(self, file_paths, image_size=256, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (256, 256))
            
        if self.transform:
            image = self.transform(image)
        else:
            image = self.resize(image)
            image = self.to_tensor(image)
            
        return image

def get_dataloaders(cfg):
    all_files = sorted(glob.glob(cfg.data.train_path))
    
    if not all_files:
        print(f"Warning: No files found at {cfg.data.train_path}")
        return None, None
        
    total_files = len(all_files)
    train_size = int(total_files * cfg.data.train_split)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    print(f"Found {total_files} images. Split: {len(train_files)} train, {len(val_files)} val.")
    
    train_dataset = MoonFrameDataset(train_files, image_size=cfg.data.image_size)
    val_dataset = MoonFrameDataset(val_files, image_size=cfg.data.image_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
