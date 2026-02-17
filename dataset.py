import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MoonFrameDataset(Dataset):
    def __init__(self, file_paths, image_size=256, transform=None, color_space="RGB"):
        self.file_paths = file_paths
        self.transform = transform
        self.color_space = color_space
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        low_image_path = self.file_paths[idx]
        # Construct high image path: replace 'low' with 'high' in the path
        high_image_path = low_image_path.replace("low", "high")
        
        try:
            low_image = Image.open(low_image_path).convert("RGB")
            high_image = Image.open(high_image_path).convert("RGB")

            if self.color_space == "YUV":
                low_image = low_image.convert("YCbCr")
                high_image = high_image.convert("YCbCr")
                
        except Exception as e:
            print(f"Error loading image pair {low_image_path}: {e}")
            low_image = Image.new("RGB" if self.color_space == "RGB" else "YCbCr", (256, 256))
            high_image = Image.new("RGB" if self.color_space == "RGB" else "YCbCr", (256, 256))
            
        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)
        else:
            low_image = self.resize(low_image)
            low_image = self.to_tensor(low_image)
            high_image = self.resize(high_image)
            high_image = self.to_tensor(high_image)
            
        return low_image, high_image

def get_dataloaders(cfg):
    all_files = sorted(glob.glob(cfg.data.train_path))
    
    if not all_files:
        print(f"Warning: No files found at {cfg.data.train_path}")
        return None, None, None
        
    total_files = len(all_files)
    train_size = int(total_files * cfg.data.train_split)
    val_size = int(total_files * cfg.data.validation_split)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size+val_size]
    test_files = all_files[train_size+val_size:]
    
    print(f"Found {total_files} images. Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")
    
    train_dataset = MoonFrameDataset(train_files, image_size=cfg.data.image_size, color_space=cfg.data.color_space)
    val_dataset = MoonFrameDataset(val_files, image_size=cfg.data.image_size, color_space=cfg.data.color_space)
    test_dataset = MoonFrameDataset(test_files, image_size=cfg.data.image_size, color_space=cfg.data.color_space)
    
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
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
