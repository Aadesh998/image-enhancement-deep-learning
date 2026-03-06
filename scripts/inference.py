import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

# Ensure src/ is in sys.path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from models import ZeroDCE
from processing import apply_post_processing
from utils import ycbcr_to_rgb


def load_image(image_path, color_space="YCbCr", image_size=256):
    """Loads an image, converts to the specified color space, resizes, and converts to tensor."""
    img = Image.open(image_path).convert("RGB")
    if color_space == "YCbCr":
        img = img.convert("YCbCr")

    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    # Add batch dimension: (1, C, H, W)
    return transform(img).unsqueeze(0)


def save_image(tensor, path, color_space="YCbCr"):
    if color_space == "YCbCr":
        tensor = ycbcr_to_rgb(tensor)

    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def main():
    parser = argparse.ArgumentParser(description="Test Zero-DCE model on moon images")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to a single image or a directory of images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the output images",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "src", "config", "config.yaml"
        ),
        help="Path to the config.yaml file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        return
    cfg = OmegaConf.load(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    model = ZeroDCE(cfg.model).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = glob.glob(os.path.join(args.input, "*.*"))
        valid_exts = (".png", ".jpg", ".jpeg")
        image_paths = [p for p in image_paths if p.lower().endswith(valid_exts)]

    if not image_paths:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(image_paths)} images. Starting inference...\n")
    color_space = cfg.data.color_space

    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)

            print(f"Processing: {filename}")

            input_tensor = load_image(
                img_path, color_space=color_space, image_size=cfg.data.image_size
            ).to(device)

            enhanced_tensor, _ = model(input_tensor)

            enhanced_pp_tensor = apply_post_processing(
                enhanced_tensor.clone(), color_space=color_space
            )

            out_input_path = os.path.join(args.output_dir, f"{name}_01_original{ext}")
            out_enhanced_path = os.path.join(
                args.output_dir, f"{name}_02_enhanced{ext}"
            )
            out_pp_path = os.path.join(args.output_dir, f"{name}_03_postprocessed{ext}")

            save_image(input_tensor, out_input_path, color_space=color_space)
            save_image(enhanced_tensor, out_enhanced_path, color_space=color_space)

            save_image(enhanced_pp_tensor, out_pp_path, color_space="RGB")

    print(f"\nDone! All results have been saved to the '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()

# python 