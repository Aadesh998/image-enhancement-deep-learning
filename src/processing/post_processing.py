import cv2
import numpy as np
import torch


def apply_post_processing(image_tensor, gamma=1.2, color_space="RGB"):
    device = image_tensor.device

    if color_space == "YCbCr":
        from utils import ycbcr_to_rgb

        image_tensor = ycbcr_to_rgb(image_tensor)

    img_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    processed_batch = []

    for i in range(img_np.shape[0]):
        img_bgr = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR)
        denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        gaussian_blur = cv2.GaussianBlur(contrast, (0, 0), 2.0)
        sharpened = cv2.addWeighted(contrast, 1.5, gaussian_blur, -0.5, 0)

        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        gamma_corrected = cv2.LUT(sharpened, table)

        result_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
        processed_batch.append(result_rgb)

    processed_np = np.stack(processed_batch, axis=0)
    processed_tensor = (
        torch.from_numpy(processed_np).permute(0, 3, 1, 2).float() / 255.0
    )

    return processed_tensor.to(device)
