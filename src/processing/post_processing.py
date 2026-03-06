import cv2
import numpy as np
import torch

def apply_post_processing(image_tensor):
    device = image_tensor.device
    
    img_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    processed_batch = []
    
    for i in range(img_np.shape[0]):
        img = img_np[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(contrast, -1, kernel)
        
        result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        processed_batch.append(result_rgb)
    
    processed_np = np.stack(processed_batch, axis=0)
    processed_tensor = torch.from_numpy(processed_np).permute(0, 3, 1, 2).float() / 255.0
    
    return processed_tensor.to(device)
