import sys
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import segmentation_models_pytorch as smp

import torch
import torchvision.transforms as transforms

from colorLUT import colorLUT


ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

def mask_image(og_img, mask):
    mask = np.array(mask, dtype=np.uint8)

    # convert  class mask to HSV coloring based on colorLUT colors
    hsv_mask = cv2.LUT(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), colorLUT)

    #covert to BGR
    rgb_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # overlap mask with og image
    # masked_image = np.where(rgb_mask == [0, 0, 0], og_img, rgb_mask)
    alpha = 0.5
    overlay = cv2.addWeighted(np.array(og_img), 1, rgb_mask, alpha, 0)
    return overlay

def main(data_path, cwd):
    # if str(data_path).endswith('.png', '.jpg', '.jpeg'):
    #     image_paths = [data_path]
    # else:
    image_paths = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    cuda_id = 1
    # Initialize the model, loss function, and optimizer
    encoder_name = 'mit_b0'
    encoder_weights =  None
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights)  
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(cwd + '/models/trained_models/Unet_mit_b0_e100_batch2_AdamW_DiceLoss_v2.2/best_model' + '/best_checkpoint.pth', weights_only=True, map_location=device)['model_state_dict'])
    
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for img_file in image_paths:
            img_path = os.path.join(data_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            t0 = time.time()
            outputs = model(input_tensor)
            t1 = time.time()
            print("\nTotal inference time: ", round(t1-t0,3), "ms")
            # Use sigmoid to build propabilities for each pixel
            preds = torch.sigmoid(outputs)
            # Convert propabilities to binary predictions (0, 1) by thresholding at 0.5
            # img = np.transpose(img, (1, 2, 0))
            
            
            preds = (preds > 0.4).int()
            img = mask_image(img, preds.squeeze().cpu())
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.show()
        # cv2.imwrite('/media/innovation/STORAGE/Projects/bluebot-Ktp/transformers/results/image{}.jpg'.format(i), img)
    
if __name__=='__main__':
    cwd = os.getcwd()
    data_path = Path("../datasets/version2/v2.2_part1/Bluebot_v2.2_mask_semantic_part1/train/images")
    main(data_path, cwd)