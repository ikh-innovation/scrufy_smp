import cv2
import os


folder = '/media/innovation/STORAGE/Projects/bluebot-Ktp/datasets/version2/v2.2_part1/Bluebot_v2.2_mask_semantic_part1/valid/labels'
d_w = 640
d_h = 640

for file in os.listdir(folder):
    img_path = os.path.join(folder, file)
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if w != d_w or h != d_h:
        resized_img = cv2.resize(img, (d_w, d_h))
        cv2.imwrite(img_path, resized_img)