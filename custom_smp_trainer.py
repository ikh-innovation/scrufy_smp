import math
import os
from tqdm import tqdm
from pathlib import Path
import sys

import torch
import torch.optim as optim
from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A

import segmentation_models_pytorch as smp
import wandb

from dataloader import BluebotDataset
from metrics import *

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))


def calculate_metrics(preds, labels):
    """Calculate accuracy and IoU for a batch.

    Args:
        preds (_type_): _description_
        labels (_type_): _description_
    """
       
    global_iou, per_class_iou = iou_metric(preds, labels)
    global_acc, per_class_acc = accuracy_metric(preds, labels)
    f1_score = f1_metric(preds, labels)
    precision_score = precision_metric(preds, labels)
    recall_score = recall_metric(preds, labels)
    
    return global_iou, per_class_iou, global_acc, per_class_acc, f1_score, precision_score, recall_score

def create_paths(root, target):
    img_dir = root + target + '/images'
    mask_dir = root + target + '/labels'
    image_paths = [os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir))]
    labels_paths = [os.path.join(mask_dir, fname) for fname in sorted(os.listdir(mask_dir))]
    
    return image_paths, labels_paths

def main(data_root, cwd):
    
    # Create paths for training and val images/labels
    train_image_paths, train_mask_paths = create_paths(data_root, "/train")
    val_image_paths, val_mask_paths = create_paths(data_root, "/valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create transformations to use in training with Albumentations
    transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    # Create pytorch dataset class for custom train/val dataset
    train_dataset = BluebotDataset(inputs=train_image_paths, targets=train_mask_paths, transform=transform)
    val_dataset = BluebotDataset(inputs=val_image_paths, targets=val_mask_paths, transform=transform)
    
    # Create pytorch dataloader classes for train/val with given batch size
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    encoder_name = 'mit_b0'
    encoder_weights = 'imagenet'
    
    model = smp.Unet(
        encoder_name=encoder_name,        # Use ResNet-34 as the encoder
        encoder_weights=encoder_weights,     # Use ImageNet pre-trained weights
        in_channels=3,                  # Input channels (RGB)
        classes=1,                      # Output channels (1 for binary segmentation)
    )
        
    num_epochs = 50
    
    # Define loss function, optimizer and learning rate scheduler
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    # loss_fn = smp.losses.SoftBCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    lf = lambda x: ((1 - math.cos(x * math.pi / num_epochs)) / 2) * (0.1 - 1) + 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Initialize Weights & Biases (W&B)
    project= "smp_custom_training"
    session_name = f"{model.__class__.__name__}_{encoder_name}_e{num_epochs}_batch{batch_size}_{optimizer.__class__.__name__}_{loss_fn.__class__.__name__}_v2.2"
    wandb.login()
    wandb.init(project=project,
                name=session_name)
    
    # Training and validation loop
    model = model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / len(train_loader))
            
        if scheduler:
            scheduler.step()
            
        learning_rate = optimizer.param_groups[0]['lr']
        wandb.log({'lr': learning_rate, 'epoch': epoch+1})
        
        # Validation loop
        model.eval()              
        val_loss = 0.0
        tot_per_class_iou = [0.0, 0.0]
        tot_per_class_acc = [0.0, 0.0]
        precision_total = 0.0
        recall_total = 0.0
        f1_total = 0.0
        total_acc = 0.0
        total_iou = 0.0
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass 
                logits = model(images)
                loss = loss_fn(logits, masks)
                
                val_loss += loss.item()
                
                # Use sigmoid to build propabilities for each pixel
                preds = torch.sigmoid(logits)
                # Convert propabilities to binary predictions (0, 1) by thresholding at 0.5
                preds = (preds > 0.5).int()
                global_iou, per_class_iou, global_acc, per_class_acc, f1_score, precision_score, recall_score = calculate_metrics(preds, masks)
                
                for idx in range(len(per_class_acc)):
                    tot_per_class_acc[idx] += per_class_acc[idx]
                    tot_per_class_iou[idx] += per_class_iou[idx]
                
                total_acc += global_acc
                total_iou += global_iou
                
                precision_total += precision_score
                recall_total += recall_score
                f1_total += f1_score
                pbar.set_postfix(val_loss=val_loss / len(val_loader))
        
        avg_val_loss = val_loss / len(val_loader)
        tot_per_class_acc = [elem / len(val_loader) for elem in tot_per_class_acc]
        tot_per_class_iou = [elem / len(val_loader) for elem in tot_per_class_iou]
        precision_avg = precision_total / len(val_loader)
        recall_avg = recall_total / len(val_loader)
        f1_avg = f1_total / len(val_loader)
        
        mean_accuracy = total_acc / len(val_loader)
        mean_iou = (tot_per_class_iou[0] + tot_per_class_iou[1]) / 2
        
        target_path = f'models/trained_models/{session_name}'

        if not os.path.isdir(cwd + '/' + target_path):
            os.makedirs(cwd + '/' +  target_path)
        if not os.path.isdir(cwd + '/' + target_path + '/best_model'):
            os.makedirs(cwd + '/' + target_path + '/best_model')
        # Save checkpoint if val loss is improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {'epoch': num_epochs,
                  'model_state_dict': model.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'train_loss': train_loss/ (len(train_loader)),
                  'iou': mean_iou,
                  'accuracy': mean_accuracy}
    
            torch.save(checkpoint, cwd + '/' + target_path + '/best_model' + '/' + f'best_checkpoint.pth')

        
        # Log the metrics to W&B
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "Precision": precision_avg,
            "Recall": recall_avg,
            "F1_score": f1_avg,
            "IoU/mIoU": mean_iou,
            f"IoU/Class_{0}": tot_per_class_iou[0],
            f"IoU/Class_{1}": tot_per_class_iou[1],
            "Acc/mean_accuary": mean_accuracy,
            f"Acc/Class_{0}": tot_per_class_acc[0],
            f"Acc/Class_{1}": tot_per_class_acc[1]
        })
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, IoU: {mean_iou:.4f}, Per Class IoU: {tot_per_class_iou}, Accuracy: {mean_accuracy:.4f}, Per Class Accuracy: {tot_per_class_acc}")

    print("Training complete.")

    print(f"Model and training components saved at {cwd + '/' + target_path + '/best_model' + '/' + f'best_checkpoint.pth'}")
    print(f"\n Best model at epoch {checkpoint['epoch']}")


if __name__=='__main__':
    cwd = os.getcwd()
    data_root = '/media/innovation/STORAGE/Projects/bluebot-Ktp/datasets/version2/v2.2_part1/Bluebot_v2.2_mask_semantic_part1'
    main(data_root, cwd)