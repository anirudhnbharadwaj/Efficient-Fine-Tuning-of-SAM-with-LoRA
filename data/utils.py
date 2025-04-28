import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from PIL import Image
from skimage.measure import label, regionprops

sns.set_style("darkgrid")

def dice_coefficient(pred, target, epsilon=1e-6):
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice

def aggregated_jaccard_index(pred, target, epsilon=1e-6):
    pred_labels = label(pred > 0)
    target_labels = label(target > 0)
    pred_instances = set(np.unique(pred_labels)) - {0}
    target_instances = set(np.unique(target_labels)) - {0}
    ious = []
    for p in pred_instances:
        pred_mask = (pred_labels == p).astype(np.float32)
        best_iou = 0
        for t in target_instances:
            target_mask = (target_labels == t).astype(np.float32)
            intersection = (pred_mask * target_mask).sum()
            union = (pred_mask + target_mask).sum() - intersection
            iou = intersection / (union + epsilon)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
    return np.mean(ious) if ious else 0.0

def panoptic_quality(pred, target, epsilon=1e-6):
    pred_labels = label(pred > 0)
    target_labels = label(target > 0)

    pred_instances = set(np.unique(pred_labels)) - {0}
    target_instances = set(np.unique(target_labels)) - {0}

    matched_pairs = []
    for p in pred_instances:
        pred_mask = (pred_labels == p).astype(np.float32)
        best_iou = 0
        best_t = None
        for t in target_instances:
            target_mask = (target_labels == t).astype(np.float32)
            intersection = (pred_mask * target_mask).sum()
            union = (pred_mask + target_mask).sum() - intersection
            iou = intersection / (union + epsilon)
            if iou > best_iou and iou > 0.5:
                best_iou = iou
                best_t = t
        if best_t is not None:
            matched_pairs.append((p, best_t))

    TP = len(matched_pairs)
    FP = len(pred_instances) - TP
    FN = len(target_instances) - TP

    if TP == 0:
        return 0.0

    sq = 0
    for p, t in matched_pairs:
        pred_mask = (pred_labels == p).astype(np.float32)
        target_mask = (target_labels == t).astype(np.float32)
        intersection = (pred_mask * target_mask).sum()
        union = (pred_mask + target_mask).sum() - intersection
        iou = intersection / (union + epsilon)
        sq += iou
    sq = sq / TP if TP > 0 else 0

    rq = TP / (TP + 0.5 * FP + 0.5 * FN + epsilon)
    pq = sq * rq

    return pq

def visualize_initial_sample(dataset, save_path):
    """Visualize a random sample's image and ground truth mask before training."""
    if not dataset:
        logging.error("Dataset is empty in visualize_initial_sample.")
        return
    
    idx = np.random.randint(0, len(dataset))
    image, mask, _ = dataset[idx]
    
    image = image.permute(1, 2, 0).numpy()  # [H, W, C]
    mask = mask.numpy()  # [H, W]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(mask, cmap="rocket", cbar=False)
    plt.title("Ground Truth Mask")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Initial sample visualization saved to {save_path}")

def visualize_fold_predictions(model, loader, fold, save_path, device):
    """Visualize predictions vs ground truth for the first sample in the validation set."""
    if not loader.dataset:
        logging.error("Dataset is empty in visualize_fold_predictions.")
        return
    
    model.eval()
    with torch.no_grad():
        # Get the first batch
        images, masks, bboxes = next(iter(loader))
        images, masks = images.to(device), masks.to(device)
        
        # Use the first sample
        image = images[0:1]  # [1, C, H, W]
        mask = masks[0]      # [H, W]
        
        # Perform inference
        outputs = model(image, [bboxes[0]])  # List of [N_i, 256, 256]
        pred = torch.sigmoid(outputs[0])     # [N_i, 256, 256]
        pred = (pred > 0.5).float()
        
        # Aggregate predictions
        pred_combined = torch.zeros_like(mask, device=device)
        for p in pred:
            p_up = torch.nn.functional.interpolate(
                p.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            pred_combined = torch.max(pred_combined, p_up)
        
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        pred_np = pred_combined.cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("Input Image")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        sns.heatmap(mask_np, cmap="rocket", cbar=False)
        plt.title("Ground Truth")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        sns.heatmap(pred_np, cmap="rocket", cbar=False)
        plt.title("Prediction")
        plt.axis("off")
        
        plt.suptitle(f"Fold {fold+1} Validation Sample")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Fold {fold+1} prediction visualization saved to {save_path}")

def compute_metrics(preds, targets):
    """Compute Dice, AJI, and PQ metrics for a batch of predictions."""
    metrics = {"Dice": [], "AJI": [], "PQ": []}
    for pred, target in zip(preds, targets):
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        metrics["Dice"].append(dice_coefficient(pred_np, target_np))
        metrics["AJI"].append(aggregated_jaccard_index(pred_np, target_np))
        metrics["PQ"].append(panoptic_quality(pred_np, target_np))
    return {k: np.mean(v) for k, v in metrics.items()}