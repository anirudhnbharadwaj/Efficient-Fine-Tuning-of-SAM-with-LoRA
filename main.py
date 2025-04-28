import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging
import numpy as np
import wandb
import argparse 
from data.data import NuInsSegDataset, get_dataloaders
from data.utils import visualize_initial_sample, visualize_fold_predictions, compute_metrics
from models.model import SAMWithLoRA

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, epoch, fold, path):
    checkpoint = {
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    fold = checkpoint["fold"]
    logging.info(f"Loaded checkpoint from {path}, resuming at epoch {epoch+1}, fold {fold+1}")
    return epoch, fold

def train_and_eval(model, train_loader, val_loader, criterion, optimizer, device, config, fold, resume=False):
    num_epochs = config["num_epochs"]
    val_interval = config["val_interval"]
    log_dir = config["log_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold+1}.pth")
    
    if resume and os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
    
    wandb.init(
        project="nuinsseg_sam_lora_trial",
        name=f"fold_{fold+1}",
        config=config,
        resume="allow" if resume else None,
        id=f"fold_{fold+1}" if resume else None
    )
    
    model.train()
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0
        with tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (images, masks, bboxes) in enumerate(pbar):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images, bboxes)
                
                loss = 0
                num_masks = 0
                for output, mask, bbox in zip(outputs, masks, bboxes):
                    num_boxes = bbox.size(0)
                    if num_boxes == 0:
                        logging.warning(f"Fold {fold+1} Epoch {epoch+1} Batch {batch_idx}: No boxes")
                        continue
                    
                    # Compute ground truth instances
                    from skimage.measure import label
                    mask_np = mask.cpu().numpy()
                    labeled_mask = label(mask_np > 0)
                    num_instances = len(np.unique(labeled_mask)) - 1  # Exclude background
                    
                    if num_boxes != num_instances:
                        logging.warning(
                            f"Fold {fold+1} Epoch {epoch+1} Batch {batch_idx}: "
                            f"Mismatch - Boxes: {num_boxes}, Instances: {num_instances}"
                        )
                        # Use min to avoid shape mismatch
                        num_boxes = min(num_boxes, num_instances)
                        if num_boxes == 0:
                            continue
                    
                    output = output[:num_boxes]  # [num_boxes, 256, 256]
                    mask_expanded = mask.unsqueeze(0).expand(num_boxes, -1, -1).float()
                    mask_resized = F.interpolate(
                        mask_expanded.unsqueeze(0), size=(256, 256), mode='nearest'
                    ).squeeze(0)  # [num_boxes, 256, 256]
                    mask_resized = (mask_resized > 0).float()
                    
                    try:
                        loss += criterion(output, mask_resized)
                        num_masks += num_boxes
                    except Exception as e:
                        logging.error(
                            f"Loss computation failed: output shape {output.shape}, "
                            f"mask_resized shape {mask_resized.shape}, error: {str(e)}"
                        )
                        continue
                
                if num_masks == 0:
                    logging.warning(f"Fold {fold+1} Epoch {epoch+1} Batch {batch_idx}: No valid masks")
                    continue
                
                loss = loss / num_masks
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"loss": train_loss / (pbar.n + 1)})
                torch.cuda.empty_cache()
                
                wandb.log({"epoch": epoch + 1, "train_loss": loss.item()})
        
        avg_loss = train_loss / len(train_loader)
        logging.info(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
        
        save_checkpoint(model, optimizer, epoch, fold, checkpoint_path)
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_metrics = {"Dice": [], "AJI": [], "PQ": []}
            with torch.no_grad():
                for images, masks, bboxes in tqdm(val_loader, desc=f"Fold {fold+1} Validation"):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images, bboxes)
                    preds = []
                    for output, mask in zip(outputs, masks):
                        pred = torch.sigmoid(output)
                        pred = (pred > 0.5).float()
                        pred_combined = torch.zeros_like(mask, device=device)
                        for p in pred:
                            p_up = F.interpolate(
                                p.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False
                            ).squeeze(0).squeeze(0)
                            pred_combined = torch.max(pred_combined, p_up)
                        preds.append(pred_combined)
                    batch_metrics = compute_metrics(preds, masks)
                    for k, v in batch_metrics.items():
                        val_metrics[k].append(v)
                    torch.cuda.empty_cache()
            
            avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            logging.info(f"Fold {fold+1} Epoch {epoch+1} Validation Metrics: {avg_metrics}")
            wandb.log({"epoch": epoch + 1, "val_dice": avg_metrics["Dice"], "val_aji": avg_metrics["AJI"], "val_pq": avg_metrics["PQ"]})
            model.train()
    
    # Final validation
    model.eval()
    val_metrics = {"Dice": [], "AJI": [], "PQ": []}
    with torch.no_grad():
        for images, masks, bboxes in tqdm(val_loader, desc=f"Fold {fold+1} Final Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images, bboxes)
            preds = []
            for output, mask in zip(outputs, masks):
                pred = torch.sigmoid(output)
                pred = (pred > 0.5).float()
                pred_combined = torch.zeros_like(mask, device=device)
                for p in pred:
                    p_up = F.interpolate(
                        p.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    pred_combined = torch.max(pred_combined, p_up)
                preds.append(pred_combined)
            batch_metrics = compute_metrics(preds, masks)
            for k, v in batch_metrics.items():
                val_metrics[k].append(v)
            torch.cuda.empty_cache()
    
    avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    logging.info(f"Fold {fold+1} Final Validation Metrics: {avg_metrics}")
    wandb.log({"fold": fold + 1, "final_val_dice": avg_metrics["Dice"], "final_val_aji": avg_metrics["AJI"], "final_val_pq": avg_metrics["PQ"]})
    
    # Save fold visualization
    viz_path = os.path.join(log_dir, f"fold_{fold+1}_final_viz.png")
    visualize_fold_predictions(model, val_loader, fold, viz_path, device)
    wandb.log({"fold_viz": wandb.Image(viz_path)})
    
    wandb.finish()
    return avg_metrics

def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    setup_logging(config["log_dir"])
    logging.info(f"Logging initialized at {config['log_dir']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    dataset = NuInsSegDataset(config["data_dir"])
    
    # Save initial visualization
    initial_viz_path = os.path.join(config["log_dir"], "initial_sample_viz.png")
    visualize_initial_sample(dataset, initial_viz_path)
    
    model = SAMWithLoRA(
        config["model_checkpoint"],
        config["lora_r"],
        config["lora_alpha"],
        config["lora_dropout"]
    )
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    kf = KFold(n_splits=config["num_folds"], shuffle=True, random_state=config["random_seed"])
    fold_results = []
    
    # Start from Fold 4 (index 3)
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset))), start=3):
        logging.info(f"Starting Fold {fold+1}/{config['num_folds']}")
        train_loader, val_loader = get_dataloaders(
            dataset,
            train_idx,
            val_idx,
            config["batch_size"],
            num_workers=config["num_workers"]
        )
        metrics = train_and_eval(
            model, train_loader, val_loader, criterion, optimizer, device,
            config, fold, resume=True
        )
        fold_results.append(metrics)
    
    avg_results = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0].keys()}
    logging.info(f"Average Fold Results: {avg_results}")
    
    torch.save(model.state_dict(), os.path.join(config["log_dir"], "final_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM with LoRA for NuInsSeg")
    parser.add_argument("--config", type=str, default="config/params.json", help="Path to config JSON file")
    args = parser.parse_args()
    
    main(args.config)