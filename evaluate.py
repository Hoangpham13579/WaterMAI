import argparse
import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import UNet, MSNet, RTFNet
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from utils.dice_score import dice_coeff, multiclass_dice_coeff

from sklearn.metrics import precision_score, recall_score

def precision_multiclass(pred: torch.Tensor, target: torch.Tensor, average='macro'):
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    return precision_score(target, pred, average=average, zero_division=0)

def recall_multiclass(pred: torch.Tensor, target: torch.Tensor, average='macro'):
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    return recall_score(target, pred, average=average, zero_division=0)

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_dice_score = 0.0
    min_dice = 1.0
    low_dice_batch_idx = None

    all_preds = []
    all_targets = []

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            total=num_val_batches,
            desc="Evaluation",
            unit="batch",
            leave=False,
        )):
            images = batch["image"].to(device=device, dtype=torch.float32)
            true_masks = batch["mask"].to(device=device, dtype=torch.long)

            # Forward pass
            masks_pred = net(images)

            if net.n_classes > 1:
                # For multi-class segmentation
                pred_probs = F.softmax(masks_pred, dim=1)
                pred_labels = pred_probs.argmax(dim=1)
                true_masks_one_hot = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()

                # Calculate Dice score
                dice_score_batch = multiclass_dice_coeff(
                    pred_probs,
                    true_masks_one_hot
                )

                # Accumulate predictions and targets
                all_preds.append(pred_labels.cpu())
                all_targets.append(true_masks.cpu())

            else:
                # For binary segmentation
                pred_probs = torch.sigmoid(masks_pred)
                pred_labels = (pred_probs > 0.5).float()
                true_masks_float = true_masks.float()

                dice_score_batch = dice_coeff(
                    pred_labels,
                    true_masks_float
                )

                all_preds.append(pred_labels.cpu())
                all_targets.append(true_masks.cpu())

            #Dice score
            total_dice_score += dice_score_batch.item()

            # Track the batch with the lowest Dice score
            if dice_score_batch.item() < min_dice:
                min_dice = dice_score_batch.item()
                low_dice_batch_idx = batch_idx

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute precision and recall
    if net.n_classes > 1:
        precision = precision_multiclass(all_preds, all_targets, average='macro')
        recall = recall_multiclass(all_preds, all_targets, average='macro')
    else:
        precision = precision_score(
            all_targets.view(-1).numpy(),
            all_preds.view(-1).numpy(),
            zero_division=0
        )
        recall = recall_score(
            all_targets.view(-1).numpy(),
            all_preds.view(-1).numpy(),
            zero_division=0
        )

    # Calculate average Dice score
    avg_dice_score = total_dice_score / num_val_batches

    return avg_dice_score, precision, recall, low_dice_batch_idx

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on test data")
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
        required=True,
    )
    parser.add_argument(
        "--input_folder",
        "-if",
        metavar="INPUT_FOLDER",
        default="",
        help="Path to the input folder containing test images and masks",
        required=True,
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="",
        help="Type of dataset (e.g., 'co', 'coir', 'cognirndwi', etc.)",
        required=True,
    )
    parser.add_argument(
        "--num_channels",
        "-nc",
        type=int,
        default=3,
        help="Number of input channels in the model",
    )
    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        default=5,
        help="Number of output classes",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize the model
    inDir = args.input_folder

    # Unet model
    if args.model == "msnet":
        net = MSNet(n_classes=args.n_classes)
    elif args.model == "rtfnet":
        net = RTFNet(n_classes=args.n_classes)
    elif args.model == "unet":
        net = UNet(n_channels=args.num_channels, n_classes=args.n_classes, bilinear=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    # Load the model checkpoint
    net.to(device=device)
    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        mask_values = checkpoint.get('mask_values', [0, 1])
    else:
        net.load_state_dict(checkpoint)
        mask_values = [0, 1] 
    logging.info("Model loaded!")

    # Test dataloader
    if args.type == 'coir':
        dir_img = inDir + "/images/coir"
        dir_mask = inDir + "/labels/mask_coir"
    elif args.type == 'condwi':
        dir_img = inDir + "/images/condwi"
        dir_mask = inDir + "/labels/mask_condwi"
    elif args.type == 'cognirndwi':
        dir_img = inDir + "/images/cognirndwi"
        dir_mask = inDir + "/labels/mask_cognirndwi"
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # Set up the test dataset and dataloader
    test_img_dir = os.path.join(args.input_folder, "images", args.type)
    test_mask_dir = os.path.join(args.input_folder, "labels", f"mask_{args.type}")

    # Create the test dataset
    test_dataset = BasicDataset(
        images_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_size=640,
        scale=1.0,
        mask_suffix="", 
        ids=None 
    )

    loader_args = dict(batch_size=16, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    dice_score, precision_value, recall_value, low_dice_batch_idx = evaluate(
        net, test_loader, device, amp=False
    )

    logging.info(
        f"Evaluation Results:\n"
        f"  Precision: {precision_value:.4f}\n"
        f"  Recall: {recall_value:.4f}\n"
        f"  Dice Score: {dice_score:.4f}\n"
        f"  Batch with Lowest Dice Score: {low_dice_batch_idx}"
    )