import argparse
import logging
import os
import sys
import joblib
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
        "--model", "-m", default="MODEL.pth", metavar="FILE", help="Specify the file in which the model is stored", required=True,
    )
    parser.add_argument(
        "--input_folder", "-if", metavar="INPUT_FOLDER", default="", help="Path to the WaterMAI_dataset folder", required=True,
    )
    parser.add_argument(
        "--weights", "-w", default="MODEL.pth", metavar="FILE", help="Checkpoint weights", required=True,
    )
    parser.add_argument(
        "--num_channels", "-nc", type=int, default=3, help="Number of input channels in the model",
    )
    parser.add_argument(
        "--type", "-t", type=str, default="", help="Type of dataset (e.g., 'co', 'coir', 'cognirndwi', etc.)", required=True,
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=2, help="size of each batch"
    )
    parser.add_argument(
        "--img_size", "-is", type=int, default=640, help="image size, currently only supports 640"
    )
    parser.add_argument(
        '--n_classes', '-c', type=int, default=5, help='Number of classes'
    )
    parser.add_argument(
        "--val_txt", "-val", type=str, required=True, help="Path to the .txt file, containing list of image paths for validation"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load models
    if args.model == "msnet":
        model = MSNet(n_channels=args.num_channels, n_classes=args.n_classes)
    elif args.model == "rtfnet":
        model = RTFNet(n_channels=args.num_channels, n_classes=args.n_classes)
    elif args.model == "unet":
        model = UNet(n_channels=args.num_channels,
                     n_classes=args.n_classes, bilinear=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(memory_format=torch.channels_last)
    model.to(device=device)

    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    # Load weight
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mask_values = checkpoint['mask_values']
    model.mask_values = mask_values
    logging.info("Model loaded!")

    # Set input folder path based on the type of dataset
    if args.type == "co":
        dir_img = args.input_folder + "/images/co"
        dir_mask = args.input_folder + "/labels/masks_co"
    elif args.type == "cognirndwi":
        dir_img = args.input_folder + "/images/cognirndwi"
        dir_mask = args.input_folder + "/labels/masks_cognirndwi"
    elif args.type == "coir":
        dir_img = args.input_folder + "/images/coir"
        dir_mask = args.input_folder + "/labels/masks_coir"
    elif args.type == "condwi":
        dir_img = args.input_folder + "/images/condwi"
        dir_mask = args.input_folder + "/labels/masks_condwi"
    elif args.type == "gnirndwi":
        dir_img = args.input_folder + "/images/gnirndwi"
        dir_mask = args.input_folder + "/labels/masks_gnirndwi"
    elif args.type == "rndwib":
        dir_img = args.input_folder + "/images/rndwib"
        dir_mask = args.input_folder + "/labels/masks_rndwib"

    # Create the test dataset
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_img_dir = os.path.join(args.input_folder, "images", args.type)
    test_mask_dir = os.path.join(args.input_folder, "labels", f"mask_{args.type}")

    # Calculate scale based on img_size
    scale = min(args.img_size / 640, 1.0)  # Assuming 640 is the default/max size

    # Create the test dataset
    try:
        with open(args.val_txt, 'r') as f:
            val_ids = [line.strip() for line in f if line.strip()]
    
        val_dataset = BasicDataset(
            images_dir=dir_img,
            mask_dir=dir_mask,
            img_size=args.img_size,
            scale=scale,
            mask_suffix='',
            ids=val_ids
        )
    except Exception as e:
        logging.error(f"Error creating datasets: {e}")
        sys.exit(1)

    test_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    # Evaluate the model
    dice_score, precision_value, recall_value, low_dice_batch_idx = evaluate(
        model, test_loader, device, amp=False
    )

    logging.info(
        f"Evaluation Results:\n"
        f"  Precision: {precision_value:.4f}\n"
        f"  Recall: {recall_value:.4f}\n"
        f"  Dice Score: {dice_score:.4f}\n"
        f"  Batch with Lowest Dice Score: {low_dice_batch_idx}"
    )