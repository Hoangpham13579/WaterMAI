import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import added to fix NameError

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from models import MSNet, RTFNet, UNet
from utils.data_loading import BasicDataset, load_image
from utils.dice_score import dice_loss


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_checkpoint: int = 10,
        dir_checkpoint: str = None,
        img_size: int = 640,
        amp: bool = False,
        train_txt: str = None,
        val_txt: str = None,
        dir_img: Path = None,
        dir_mask: Path = None,
):

    # Calculate scale based on img_size
    scale = min(img_size / 640, 1.0)  # Assuming 640 is the default/max size

    # Load datasets based on train and validation .txt files
    try:
        with open(train_txt, 'r') as f:
            train_ids = [line.strip() for line in f if line.strip()]
        with open(val_txt, 'r') as f:
            val_ids = [line.strip() for line in f if line.strip()]

        train_dataset = BasicDataset(
            images_dir=dir_img,
            mask_dir=dir_mask,
            img_size=img_size,
            scale=scale,
            mask_suffix='',
            ids=train_ids
        )

        val_dataset = BasicDataset(
            images_dir=dir_img,
            mask_dir=dir_mask,
            img_size=img_size,
            scale=scale,
            mask_suffix='',
            ids=val_ids
        )

    except Exception as e:
        logging.error(f"Error creating datasets: {e}")
        sys.exit(1)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    if n_val == 0 or n_train == 0:
        logging.error("Dataset has zero samples for training or validation.")
        sys.exit(1)

    # Create data loaders
    loader_args = dict(batch_size=batch_size,
                       num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     every {save_checkpoint} epochs
        Device:          {device.type}
        Image size:      {img_size}
        Mixed Precision: {amp}
    ''')

    # Set up optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5)
    # Auto Mixed Precision (float16 + float32) -> improve performance & reduce memory usage
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # Get unique values from the mask in the training dataset
    mask_values = train_dataset.get_mask_values()
    model.mask_values = mask_values  # Attach mask_values to the model

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device.type if device.type != 'mps' else 'cpu', enabled=amp
                ):
                    masks_pred = model(images)
                    if model.n_classes > 1:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(
                                0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    else:
                        loss = criterion(masks_pred.squeeze(1),
                                         true_masks.float())
                        loss += dice_loss(
                            torch.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False
                        )

                # Backward pass and optimizer step
                if amp:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Log average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch}: Training Loss: {avg_train_loss}')

        # Evaluation round
        avg_dice_score, precision, recall, low_dice_batch_idx = evaluate(model, val_loader, device, amp)

        logging.info(
            f"Evaluation Results:\n"
            f"  Precision: {precision:.4f}\n"
            f"  Recall: {recall:.4f}\n"
            f"  Dice Score: {avg_dice_score:.4f}\n"
            f"  Batch with Lowest Dice Score: {low_dice_batch_idx}"
        )

        # Update the learning rate scheduler
        scheduler.step(avg_dice_score)

        # Save checkpoint
        if epoch % save_checkpoint == 0:
            try:
                checkpoint_dir = Path(dir_checkpoint)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / \
                    f'checkpoint_epoch{epoch}.pth'
                # Save model state_dict along with mask_values
                state = {
                    'model_state_dict': model.state_dict(),
                    'mask_values': mask_values
                }
                torch.save(state, str(checkpoint_path))
                logging.info(
                    f'Checkpoint {epoch} saved successfully at {checkpoint_path}')
            except Exception as e:
                logging.error(
                    f'Failed to save checkpoint at epoch {epoch}. Error: {str(e)}')

    logging.info(f'Training completed. Checkpoint directory: {dir_checkpoint}')
    checkpoint_files = list(Path(dir_checkpoint).glob('*.pth'))
    logging.info(
        f'Checkpoint files found: {[f.name for f in checkpoint_files]}')

    if not checkpoint_files:
        logging.warning('No checkpoint files found after training!')
        logging.info(
            f'Contents of checkpoint directory: {list(Path(dir_checkpoint).iterdir())}')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet, MSNet, and RTFNet on images and target masks'
    )
    parser.add_argument(
        "--model",
        "-m",
        default="msnet",
        help="msnet, unet, rtfnet",
        required=True,
    )
    parser.add_argument(
        "--input_folder",
        "-if",
        metavar="input_folder",
        default="",
        help="Path to folder containing images and label (mask) folders for training only",
        required=True
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="",
        help="co (RGB), coir (RGB+NIR), cognirndwi (RGB+Green+NIR+NDWI))",
    )
    parser.add_argument(
        '--epochs',
        '-e',
        metavar='E',
        type=int,
        default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        '-b', dest='batch_size',
        metavar='B',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        '-l',
        metavar='LR',
        type=float,
        default=1e-4,
        help='Learning rate',
        dest='lr'
    )
    parser.add_argument(
        '--load',
        '-f',
        type=str,
        default='',
        help='Load model from a .pth file'
    )
    parser.add_argument(
        '--img-size',
        '-is',
        dest="img_size",
        type=int,
        default=640,
        help='MSNet, RTFNet models only allow 640*640 input size'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Use mixed precision'
    )
    parser.add_argument(
        '--n_classes',
        '-c',
        type=int,
        default=5,
        help='Number of classes'
    )
    parser.add_argument(
        "--num_channels", 
        "-nc", 
        type=int, 
        default=3, 
        help="Number of channels in model"
    )
    parser.add_argument(
        '--checkpoint-save',
        '-sc',
        dest='save_checkpoint',
        type=int,
        default=10,
        help='Save checkpoint every N epochs, "epoch" modulo (%) "save_checkpoint" should equal to 0'
    )
    parser.add_argument(
        '--train_txt',
        '-train',
        type=str,
        required=True,
        help='Path to the .txt file, containing list of image files paths for training'
    )
    parser.add_argument(
        '--val_txt',
        '-val',
        type=str,
        required=True,
        help='Path to the .txt file, containing list of image files paths for validation'
    )
    parser.add_argument(
        '--dir_checkpoint',
        '-dc',
        type=str,
        required=True,
        help='Path to the directory to save checkpoints'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

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

    # Determine number of input channels dynamically if not specified
    if args.num_channels is None:
        sample_img_path = next(dir_img.glob('*.*'))
        sample_img = load_image(sample_img_path, img_size=args.img_size)
        n_channels = sample_img.shape[0]
        logging.info(
            f'Detected {n_channels} input channels from sample image.')
    else:
        n_channels = args.num_channels
        logging.info(f'Using specified number of input channels: {n_channels}')

    # Load models
    if args.model == "msnet":
        model = MSNet(n_channels=args.num_channels, n_classes=args.n_classes)
    elif args.model == "rtfnet":
        model = RTFNet(n_channels=args.num_channels, n_classes=args.n_classes)
    elif args.model == "unet":
        model = UNet(n_channels=args.num_channels,
                     n_classes=args.n_classes, bilinear=False)
    model.to(memory_format=torch.channels_last)
    model.to(device=device)

    # Apply weight initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_weights)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Optionally load model weights if specified
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            mask_values = checkpoint['mask_values']
            model.mask_values = mask_values
            logging.info(f'Model and mask values loaded from {args.load}')
        else:
            model.load_state_dict(checkpoint)
            logging.info(f'Model loaded from {args.load}')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.img_size,
            amp=args.amp,
            save_checkpoint=args.save_checkpoint,
            dir_checkpoint=args.dir_checkpoint,
            train_txt=args.train_txt,
            val_txt=args.val_txt,
            dir_img=dir_img,
            dir_mask=dir_mask
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
