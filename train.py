import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from models import MSNet, RTFNet, UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

# Directory save checkpoints
dir_checkpoint = Path("./checkpoints/")
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)


def train_model(
    model,
    input_folder,
    device,
    epochs: int = 2,
    checkpoint_save: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-2,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_size: int = 640,
    amp: bool = True,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    if args.type == "coir":
        dir_img = input_folder + "/images/coir"
        dir_mask = input_folder + "/labels/masks_coir"
    elif args.type == "condwi":
        dir_img = input_folder + "/images/condwi"
        dir_mask = input_folder + "/labels/masks_condwi"
    elif args.type == "cognirndwi":
        dir_img = input_folder + "/images/cognirndwi"
        dir_mask = input_folder + "/labels/masks_cognirndwi"

    try:
        dataset = BasicDataset(dir_img, dir_mask, img_size)
    except Exception as e:
        print("Loading dataset error: ", e)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project="WaterMAI", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            img_size=img_size,
            amp=amp,
        )
    )

    # NOTE: "Mixed Precision" uses both 32-bit and 16-bit floating point types to speed up training (3x faster in GPU & 60% faster in TPU)
    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:     {img_size}
        Mixed Precision: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True
    )

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lrf = 0.2
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf  # linear
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

                assert images.shape[1] == model.n_channels, (
                    f"Network has been defined with {model.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(
                    device.type if device.type != "mps" else "cpu", enabled=amp
                ):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(
                            torch.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                        )
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes)
                            .permute(0, 3, 1, 2)
                            .float(),
                            multiclass=True,
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                )
                pbar.set_postfix(**{"loss (batch)": loss.item()})

        # Evaluation round
        with torch.no_grad():
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace("/", ".")
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())

            dice_score, precision, recall = evaluate(model, val_loader, device, amp)
            scheduler.step(dice_score)

            logging.info(
                "Validation Dice score: {}; Precision: {}; Recall: {}".format(
                    dice_score, precision, recall
                )
            )
            try:
                experiment.log(
                    {
                        "learning rate": optimizer.param_groups[0]["lr"],
                        "validation Dice": dice_score,
                        "Precision": precision,
                        "Recall": recall,
                        "images": wandb.Image(images[0].cpu()),
                        "masks": {
                            "true": wandb.Image(true_masks[0].float().cpu()),
                            "pred": wandb.Image(
                                masks_pred.argmax(dim=1)[0].float().cpu()
                            ),
                        },
                        "step": global_step,
                        "epoch": epoch,
                        **histograms,
                    }
                )
            except:
                pass

        if save_checkpoint and (epoch % checkpoint_save == 0):
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(
                state_dict, str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch))
            )
            logging.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the MSNet on images and target masks"
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
        help="Path to dataset folder containing imgs and masks folder",
        required=True,
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="",
        help="co, coir, cognirndwi (RGB+Green+NIR+NDWI))",
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=0.001,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--img-size",
        "-is",
        dest="img_size",
        type=float,
        default=640,
        help="MSNet model only allows 640*640 input size",
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--num_channels", "-nc", type=int, default=3, help="Number of channels in model"
    )
    parser.add_argument(
        "--n_classes", "-c", type=int, default=1, help="Number of classes"
    )
    parser.add_argument(
        "--checkpoint-save",
        "-sch",
        dest="checkpoint_save",
        type=int,
        default=20,
        help="Number of epoch each checkpoint saved",
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
        model = UNet(
            n_channels=args.num_channels, n_classes=args.n_classes, bilinear=False
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")
    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    model.to(device=device)
    try:
        train_model(
            model=model,
            input_folder=args.input_folder,
            epochs=args.epochs,
            checkpoint_save=args.checkpoint_save,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.img_size,
            val_percent=args.val / 100,
            amp=args.amp,
        )
    except torch.cuda.OutOfMemoryError():
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            input_folder=args.input_folder,
            epochs=args.epochs,
            checkpoint_save=args.checkpoint_save,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.img_size,
            val_percent=args.val / 100,
            amp=args.amp,
        )

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
from unet import UNet
from utils.data_loading import BasicDataset, load_image
from utils.dice_score import dice_loss

dir_checkpoint = Path('./checkpoints_cogirndwi/fold_4')

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_checkpoint: int = 10,
        img_size: int = 640,
        amp: bool = False,
        train_txt: str = None,
        val_txt: str = None,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):  
    if args.type == "coir":
        dir_img = input_folder + "/images/coir"
        dir_mask = input_folder + "/labels/masks_coir"
    elif args.type == "condwi":
        dir_img = input_folder + "/images/condwi"
        dir_mask = input_folder + "/labels/masks_condwi"
    elif args.type == "cognirndwi":
        dir_img = input_folder + "/images/cognirndwi"
        dir_mask = input_folder + "/labels/masks_cognirndwi"

    try:
        dataset = BasicDataset(dir_img, dir_mask, img_size)
    except Exception as e:
        print("Loading dataset error: ", e)
    # Calculate scale based on img_size
    scale = min(img_size / 640, 1.0)  # Assuming 640 is the default/max size

    #Create datasets from train and validation txt files
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

    #Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # Get mask values from the training dataset
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
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes > 1:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    else:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
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
        val_dice_score, val_loss = evaluate(model, val_loader, device, amp)

        logging.info(f'Epoch {epoch}: Validation Loss: {val_loss}, Validation Dice Score: {val_dice_score}')

        # Update the learning rate scheduler
        scheduler.step(val_dice_score)

        # Save checkpoint
        if epoch % save_checkpoint == 0:
            try:
                checkpoint_dir = Path(dir_checkpoint)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
                # Save model state_dict along with mask_values
                state = {
                    'model_state_dict': model.state_dict(),
                    'mask_values': mask_values
                }
                torch.save(state, str(checkpoint_path))
                logging.info(f'Checkpoint {epoch} saved successfully at {checkpoint_path}')
            except Exception as e:
                logging.error(f'Failed to save checkpoint at epoch {epoch}. Error: {str(e)}')

    logging.info(f'Training completed. Checkpoint directory: {dir_checkpoint}')
    checkpoint_files = list(Path(dir_checkpoint).glob('*.pth'))
    logging.info(f'Checkpoint files found: {[f.name for f in checkpoint_files]}')

    if not checkpoint_files:
        logging.warning('No checkpoint files found after training!')
        logging.info(f'Contents of checkpoint directory: {list(Path(dir_checkpoint).iterdir())}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='', help='Load model from a .pth file')
    parser.add_argument("--input_folder", "-if", metavar="input_folder", default="", 
                        help="Path to dataset folder containing imgs and masks folder", required=True)
    parser.add_argument('--save_checkpoint', '-sc', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--size', '-s', type=int, default=640, help='Image size')
    parser.add_argument('--train_txt', '-train', type=str, required=True, help='Path to the train txt file')
    parser.add_argument('--val_txt', '-val', type=str, required=True, help='Path to the validation txt file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    parser.add_argument('--n_channels', '-nc', type=int, default=None,
                        help='Number of input channels')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Determine number of input channels dynamically if not specified
    if args.n_channels is None:
        sample_img_path = next(dir_img.glob('*.*'))
        sample_img = load_image(sample_img_path, img_size=args.size)
        n_channels = sample_img.shape[0] 
        logging.info(f'Detected {n_channels} input channels from sample image.')
    else:
        n_channels = args.n_channels
        logging.info(f'Using specified number of input channels: {n_channels}')

    # Initialize the model
    model = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    model.to(memory_format=torch.channels_last)
    model.to(device=device)

    #Apply weight initialization
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
            input_folder=args.input_folder,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.size,
            amp=args.amp,
            save_checkpoint=args.save_checkpoint,
            train_txt=args.train_txt,
            val_txt=args.val_txt
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)