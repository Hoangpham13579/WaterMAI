import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from os.path import splitext, isfile, join
import cv2
import glob
import tifffile as tiff
from pathlib import Path
from skimage.transform import resize

from models import UNet, MSNet, RTFNet
from utils.data_loading import load_image 

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.tensor(full_img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored', required=True)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--input_folder', '-if', metavar='INPUT_FOLDER', default='', help='Input folder containing images')
    parser.add_argument('--output_folder', '-of', metavar='OUTPUT_FOLDER', default='', help='Output folder for masks', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel positive')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--num_channels', '-nc', type=int, default=3, help='Number of input channels in the model')
    parser.add_argument('--img_size', '-sz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--mask_suffix', '-ms', type=str, default='', help='Suffix for mask files')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()

def mask_to_image(mask: np.ndarray, mask_values):
    """
    Convert a mask array to an RGB image using the provided mask_values (colors).
    Args:
        mask (np.ndarray): The mask array with class labels.
        mask_values (list): List of RGB tuples representing class colors.
    Returns:
        PIL.Image: The mask converted to an RGB image.
    """
    out_shape = (mask.shape[0], mask.shape[1], 3)  # RGB image
    out = np.zeros(out_shape, dtype=np.uint8)

    for i, color in enumerate(mask_values):
        out[mask == i] = color

    return Image.fromarray(out)

def load_image(filename, img_size):
    ext = ''.join(Path(filename).suffixes).lower()
    if ext in ['.tiff', '.tif']:
        img = tiff.imread(str(filename))
        img = np.array(img, dtype=np.float32)

        if img.max() > 1:
            img /= 255.0

        # Determine the number of channels
        if img.ndim == 2:
            num_channels = 1
        else:
            num_channels = img.shape[2]

        # Resize the image
        if num_channels <= 4:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        else:
            # Use skimage.transform.resize for images with more than 4 channels
            img = resize(img, (img_size, img_size, num_channels), anti_aliasing=True, preserve_range=True)

        # Handle image dimensions
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # Add channel dimension
        else:
            img = img.transpose((2, 0, 1))  # Transpose to (channels, height, width)

        img = np.ascontiguousarray(img)
        return img

    elif ext in ['.png', '.jpg', '.jpeg']:
        img = cv2.imread(str(filename), cv2.IMREAD_COLOR)  # BGR format
        if img is None:
            raise FileNotFoundError(f"Unable to read image file: {filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return img

    else:
        raise ValueError(f'Unsupported image format: {ext}')

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.model == "msnet":
        net = MSNet(n_classes=args.n_classes)
    elif args.model == "rtfnet":
        net = RTFNet(n_classes=args.n_classes)
    elif args.model == "unet":
        net = UNet(n_channels=args.num_channels, n_classes=args.n_classes, bilinear=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}")
    logging.info(f"Using device {device}")

    # Load model and mask_values
    net.to(device=device)
    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        mask_values = checkpoint.get('mask_values', [(0, 0, 0), (255, 255, 255)])
    else:
        net.load_state_dict(checkpoint)
        mask_values = checkpoint.get('mask_values', [(0, 0, 0), (255, 255, 255)])  # Default values

    logging.info("Model loaded!")

    if args.input_folder != '':
        # Collect input image paths
        img_extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
        inp_paths = []
        for ext in img_extensions:
            inp_paths.extend(glob.glob(os.path.join(args.input_folder, ext)))

        if not inp_paths:
            logging.error(f"No images found in input folder {args.input_folder}")
            exit(1)

        for i, filename in enumerate(inp_paths):
            logging.info(f'Predicting image {filename} ...')
            img = load_image(filename, img_size=args.img_size)

            # Image has the expected number of channels
            if img.shape[0] != args.num_channels:
                raise ValueError(f"Expected image with {args.num_channels} channels, but got {img.shape[0]} channels.")

            # Predict the mask
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            name = os.path.splitext(os.path.basename(filename))[0]
            out_filename = os.path.join(args.output_folder, name + '_output.png')
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

    elif args.input is not None:
        for i, filename in enumerate(args.input):
            logging.info(f'Predicting image {filename} ...')
            img = load_image(filename, img_size=args.img_size)

            if img.shape[0] != args.num_channels:
                raise ValueError(f"Expected image with {args.num_channels} channels, but got {img.shape[0]} channels.")

            # Predict the mask
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            name = os.path.splitext(os.path.basename(filename))[0]
            out_filename = os.path.join(args.output_folder, name + '_output.png')
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
    else:
        logging.error("No input images provided. Use --input or --input_folder.")
        exit(1)
