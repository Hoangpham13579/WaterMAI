import logging
import numpy as np
import torch
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from skimage.transform import resize
import cv2
from tqdm import tqdm
import tifffile as tiff

def unique_mask_values(idx, mask_dir, mask_suffix, img_size):
    idx = idx.strip()
    idx = idx.replace('./', '')
    idx = Path(idx).stem 

    mask_files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    if not mask_files:
        raise FileNotFoundError(f"No mask file found for ID {idx} with pattern {idx + mask_suffix + '.*'} in {mask_dir}")
    mask_file = mask_files[0]
    mask = np.asarray(load_mask(mask_file, img_size))
    if mask.ndim == 2:
        unique_vals = np.unique(mask)
    else:
        unique_vals = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    return unique_vals

def load_image(filename, img_size):
    ext = splitext(str(filename))[1].lower()
    if ext in ['.tiff', '.tif']:
        img = tiff.imread(str(filename))
        img = np.array(img, dtype=np.float32)
        if img.max() > 1:
            img /= 255.0

        # Determine the number of channels
        num_channels = img.shape[2] if img.ndim == 3 else 1

        if num_channels <= 4:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        else:
            img = resize(img, (img_size, img_size, num_channels), anti_aliasing=True, preserve_range=True)
        
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # Add channel dimension
        else:
            img = img.transpose((2, 0, 1))  # Transpose to (channels, height, width)
        img = np.ascontiguousarray(img)
        return img

    elif ext in ['.png', '.jpg', '.jpeg']:
        img = cv2.imread(str(filename), cv2.IMREAD_COLOR)  # BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return img

    else:
        raise ValueError(f'Unsupported image format: {ext}')

def load_mask(filename, img_size):
    ext = splitext(str(filename))[1].lower()
    if ext in ['.png', '.jpg', '.jpeg']:
        mask = cv2.imread(str(filename), cv2.IMREAD_COLOR)  # Load as BGR
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return mask

    elif ext in ['.tiff', '.tif']:
        mask = tiff.imread(str(filename))
        mask = np.array(mask, dtype=np.float32)
        
        # Determine the number of channels
        num_channels = mask.shape[2] if mask.ndim == 3 else 1

        if num_channels <= 4:
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = resize(mask, (img_size, img_size, num_channels), order=0, preserve_range=True, anti_aliasing=False)
        
        if mask.ndim == 2:
            mask = np.stack((mask,) * 3, axis=-1)
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask = np.concatenate([mask] * 3, axis=2)
        elif mask.ndim == 3 and mask.shape[2] != 3:
            mask = mask[:, :, :3]
        return mask

    else:
        raise ValueError(f'Unsupported mask format {ext}')

def convert_mask_to_class_labels(mask, mask_values):
    #Convert a mask image to class labels based on mask_values.
    mask_shape = mask.shape[:2]
    label_mask = np.full(mask_shape, fill_value=-1, dtype=np.int64)

    if mask.ndim == 2:
        raise ValueError("Expected 3D mask, but got 2D mask.")
    else:
        mask_flat = mask.reshape(-1, mask.shape[2])
        label_mask_flat = label_mask.reshape(-1)
        for i, v in enumerate(mask_values):
            v = np.array(v)
            matches = np.all(mask_flat == v, axis=1)
            label_mask_flat[matches] = i
        label_mask = label_mask_flat.reshape(mask_shape)

    # After mapping
    unmapped = (label_mask == -1)
    if np.any(unmapped):
        unmapped_pixels = mask.reshape(-1, mask.shape[2])[unmapped.flatten()]
        unique_unmapped = np.unique(unmapped_pixels, axis=0)
        logging.error(f"Found unmapped pixel values in mask: {unique_unmapped.tolist()}")
        raise ValueError(f"Unmapped pixel values found in mask.")

    return label_mask

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, img_size=640, mask_suffix: str = '', ids=None, mask_values=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.scale = scale
        self.mask_suffix = mask_suffix
        if ids is not None:
            self.ids = [self.process_id(id) for id in ids]
        else:
            self.ids = [splitext(file)[0] for file in listdir(self.images_dir)
                        if isfile(join(self.images_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        if mask_values is not None:
            self.mask_values = mask_values
            logging.info(f'Using provided mask values: {self.mask_values}')
        else:
            logging.info('Scanning mask files to determine unique values')
            unique = []
            for id in tqdm(self.ids):
                unique.append(unique_mask_values(id, self.mask_dir, self.mask_suffix, self.img_size))

            self.mask_values = list(map(tuple, sorted(np.unique(np.concatenate(unique), axis=0).tolist())))
            logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    def process_id(self, id_str):
        id_str = id_str.strip()
        id_str = id_str.replace('./', '')  
        id_str = Path(id_str).stem  
        return id_str

    @staticmethod
    def preprocess(img, is_mask):
        if is_mask:
            return img
        else:
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        # Find the image and mask files
        img_files = list(self.images_dir.glob(name + '.*'))
        mask_files = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))

        assert len(img_files) > 0, f'No image found for ID {name} in {self.images_dir}'
        assert len(mask_files) > 0, f'No mask found for ID {name} in {self.mask_dir}'

        img_file = img_files[0]
        mask_file = mask_files[0]

        img = load_image(img_file, self.img_size)
        mask = load_mask(mask_file, self.img_size)

        assert img.shape[1:] == mask.shape[:2], \
            f'Image and mask {name} should be the same size, but are {img.shape[1:]} and {mask.shape[:2]}'

        mask = convert_mask_to_class_labels(mask, self.mask_values)

        unique_labels = np.unique(mask)
        logging.debug(f"Unique labels in mask {name}: {unique_labels}")
        if not np.all((unique_labels >= 0) & (unique_labels < len(self.mask_values))):
            invalid_labels = unique_labels[(unique_labels < 0) | (unique_labels >= len(self.mask_values))]
            logging.error(f"Invalid labels in mask {name}: {invalid_labels}")
            raise ValueError(f"Invalid labels in mask {name}: {invalid_labels}")

        img = self.preprocess(img, is_mask=False)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def get_mask_values(self):
        """Return the mask values (colors) used for class labels."""
        return self.mask_values