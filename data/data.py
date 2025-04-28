import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from glob import glob
import logging
from torchvision import transforms

class NuInsSegDataset(Dataset):
    def __init__(self, data_dir, organ_names=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.bbox_prompts = []

        if organ_names is None:
            organ_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

        for organ in organ_names:
            img_dir = os.path.join(data_dir, organ, "tissue images")
            mask_dir = os.path.join(data_dir, organ, "label masks modify")
            images = sorted(glob(os.path.join(img_dir, "*.png")))
            masks = sorted(glob(os.path.join(mask_dir, "*.tif")))
            assert len(images) == len(masks), f"Mismatch in {organ}: {len(images)} images, {len(masks)} masks"

            for img_path, mask_path in zip(images, masks):
                mask = np.array(Image.open(mask_path))
                from skimage.measure import label, regionprops
                labeled_mask = label(mask > 0)
                regions = regionprops(labeled_mask)
                bboxes = [r.bbox for r in regions]
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.bbox_prompts.append(bboxes)

        logging.info(f"Dataset initialized with {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        bboxes = self.bbox_prompts[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        orig_w, orig_h = image.size
        resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR)
        resize_mask = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        image = resize(image)
        mask = resize_mask(mask)

        image = np.array(image)
        mask = np.array(mask)

        scale_x = 1024 / orig_w
        scale_y = 1024 / orig_h
        bboxes = [[b[1] * scale_x, b[0] * scale_y, b[3] * scale_x, b[2] * scale_y] for b in bboxes]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        return image, mask, bboxes

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    bboxes = [item[2] for item in batch]
    return images, masks, bboxes

def get_dataloaders(dataset, train_idx, val_idx, batch_size, num_workers=0):
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    return train_loader, val_loader