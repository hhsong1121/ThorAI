import os
import glob
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD,
    ScaleIntensityD, ResizeD, RandFlipD
)
from monai.data import Dataset, DataLoader


def create_data_dicts(image_dir, mask_dir):
    print("Pairing NIH images with CheXmask masks...")
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    image_paths += glob.glob(os.path.join(image_dir, "*.jpg"))

    data_dicts = []
    match_count = 0

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, base_name)
        if os.path.exists(mask_path):
            data_dicts.append({"image": img_path, "label": mask_path})
            match_count += 1

    print(f"Matched {match_count} image/mask pairs.")
    return data_dicts


nih_base = os.getenv("NIH_CXR_BASE", "/path/to/NIH_CXR")
chexmask_base = os.getenv("CHEXMASK_BASE", "/path/to/CheXmask")

ORIGINAL_IMAGES_DIR = os.getenv("NIH_CXR_DIR", os.path.join(nih_base, "images"))
CHEXMASK_DIR = os.getenv("CHEXMASK_DIR", os.path.join(chexmask_base, "Preprocessed"))

train_files = create_data_dicts(ORIGINAL_IMAGES_DIR, CHEXMASK_DIR)

if len(train_files) > 0:
    train_transforms = Compose([
        LoadImageD(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        ScaleIntensityD(keys=["image"]),
        ResizeD(keys=["image", "label"], spatial_size=(224, 224)),
        RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    check_data = next(iter(train_loader))
    print(f"Batch loaded: image {check_data['image'].shape}, mask {check_data['label'].shape}")
else:
    print("No matched files found. Set NIH_CXR_DIR and CHEXMASK_DIR environment variables.")
