import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torch import autocast, GradScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class VinDrCXRDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_ids = self.df['image_id'].unique()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.png")
        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.functional.to_tensor(img)

        records = self.df[self.df['image_id'] == image_id]
        boxes = []
        labels = []

        for _, row in records.iterrows():
            if row['class_id'] == 14:
                continue

            x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(row['class_id'] + 1)

        if len(boxes) == 0:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            }
            return img, target

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.image_ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    print("VinDr-CXR object detection fine-tuning")

    csv_path = os.getenv("VINDR_CSV_PATH", "../utils_data/master_vindr_labels.csv")
    img_dir = os.getenv("VINDR_IMG_DIR", "./vinbigdata/train")
    checkpoint_dir = os.getenv("VINDR_CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = VinDrCXRDataset(csv_path, img_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=int(os.getenv("VINDR_BATCH_SIZE", "16")),
        shuffle=True,
        num_workers=int(os.getenv("VINDR_NUM_WORKERS", "4")),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    model = get_model(num_classes=15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scaler = GradScaler()
    num_epochs = int(os.getenv("VINDR_EPOCHS", "10"))

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for imgs, targets in data_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - avg loss: {epoch_loss / len(data_loader):.4f}")
        save_path = os.path.join(checkpoint_dir, f"vindr_det_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)

    print("Training complete.")
