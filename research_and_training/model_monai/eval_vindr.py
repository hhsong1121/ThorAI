import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from train_vindr_detection import get_model, VinDrCXRDataset, collate_fn

if __name__ == "__main__":
    print("VinDr-CXR model evaluation (COCO mAP)")

    csv_path = os.getenv("VINDR_CSV_PATH", "../utils_data/master_vindr_labels.csv")
    img_dir = os.getenv("VINDR_IMG_DIR", "./vinbigdata/train")
    weights_path = os.getenv("VINDR_WEIGHTS_PATH", "./checkpoints/vindr_det_epoch_10.pth")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=15)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = VinDrCXRDataset(csv_path, img_dir)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    metric = MeanAveragePrecision(iou_type="bbox")

    print("Evaluating...")
    with torch.no_grad():
        for imgs, targets in tqdm(data_loader):
            imgs = list(img.to(device) for img in imgs)
            preds = model(imgs)
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds, targets)

    result = metric.compute()
    print("\nResults")
    print(f"mAP (Average)     : {result['map'].item():.4f}")
    print(f"mAP@0.5           : {result['map_50'].item():.4f}")
    print(f"mAP@0.75          : {result['map_75'].item():.4f}")
