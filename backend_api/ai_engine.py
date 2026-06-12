import os
import io
import base64

import torch
import torchxrayvision as xrv
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import skimage
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from config import DEMO_MODE, VINDR_WEIGHTS_PATH


class CXREngine:
    def __init__(self):
        mode_label = "NIH only (demo)" if DEMO_MODE else "NIH + VinDr"
        print(f"Loading CXR engine ({mode_label})...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.vindr_enabled = False

        self.nih_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device)
        self.nih_model.eval()
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(self.save_gradient)

        target_layer = self.nih_model.features.norm5
        target_layer.register_forward_hook(forward_hook)

        self.vindr_model = None
        self.vindr_labels = {
            1: "Aortic enlargement", 2: "Atelectasis", 3: "Calcification", 4: "Cardiomegaly",
            5: "Consolidation", 6: "ILD", 7: "Infiltration", 8: "Lung Opacity",
            9: "Nodule/Mass", 10: "Other lesion", 11: "Pleural effusion", 12: "Pleural thickening",
            13: "Pneumothorax", 14: "Pulmonary fibrosis"
        }

        if not DEMO_MODE:
            self._load_vindr_model()

    def _load_vindr_model(self):
        self.vindr_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.vindr_model.roi_heads.box_predictor.cls_score.in_features
        self.vindr_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)

        try:
            self.vindr_model.load_state_dict(torch.load(VINDR_WEIGHTS_PATH, map_location=self.device))
            self.vindr_model.to(self.device)
            self.vindr_model.eval()
            self.vindr_enabled = True
            print("VinDr detection weights loaded.")
        except FileNotFoundError:
            print(f"VinDr weights not found: {VINDR_WEIGHTS_PATH}")
        except Exception as e:
            print(f"VinDr weight load failed: {e}")

    def save_gradient(self, grad):
        self.gradients = grad

    def analyze(self, image_bytes: bytes) -> tuple:
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            img_gray = img_pil.convert("L")
            img_array_224 = skimage.transform.resize(np.array(img_gray), (224, 224), preserve_range=True)
            norm_array = xrv.datasets.normalize(img_array_224, 255)
            img_tensor_nih = torch.from_numpy(norm_array).unsqueeze(0).unsqueeze(0).float().to(self.device)
            img_tensor_nih.requires_grad = True

            self.nih_model.zero_grad()
            outputs_nih = self.nih_model(img_tensor_nih)

            results_nih = dict(zip(self.nih_model.pathologies, outputs_nih[0].detach().cpu().numpy()))
            scores = {
                "Atelectasis": float(results_nih.get("Atelectasis", 0.0)),
                "Effusion": float(results_nih.get("Effusion", 0.0)),
                "Pneumonia": float(results_nih.get("Pneumonia", 0.0)),
                "Pneumothorax": float(results_nih.get("Pneumothorax", 0.0))
            }

            target_idx = outputs_nih.argmax(dim=1).item()
            outputs_nih[0, target_idx].backward()

            if self.gradients is not None and self.activations is not None:
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
                weighted_activations = self.activations * pooled_gradients
                heatmap = torch.mean(weighted_activations, dim=1).squeeze()
                heatmap = torch.max(heatmap, torch.zeros_like(heatmap)).detach().cpu().numpy()
                if np.max(heatmap) > 0:
                    heatmap /= np.max(heatmap)
            else:
                heatmap = np.zeros((7, 7))

            detected_items = []
            boxes_to_draw = []
            if self.vindr_enabled and self.vindr_model is not None:
                img_tensor_vindr = torchvision.transforms.functional.to_tensor(img_pil).to(self.device)
                with torch.no_grad():
                    vindr_preds = self.vindr_model([img_tensor_vindr])[0]

                for box, label, score in zip(vindr_preds['boxes'], vindr_preds['labels'], vindr_preds['scores']):
                    if score > 0.3:
                        label_name = self.vindr_labels.get(label.item(), "Unknown")
                        detected_items.append(f"{label_name} ({score:.1%})")
                        boxes_to_draw.append((box.cpu().numpy(), label_name))

            img_array_orig = np.array(img_pil)
            heatmap_resized = skimage.transform.resize(heatmap, (img_array_orig.shape[0], img_array_orig.shape[1]))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_array_orig)
            ax.imshow(heatmap_resized, cmap='jet', alpha=0.3)

            for box, label_name in boxes_to_draw:
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x_min, y_min - 10, label_name, color='red', fontsize=10, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

            heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return scores, heatmap_b64, list(set(detected_items))

        except Exception as e:
            print(f"CXR analysis error: {e}")
            return {"Atelectasis": 0.0, "Effusion": 0.0, "Pneumonia": 0.0, "Pneumothorax": 0.0}, "", []
