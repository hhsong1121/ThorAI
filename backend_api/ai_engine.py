import io
import base64

import torch
import torchxrayvision as xrv
import skimage
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class CXREngine:
    def __init__(self):
        print("Loading CXR engine (NIH Grad-CAM)...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.nih_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device)
        self.nih_model.eval()
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(self.save_gradient)

        target_layer = self.nih_model.features.norm5
        target_layer.register_forward_hook(forward_hook)

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

            img_array_orig = np.array(img_pil)
            heatmap_resized = skimage.transform.resize(heatmap, (img_array_orig.shape[0], img_array_orig.shape[1]))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_array_orig)
            ax.imshow(heatmap_resized, cmap='jet', alpha=0.3)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

            heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return scores, heatmap_b64, []

        except Exception as e:
            print(f"CXR analysis error: {e}")
            return {"Atelectasis": 0.0, "Effusion": 0.0, "Pneumonia": 0.0, "Pneumothorax": 0.0}, "", []
