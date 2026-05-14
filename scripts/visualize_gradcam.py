import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.dataset import get_dataloaders
from src.models.attention_resnet import AttentionResNet
from src.models.baseline_resnet import get_baseline_resnet


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_index):
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[:, class_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def build_model(model_type, model_name, checkpoint_path, device):
    if model_type == "baseline":
        model = get_baseline_resnet(model_name=model_name, num_classes=2)
        target_layer = model.layer4[-1]
    elif model_type == "eca":
        model = AttentionResNet(model_name=model_name, num_classes=2)
        target_layer = model.backbone.layer4[-1]
    else:
        raise ValueError("MODEL_TYPE must be 'baseline' or 'eca'")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, target_layer


def overlay_heatmap(image, heatmap):
    image_array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    color_map = plt.get_cmap("jet")(heatmap)[:, :, :3]
    overlay = 0.55 * image_array + 0.45 * color_map
    return np.clip(overlay, 0, 1)


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    model_name = os.environ.get("MODEL_NAME", "resnet18")
    model_type = os.environ.get("MODEL_TYPE", "eca")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/gradcam"))
    samples_per_group = int(os.environ.get("SAMPLES_PER_GROUP", 8))
    seed = int(os.environ.get("SEED", 42))

    if checkpoint_path is None:
        raise ValueError("Set CHECKPOINT_PATH to the model checkpoint to visualize.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=1,
        num_workers=0,
        transform_mode="rgb",
    )

    model, target_layer = build_model(model_type, model_name, checkpoint_path, device)
    gradcam = GradCAM(model, target_layer)

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    correct = []
    incorrect = []
    samples = test_loader.dataset.samples
    rng.shuffle(samples)

    with torch.no_grad():
        for image_path, true_index in samples:
            image = Image.open(image_path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
            output = model(tensor)
            pred_index = int(output.argmax(dim=1).item())
            row = {
                "image_path": image_path,
                "true_index": true_index,
                "pred_index": pred_index,
            }
            if pred_index == true_index and len(correct) < samples_per_group:
                correct.append(row)
            elif pred_index != true_index and len(incorrect) < samples_per_group:
                incorrect.append(row)

            if len(correct) >= samples_per_group and len(incorrect) >= samples_per_group:
                break

    selected = [("Correct", item) for item in correct] + [
        ("Incorrect", item) for item in incorrect
    ]

    columns = 4
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.6))
    axes = np.atleast_2d(axes)

    for axis in axes.flat:
        axis.axis("off")

    for index, (group_name, item) in enumerate(selected):
        axis = axes[index // columns][index % columns]
        image = Image.open(item["image_path"]).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        heatmap = gradcam(tensor, item["pred_index"])
        overlay = overlay_heatmap(image, heatmap)
        true_label = class_names[item["true_index"]]
        pred_label = class_names[item["pred_index"]]

        axis.imshow(overlay)
        axis.set_title(
            f"{group_name}\nA: {true_label} | P: {pred_label}",
            fontsize=9,
        )
        axis.axis("off")

    gradcam.close()
    plt.tight_layout()
    output_path = output_dir / f"{model_type}_{model_name}_gradcam.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved Grad-CAM grid to:", output_path)


if __name__ == "__main__":
    main()
