import numpy as np
from PIL import Image
from torchvision import transforms


class FFTHighPass:
    def __init__(self, radius_ratio: float = 0.08):
        self.radius_ratio = radius_ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width, _ = array.shape
        center_y, center_x = height // 2, width // 2
        radius = max(1, int(min(height, width) * self.radius_ratio))

        y_grid, x_grid = np.ogrid[:height, :width]
        low_frequency_mask = (
            (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2
        ) <= radius**2

        filtered_channels = []
        for channel_index in range(3):
            channel = array[:, :, channel_index]
            spectrum = np.fft.fftshift(np.fft.fft2(channel))
            spectrum[low_frequency_mask] = 0
            high_pass = np.fft.ifft2(np.fft.ifftshift(spectrum))
            high_pass = np.abs(high_pass)
            high_pass = high_pass - high_pass.min()
            max_value = high_pass.max()
            if max_value > 0:
                high_pass = high_pass / max_value
            filtered_channels.append(high_pass)

        output = np.stack(filtered_channels, axis=2)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")


def get_transforms(transform_mode: str = "rgb"):
    if transform_mode not in {"rgb", "fft_highpass"}:
        raise ValueError("transform_mode must be 'rgb' or 'fft_highpass'")

    frequency_transform = []
    if transform_mode == "fft_highpass":
        frequency_transform = [FFTHighPass()]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        *frequency_transform,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        *frequency_transform,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform
