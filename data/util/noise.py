import torch
import random

def add_gauss_noise(
    img: torch.tensor,
    min_sigma: float,
    max_sigma: float
) -> torch.tensor:
    mask = img != -1
    noise = torch.randn_like(img) * random.uniform(min_sigma, max_sigma)
    noisy_img = img + noise * mask.float()
    noisy_img = torch.clamp(noisy_img, -1, 1)
    return noisy_img

def add_outlier_noise(
    img: torch.tensor,
    outlier_perc: float
) -> torch.tensor:
    assert 0 <= outlier_perc <= 100
    c, h, w = img.shape
    flatten_img = img.view(-1)
    num_outlier_pixels = int(h * w * (outlier_perc / 100))
    random_ids = torch.randint(0, h * w, (num_outlier_pixels,))
    random_vals = 2 * torch.rand(num_outlier_pixels) - 1
    flatten_img[random_ids] = random_vals
    flatten_img = torch.clamp(flatten_img, -1, 1)
    return flatten_img.view(c, h, w)