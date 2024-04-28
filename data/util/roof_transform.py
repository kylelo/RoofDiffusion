import torch
import cv2

class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size  # (width, height)
        
    def __call__(self, img):
        resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized_img

class HeightNormalize(object):
    def __init__(self):
        self.height_range = -1
        self.mid_height = -1
        self.valid_mask = None

    def __call__(self, height: torch.tensor, height_range: float = 10):
        """Normalize height map for diffusion model.

        Args:
            height_map (torch.tensor): the height map in meters
            height_range (float, optional): The difference between min and max roof height. Defaults to 0.
        """
        if torch.all(height==0):
            self.height_range = torch.tensor(height_range, dtype=torch.float32)
            self.mid_height = torch.tensor(0, dtype=torch.float32)
            self.valid_mask = height
            return height


        mask = height > 0

        max_h, min_h = torch.max(height[mask]), torch.min(height[mask])

        mid_h = (max_h + min_h) / 2
        diff_h = max_h - min_h

        # offset lowest roof height to have center at zero
        height -= mid_h

        # Adjust to [0, height_range] if roof exceed height_range
        if diff_h > height_range:
            height_range = diff_h + 1e-8
        
        # scale to [-0.5, 0.5]
        height /= height_range 

        # scale to [-1, 1]
        height *= 2

        # set invalid pixel to -1 
        height[~mask] = -1

        self.height_range = torch.tensor(height_range, dtype=torch.float32)
        self.mid_height = mid_h
        self.valid_mask = mask

        return height