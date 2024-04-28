from typing import Dict, List

import numpy as np
import torch
import random
from scipy.ndimage import zoom, rotate
from PIL import Image

def random_height_scaling(height: torch.tensor, footprint: torch.tensor, probability: float = 0.5, scale_factor: float = 0.4):
    assert 0 <= probability <= 1
    assert -1.0001 <= torch.min(height) and torch.max(height) <= 1.0001

    if random.random() > probability:
        return height
    
    mask = footprint > 0

    min_h, max_h = torch.min(height[mask]), torch.max(height[mask])

    # Make sure after scaling min max pixel values still in [-1,1]
    max_scale = min(1 / (max(-min_h, max_h) + 1e-6), 1+scale_factor)
    
    scale = random.uniform(1-scale_factor, max_scale)
    height[mask] *= scale

    assert -1.0001 <= torch.min(height) and torch.max(height) <= 1.0001

    return height


def random_height_shifting(height: torch.tensor, footprint: torch.tensor, probability: float = 0.5):
    assert 0 <= probability <= 1
    assert -1.0001 <= torch.min(height) and torch.max(height) <= 1.0001

    if random.random() > probability:
        return height
    
    mask = footprint > 0

    min_h, max_h = torch.min(height[mask]), torch.max(height[mask])

    # Make sure after offseting pixel values will not exceed [-1, 1]
    height[mask] += random.uniform(-1-min_h, 1-max_h)

    assert -1.0001 <= torch.min(height) and torch.max(height) <= 1.0001

    return height

def add_tree_noise(
    img: torch.tensor, 
    footprint: torch.tensor, 
    p: Dict[str, float], 
    trees: List[str],
):
    if random.random() > p['probability']:
        return img
    
    # num of tree for one building
    num_tree_to_plant = random.randint(p["min_tree_count"], p["max_tree_count"])
    
    img = (img + 1) / 2
    
    for _ in range(num_tree_to_plant):
        for _ in range(100):
            # Randomly choose a tree
            tree_id = random.randint(0, len(trees)-1)
            tree_img = np.array(Image.open(trees[tree_id]))

            # adjust z scale randomly
            tree_mask = tree_img > 2
            tree_img = (tree_img - np.min(tree_img)) / (np.max(tree_img) - np.min(tree_img))
            tree_img[tree_mask] *= random.uniform(p['min_height_scale'], p['max_height_scale'])

            # move the tree randomly in vertical direction

            # from 1 (max) to the min height of roof
            if not torch.any(img > 0.001):
                return img
            min_roof_height = torch.min(img[img > 0.001])
            tree_top_height = random.uniform(1, min_roof_height).item()

            # adjust tree height not to exceed 1
            tree_img[tree_mask] -= (np.max(tree_img[tree_mask]) - tree_top_height)
            tree_img[~tree_mask] = 0

            # adjust tree coverage area randomly
            h_scale = random.uniform(p['min_xy_scale'], p['max_xy_scale'])
            w_scale = h_scale + random.uniform(-p['max_xy_scale_diff'], p['max_xy_scale_diff'])
            tree_img = zoom(tree_img, (h_scale, w_scale))

            # rotate the tree randomly
            angle = random.uniform(0, 360)
            tree_img = rotate(tree_img, angle, reshape=True, cval=-1)
            tree_img[tree_img < 0] = 0

            # randomly select a coordinate for placing the center of tree in roof image
            non_footprint_coords = np.column_stack(np.where(footprint==0))
            tree_coord = random.choice(non_footprint_coords)[1:]
            
            y0_raw = tree_coord[0] - tree_img.shape[0] // 2
            x0_raw = tree_coord[1] - tree_img.shape[1] // 2
            y1_raw = y0_raw + tree_img.shape[0]
            x1_raw = x0_raw + tree_img.shape[1]

            y0 = max(y0_raw, 0)
            x0 = max(x0_raw, 0)
            y1 = min(y1_raw, img.shape[1])
            x1 = min(x1_raw, img.shape[2])

            t_y0 = max(0, -y0_raw)
            t_x0 = max(0, -x0_raw)
            t_y1 = t_y0 + (y1 - y0)
            t_x1 = t_x0 + (x1 - x0)
            
            # tree_img[t_y0:t_y1,t_x0:t_x1] *= footprint[0,y0:y1,x0:x1]

            num_replaced_pixels = np.sum(tree_img[t_y0:t_y1, t_x0:t_x1] > img[0, y0:y1, x0:x1].numpy())
            
            if num_replaced_pixels < 20:
                continue

            img[0,y0:y1,x0:x1] = np.maximum(img[0,y0:y1,x0:x1], tree_img[t_y0:t_y1,t_x0:t_x1])

            break
    
    # img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    # viz_img = img.numpy()[0] * 255
    # viz_img = viz_img.astype(np.uint8)
    
    # from PIL import Image
    # Image.fromarray(viz_img).save(f'./debug/building_tree.png')

    return (img * 2) - 1