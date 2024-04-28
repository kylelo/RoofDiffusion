import os
import numpy as np
from PIL import Image
from data.dataset import RoofDataset
from torch.utils.data import DataLoader

################# Parameters #################
sig=[[95,30,5],[95,80,5]] # [[sparsity (percentage), inpcompleteness (local removal percentage)], ...]
DATA_ROOT = "./dataset/PoznanRD/test_img.flist"
FOOTPRINT_ROOT = "./dataset/PoznanRD/test_footprint.flist"
OUTPUT_ROOT = "./dataset/PoznanRD/benchmark/custom"
MAX_GAUSS_NOISE_SIGMA = 0.05
OUTLIER_NOISE_PERCENTAGE = 0.01
TREE_PROBABILITY = 0.3
MIN_TREE_COUNT = 1
MAX_TREE_COUNT = 3
TOTAL_DATA = 1000
MIN_LOCAL_REMOVE_SIGMA = 0.1
MAX_LOCAL_REMOVE_SIGMA = 0.12
USE_FOOTPRINT = True
##############################################

for s, i, g in sig:

    MASK_CONFIG = {
        "down_res_pct": [s],
        "local_remove": [[MIN_LOCAL_REMOVE_SIGMA, MAX_LOCAL_REMOVE_SIGMA, g]],
        "local_remove_percentage": i
    }
    NOISE_CONFIG = {
        "min_gauss_noise_sigma": 0,
        "max_gauss_noise_sigma": MAX_GAUSS_NOISE_SIGMA,
        "outlier_noise_percentage": OUTLIER_NOISE_PERCENTAGE
    }
    DATA_AUG = {
        "rotate": 0,
        "tree": {
            "flist_path": "/home/kylelo/research/Meta/data/tree/test_tree.flist",
            "probability": TREE_PROBABILITY,
            "use_rotate": True,

            "min_tree_count": MIN_TREE_COUNT,
            "max_tree_count": MAX_TREE_COUNT,

            "min_xy_scale": 1.0,
            "max_xy_scale": 3.0,
            "max_xy_scale_diff": 0.1,

            "min_height_scale": 1.0,
            "max_height_scale": 2.0
        }
    }
    FOOTPRINT_AS_MASK = True
    RECOVER_REAL_HEIGHT = True

    ##########################################################################

    roof_dataset = RoofDataset(
        data_root=DATA_ROOT,
        footprint_root=FOOTPRINT_ROOT,
        mask_config=MASK_CONFIG,
        noise_config=NOISE_CONFIG,
        data_aug=DATA_AUG,
        footprint_as_mask=FOOTPRINT_AS_MASK,
        use_footprint=USE_FOOTPRINT,
        recover_real_height=RECOVER_REAL_HEIGHT
    )

    roof_loader = DataLoader(roof_dataset, batch_size=1, shuffle=False)

    # Output directory for saving images
    OUTPUT_DIR = f"{OUTPUT_ROOT}/s{s}_i{i}"
    img_dir = os.path.join(OUTPUT_DIR, 'roof_img')
    footprint_dir = os.path.join(OUTPUT_DIR, 'roof_footprint')
    gt_dir = os.path.join(OUTPUT_DIR, 'roof_gt')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(footprint_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    img_flist = []
    footprint_flist = []
    gt_flist = []

    # Iterate through the dataset and save PNG files
    num_collected = 0
    for i, data_dict in enumerate(roof_loader):
        cond_img = data_dict['cond_image'][0][0]
        gt_img = data_dict['gt_image'][0][0]
        mask = data_dict['mask'][0][0]
        basename = os.path.basename(data_dict['path'][0])
        filename = os.path.splitext(basename)[0]

        img_path = os.path.join(img_dir, f"{filename}.png")
        gt_path = os.path.join(gt_dir, f"{filename}.png")
        footprint_path = os.path.join(footprint_dir, f"{filename}_footprint.png")

        cond_img[cond_img < 0] = 0
        gt_img[gt_img < 0] = 0

        cond_img = cond_img.numpy().astype('uint16')
        gt_img = gt_img.numpy().astype('uint16')

        if np.sum(cond_img) == 0 or np.sum(gt_img) == 0:
            continue

        # Save images as PNG
        Image.fromarray(cond_img).save(img_path)
        Image.fromarray(gt_img).save(gt_path)
        Image.fromarray(mask.numpy().astype('uint8')).save(footprint_path)

        img_flist.append(img_path)
        footprint_flist.append(footprint_path)
        gt_flist.append(gt_path)

        num_collected += 1
        if num_collected == TOTAL_DATA:
            break

    # Save flist files
    with open(os.path.join(OUTPUT_DIR, 'img.flist'), 'w') as f:
        f.write('\n'.join(img_flist))

    with open(os.path.join(OUTPUT_DIR, 'footprint.flist'), 'w') as f:
        f.write('\n'.join(footprint_flist))

    with open(os.path.join(OUTPUT_DIR, 'gt.flist'), 'w') as f:
        f.write('\n'.join(gt_flist))

    print(f"Images generated in {OUTPUT_DIR}/")
