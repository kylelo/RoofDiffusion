import os
import argparse
import numpy as np
from PIL import Image

def calculate_metrics(gt_image, pred_image, footprint_image, use_valid_gt=False, use_footprint=True):
    if use_valid_gt:
        mask = np.array(gt_image) > 0
    elif use_footprint:
        mask = np.array(footprint_image) > 0
    else:
        mask = np.ones_like(gt_image) > 0

    # convert to meters
    gt_pixels = np.array(gt_image)[mask] / 256.0
    pred_pixels = np.array(pred_image)[mask] / 256.0

    rmse = np.sqrt(np.mean((gt_pixels - pred_pixels) ** 2))
    mae = np.mean(np.abs(gt_pixels - pred_pixels))

    return rmse, mae

def IoU(pred_image, footprint_image):
    pred_image = pred_image / 256.0
    pred_binary = pred_image > 0

    footprint_binary = footprint_image > 0
    n_intersect = np.sum(pred_binary * footprint_binary)
    n_union = np.sum(np.logical_or(pred_binary, footprint_binary).astype(int))
    return n_intersect / n_union


def main():
    parser = argparse.ArgumentParser(description='Calculate RMSE and MAE for image sets.')
    parser.add_argument('--gt_dir', default='/home/kylelo/research/Meta/data/GT/poznan_new/benchmark/s95_i50/roof_gt', help='Path to the folder containing ground truth images.')
    parser.add_argument('--pred_dir', default='/home/kylelo/research/Palette-Image-to-Image-Diffusion-Models/experiments/test_roof_completion_230923_102246/results/test/0', help='Path to the folder containing predicted images.')
    parser.add_argument('--footprint_dir', default='/home/kylelo/research/Meta/data/GT/poznan_new/benchmark/s95_i50/roof_footprint', help='Path to the folder containing footprint images.')
    parser.add_argument('--use_valid_gt', default=False, action='store_true')
    parser.add_argument('--img_name_prefix', default='BID')
    parser.add_argument('--no_footprint', action='store_true', default=False)
    args = parser.parse_args()

    filenames = [f for f in os.listdir(args.pred_dir) if (f.endswith('.png') and f.startswith(args.img_name_prefix))]
    rmse_list = []
    mae_list = []
    iou_list = []

    for fname in filenames:
        gt_path = os.path.join(args.gt_dir, fname)
        pred_path = os.path.join(args.pred_dir, fname)
        footprint_path = os.path.join(args.footprint_dir, fname.replace('.png', '_footprint.png'))

        if (not os.path.exists(gt_path)) or (not os.path.exists(pred_path)) or (not os.path.exists(footprint_path)):
            continue
        
        # Read the corresponding files using PIL to preserve raw values
        gt_image = Image.open(gt_path)
        pred_image = Image.open(pred_path)
        footprint_image = Image.open(footprint_path)

        rmse, mae = calculate_metrics(gt_image, pred_image, footprint_image, use_valid_gt=args.use_valid_gt, use_footprint=(not args.no_footprint))
        rmse_list.append(rmse)
        mae_list.append(mae)
        iou_list.append(IoU(np.array(pred_image), np.array(footprint_image)))

    # Compute average, worst, and best RMSE and MAE
    avg_rmse = np.mean(rmse_list)
    worst_rmse = np.max(rmse_list)
    best_rmse = np.min(rmse_list)

    avg_mae = np.mean(mae_list)
    worst_mae = np.max(mae_list)
    best_mae = np.min(mae_list)

    avg_mae = np.mean(mae_list)
    worst_mae = np.max(mae_list)
    best_mae = np.min(mae_list)

    avg_iou = np.mean(iou_list)

    print(f"Average MAE: {avg_mae}")
    print(f"Worst MAE: {worst_mae}")
    print(f"Best MAE: {best_mae}")

    print(f"Average RMSE: {avg_rmse}")
    print(f"Worst RMSE: {worst_rmse}")
    print(f"Best RMSE: {best_rmse}")
    print(f"IoU = {avg_iou * 100:.2f}")
    # print(f"{avg_mae:.3f} & {avg_rmse:.3f}")




if __name__ == "__main__":
    main()
