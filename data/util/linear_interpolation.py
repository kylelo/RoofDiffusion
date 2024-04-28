import cv2
import numpy as np
import os
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree

def linear_interpolation(image, mask):
    zero_coords = np.argwhere(mask == 0)
    non_zero_coords = np.argwhere(mask != 0)
    non_zero_values = image[non_zero_coords[:, 0], non_zero_coords[:, 1]]

    # First pass: Perform linear interpolation
    interpolated_values = griddata(non_zero_coords, non_zero_values, zero_coords, method='linear')

    # Update the image and mask with first-pass interpolated values
    for idx, coord in enumerate(zero_coords):
        x, y = coord[0], coord[1]
        interpolated_value = np.nan_to_num(interpolated_values[idx], nan=0)
        image[x, y] = interpolated_value.astype(np.uint16)
        if interpolated_value != 0:
            mask[x, y] = 1  # Update the mask

    # Second pass: Use both original and first-pass interpolated non-zero pixels to fill remaining zeros
    zero_coords = np.argwhere(mask == 0)  # Update zero_coords based on the new mask
    non_zero_coords = np.argwhere(mask != 0)  # Update non_zero_coords based on the new mask
    non_zero_values = image[non_zero_coords[:, 0], non_zero_coords[:, 1]]

    # Build a KDTree for fast nearest-neighbor lookup
    kdtree = cKDTree(non_zero_coords)
    
    # Find distances and indices of the nearest non-zero pixels for each zero pixel
    _, indices = kdtree.query(zero_coords, k=1)

    for idx, coord in enumerate(zero_coords):
        x, y = coord[0], coord[1]
        nearest_idx = indices[idx]
        image[x, y] = non_zero_values[nearest_idx].astype(np.uint16)

    return image
    
def process_file(filename, input_folder, output_folder):
    print(f"Processing {filename}...")
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    depth_map = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    zero_mask = (depth_map != 0).astype(np.uint8)

    interpolated_depth_map = linear_interpolation(depth_map, zero_mask)
    cv2.imwrite(output_path, interpolated_depth_map.astype(np.uint16))

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ProcessPoolExecutor() as executor:
        filenames = [f for f in os.listdir(input_folder) if f.endswith(".png")]
        executor.map(process_file, filenames, [input_folder]*len(filenames), [output_folder]*len(filenames))


if __name__ == "__main__":
    input_folder = "/home/kylelo/research/Meta/data/KITTI/kitti_depth/train/2011_09_26_drive_0028_sync/proj_depth/groundtruth/image_02"  # Replace with the path to your input folder
    output_folder = "/home/kylelo/research/Meta/data/interpolation_test"  # Replace with the path to your output folder

    process_folder(input_folder, output_folder)
