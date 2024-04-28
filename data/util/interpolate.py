import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from osgeo import gdal
from pykrige.ok import OrdinaryKriging

def interpolate_arr(heightmap, footprint, algorithm):
    heightmap_masked = np.where(footprint > 0, heightmap, 0)

    if algorithm in ["ns", "telea"]:
        # Create an inpainting mask, indicating missing values
        missing_values_mask = np.where((heightmap_masked == 0) & (footprint > 0), 255, 0).astype(np.uint8)

        # Perform inpainting to fill in the missing values
        cv_mode = {"ns": cv2.INPAINT_NS, "telea": cv2.INPAINT_TELEA}
        heightmap_interpolated = cv2.inpaint(heightmap_masked, missing_values_mask, inpaintRadius=3, flags=cv_mode[algorithm])
    else:
        # Positions of zero and nonzero elements in the heightmap
        zero_positions = np.where((heightmap_masked == 0) & (footprint > 0))
        nonzero_positions = np.where(heightmap_masked > 0)

        # Form the points to search from
        points_to_search_from = np.array(nonzero_positions).T

        # Create KDTree for efficient nearest neighbor search
        kdtree = KDTree(points_to_search_from)

        # Form the points to search
        points_to_search = np.array(zero_positions).T

        # # Find indices of 3 nearest neighbors
        # dist, indices = kdtree.query(points_to_search, k=3)
        # print(len(indices))

        # # Compute mean of 3 nearest non-zero neighbors
        # interpolated_values = np.mean(heightmap_masked[nonzero_positions][indices], axis=1)

        # Find indices of 3 nearest neighbors
        dist, indices = kdtree.query(points_to_search, k=3)
            
        # Compute mean of available nearest non-zero neighbors
        interpolated_values = []

        for index_set, distance_set in zip(indices, dist):
            # Filter out the indices that are not valid neighbors (distance = inf)
            valid_indices = index_set[distance_set != np.inf]

            if valid_indices.size > 0:
                # Compute mean of the available neighbors
                mean_value = np.mean(heightmap_masked[nonzero_positions][valid_indices])
                interpolated_values.append(mean_value)
            else:
                # No neighbors found, handle this case (e.g. append NaN or a placeholder value)
                interpolated_values.append(0)

        interpolated_values = np.array(interpolated_values)


        # Create a new heightmap that includes the interpolated values
        heightmap_interpolated = np.copy(heightmap_masked)
        heightmap_interpolated[zero_positions] = interpolated_values

    return heightmap_interpolated

def kriging(input_array):
    # Find the coordinates of the missing values and the known values
    mask = input_array > 0
    nan_indices = np.argwhere(~mask)
    known_indices = np.argwhere(mask)

    # Prepare input for Kriging
    known_x = known_indices[:, 1]
    known_y = known_indices[:, 0]
    known_vals = input_array[mask]

    nan_x = nan_indices[:, 1]
    nan_y = nan_indices[:, 0]

    # Perform Kriging interpolation
    OK = OrdinaryKriging(
        known_x, known_y, known_vals, variogram_model="linear", verbose=False,
        enable_plotting=False
    )

    # The interpolated values will be stored in 'zvalues'
    zvalues, _ = OK.execute("grid", nan_x.astype(np.float64), nan_y.astype(np.float64), backend='loop', n_closest_points=2)

    # Update the original array with interpolated values
    output_array = np.copy(input_array)
    for val, (xi, yi) in zip(zvalues, nan_indices):
        output_array[xi, yi] = val

    return output_array

def idw(input_array, footprint):

    # Set pixel to be inpainted to -1
    # inpaint_mask = 1 - np.uint16((input_array == 0) & (footprint > 0))
    inpaint_mask = np.uint16(footprint == 0)
    # input_array[inpaint_mask] = 65535

    # Create an in-memory GDAL dataset from the NumPy array
    mem_driver = gdal.GetDriverByName('MEM')
    ds = mem_driver.Create('', input_array.shape[1], input_array.shape[0], 2, gdal.GDT_UInt16)
    ds.GetRasterBand(1).WriteArray(input_array)
    ds.GetRasterBand(2).WriteArray(inpaint_mask)

    # # Set "No Data" value
    # no_data_value = 65535  # Replace this with your specific "no data" value
    # band = ds.GetRasterBand(1)
    # band.SetNoDataValue(no_data_value)
    # band.FlushCache()

    # Create an in-memory dataset for the filled data
    filled_ds = mem_driver.CreateCopy("", ds)
    
    # Fill "no data" areas
    gdal.FillNodata(targetBand=filled_ds.GetRasterBand(1), maskBand=filled_ds.GetRasterBand(2), maxSearchDist=100, smoothingIterations=0)

    # Read filled data back to a NumPy array
    filled_array = filled_ds.GetRasterBand(1).ReadAsArray()

    # return filled_array * footprint
    return filled_array


def scipy_interpolate(img, footprint, algorithm):
    """
    Interpolate the non-zero values of the image within the given footprint using cubic interpolation,
    and fill in the remaining zero values using the nearest method.

    Parameters:
        img - input image
        footprint - boolean mask for the region to be processed

    Returns:
        interpolated image
    """
    assert algorithm in ['cubic', 'linear', 'nearest']
    # Get the coordinates of the non-zero values and the values themselves
    nonzero_y, nonzero_x = np.where((img != 0) & footprint)
    nonzero_values = img[nonzero_y, nonzero_x]

    # Get the coordinates within the footprint
    footprint_y, footprint_x = np.where(footprint)

    # # Perform interpolation within the convex hull using the cubic method
    output_img = np.copy(img)
    if len(nonzero_values) >= 4:
    # Use cubic interpolation if there are enough points
        interpolated = griddata((nonzero_y, nonzero_x), nonzero_values, (footprint_y, footprint_x), method=algorithm)
        for (f_y, f_x, value) in zip(footprint_y, footprint_x, interpolated):
            if not np.isnan(value):
                output_img[f_y, f_x] = value
        
        # interpolated[interpolated < 0] = 0

        # # extrapolation
        # if algorithm in ['cubic', 'linear']:
        #     nonzero_y, nonzero_x = np.where((output_img != 0) & footprint)
        #     nonzero_values = output_img[nonzero_y, nonzero_x]
        #     interpolated = griddata((nonzero_y, nonzero_x), nonzero_values, (footprint_y, footprint_x), method="nearest")
        #     for (f_y, f_x, value) in zip(footprint_y, footprint_x, interpolated):
        #         if not np.isnan(value):
        #             output_img[f_y, f_x] = value
        output_img[output_img < 0] = 0
    else:
        interpolated = output_img
# else:
# # Fallback to nearest interpolation if not enough points for cubic
# interpolated = griddata((nonzero_y, nonzero_x), nonzero_values, (footprint_y, footprint_x), method='linear')

# Create the output image and populate with cubic interpolation



    # Get the coordinates of the zero values within the footprint
    # zero_y, zero_x = np.where((output_img == 0) & footprint)

    # # Perform interpolation for the zero values using the nearest method
    # nonzero_y, nonzero_x = np.where((output_img != 0) & footprint)
    # nonzero_values = output_img[nonzero_y, nonzero_x]
    # img_interpolated_nearest = griddata((nonzero_y, nonzero_x), nonzero_values, (zero_y, zero_x), method='nearest')

    # # Update the output image with the values interpolated by the nearest method
    # for (z_y, z_x, value) in zip(zero_y, zero_x, img_interpolated_nearest):
    #     output_img[z_y, z_x] = value
    
    # output_img = np.nan_to_num(output_img)
    # output_img = np.clip(output_img, 0, 255)

    return output_img