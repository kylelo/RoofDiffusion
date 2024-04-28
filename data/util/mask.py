# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import random

from scipy import ndimage


def random_cropping_bbox(img_shape=(256,256), mask_mode='onedirection'):
    h, w = img_shape
    if mask_mode == 'onedirection':
        _type = np.random.randint(0, 4)
        if _type == 0:
            top, left, height, width = 0, 0, h, w//2
        elif _type == 1:
            top, left, height, width = 0, 0, h//2, w
        elif _type == 2:
            top, left, height, width = h//2, 0, h//2, w
        elif _type == 3:
            top, left, height, width = 0, w//2, h, w//2
    else:
        target_area = (h*w)//2
        width = np.random.randint(target_area//h, w)
        height = target_area//width
        if h==height:
            top = 0
        else:
            top = np.random.randint(0, h-height)
        if w==width:
            left = 0
        else:
            left = np.random.randint(0, w-width)
    return (top, left, height, width)

def random_bbox(img_shape=(256,256), max_bbox_shape=(128, 128), max_bbox_delta=40, min_margin=20):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)
        
    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask


def brush_stroke_mask(img_shape,
                      num_vertices=(4, 12),
                      mean_angle=2 * math.pi / 5,
                      angle_range=2 * math.pi / 15,
                      brush_width=(12, 40),
                      max_loops=4,
                      dtype='uint8'):
    """Generate free-form mask.

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
        dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :, None]
    return mask


def random_irregular_mask(img_shape,
                          num_vertices=(4, 8),
                          max_angle=4,
                          length_range=(10, 100),
                          brush_width=(10, 40),
                          dtype='uint8'):
    """Generate random irregular masks.

    This is a modified version of free-form mask implemented in
    'brush_stroke_mask'.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 8).
        max_angle (float): Max value of angle at each vertex. Default 4.0.
        length_range (int | tuple[int]): (min_length, max_length). If only give
            an integer, we will fix the length of brush. Default: (10, 100).
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (10, 40).
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    h, w = img_shape[:2]

    mask = np.zeros((h, w), dtype=dtype)
    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        raise TypeError('The type of length_range should be int'
                        f'or tuple[int], but got type: {length_range}')
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        # from the start point, randomly setlect n \in [1, 6] directions.
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(
            min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(
            min_brush_width, max_brush_width, size=direction_num)
        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            # compute end point according to the random angle
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y
    mask = np.expand_dims(mask, axis=2)

    return mask


def get_irregular_mask(img_shape, area_ratio_range=(0.15, 0.5), **kwargs):
    """Get irregular mask with the constraints in mask ratio

    Args:
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    mask = random_irregular_mask(img_shape, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) /
                           (img_shape[0] * img_shape[1])) < max_ratio:
        mask = random_irregular_mask(img_shape, **kwargs)

    return mask

# def get_down_res_mask(footprint, missing_percentage):
#     """Note: if the percentage of missing point is not large, this function might
#     be the bottleneck for the training! 
#     """
#     # Create a copy of the footprint
#     mask = np.ones_like(footprint)
    
#     num_pixels = np.sum(footprint) # sum of all pixels that are 1
#     num_pixels_to_keep = int(num_pixels * (100 - missing_percentage) / 100)
    
#     # get coordinates of pixels that are 1
#     coords = np.argwhere(footprint)

#     for _ in range(num_pixels_to_keep):
#         # randomly choose a pixel among those that are 1
#         index = random.choice(range(len(coords)))
#         c, w, h = coords[index]
#         mask[c, w, h] = 0
#         # remove the used coordinate
#         coords = np.delete(coords, index, 0)
        
#     return mask * footprint

def get_down_res_mask(footprint, missing_percentage):
    mask = np.zeros_like(footprint)
    
    num_pixels = np.sum(footprint)
    num_pixels_to_remove = int(num_pixels * missing_percentage / 100)
    
    # get indices of pixels that are 1
    coords = np.argwhere(footprint)
    indices_to_remove = np.random.choice(coords.shape[0], num_pixels_to_remove, replace=False)
    
    mask[tuple(coords[indices_to_remove].T)] = 1

    return mask * footprint


def get_multi_gauss_mask(
    footprint: np.array,
    min_sigma_ratio: float = 0.1,
    max_sigma_ratio: float = 0.3,
    n_gauss_mask: int = 5,
    remove_percentage: float = -1,
):
    """A gradient mask inspired by Gaussian Mixture Models for synthesizing locally missing pixels.

    Args:
        mask_size (Tuple[int, int]): height width of mask
        min_sigma_ratio (float, optional): min sigma of gauss distribution. Defaults to 0.1.
        max_sigma_ratio (float, optional): max sigma of gauss distribution. Defaults to 0.3.
        n_gauss_mask (int, optional): number of guass distribution to form the final mask. Defaults to 5.
        remove_percentage (float, optional): percentage of points to be removed and -1 means no limitation.

    Returns:
        np.array: an image with value one representing the pixels to be masked out.
    """
    assert min_sigma_ratio <= max_sigma_ratio
    assert footprint.ndim == 3
    h, w = footprint.shape[1:]
    mask = np.zeros((h, w), dtype=bool)
    prob_mask = np.zeros((h, w), dtype=float)  # Keep track of cumulative probabilities

    sigma_ratio = random.uniform(min_sigma_ratio, max_sigma_ratio)
    sigma = sigma_ratio * min(h, w) # avoid too large gauss mask

    for _ in range(n_gauss_mask):
        x = random.randint(0, w)
        y = random.randint(0, h)

        y_grid, x_grid = np.ogrid[-y:h-y, -x:w-x]

        d = np.sqrt(y_grid ** 2 + x_grid ** 2)
        gaussian_dist = np.exp(-(d**2 / (2*sigma**2)))

        rand_vals = np.random.rand(w, h)

        prob_mask += gaussian_dist  # Accumulate probabilities
        mask |= (gaussian_dist > rand_vals)

    assert np.min(footprint) >= 0 and np.max(footprint) <= 1
    valid_mask = (footprint >  0)[0]
    mask &= valid_mask

    if remove_percentage > -1:

        pixels_to_be_removed = int(np.count_nonzero(footprint) * (remove_percentage / 100.0))
        current_removed_pixels = np.count_nonzero(mask)

        prev_remove_pixel_count = 0
        while current_removed_pixels != pixels_to_be_removed:
            if current_removed_pixels > pixels_to_be_removed:
                # Randomly unmask some pixels based on probability
                removed_indices = np.argwhere(mask)
                prob_values = prob_mask[mask]
                if np.sum(prob_values) == 0:
                    prob_values = np.ones_like(prob_values)
                selected_idx = np.random.choice(removed_indices.shape[0], 1, p=prob_values/np.sum(prob_values))
                mask[tuple(removed_indices[selected_idx][0])] = False

            elif current_removed_pixels < pixels_to_be_removed:
                # Randomly mask additional pixels based on probability
                unremoved_indices = np.argwhere((~mask) & valid_mask)
                prob_values = prob_mask[(~mask) & valid_mask]
                if np.sum(prob_values) == 0:
                    prob_values = np.ones_like(prob_values)
                selected_idx = np.random.choice(unremoved_indices.shape[0], 1, p=prob_values/np.sum(prob_values))
                mask[tuple(unremoved_indices[selected_idx][0])] = True

            current_removed_pixels = np.count_nonzero(mask)
            if prev_remove_pixel_count == current_removed_pixels:
                print("Warning: Can not generate Gauss mask removing exact target pixel number!")
                break
            
            prev_remove_pixel_count = current_removed_pixels

    return mask