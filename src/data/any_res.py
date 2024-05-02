from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast



def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def select_best_resolution_v2(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size and aspect ratio.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    original_aspect_ratio = original_height / original_width
    original_area = original_width * original_height
    best_fit = None
    min_aspect_ratio_diff = float('inf')
    min_area_ratio = float('inf')

    for width, height in possible_resolutions:
        aspect_ratio = height / width
        area = width * height
        aspect_ratio_diff = max(aspect_ratio, original_aspect_ratio) / min(aspect_ratio, original_aspect_ratio)
        area_ratio = max(area, original_area) / min(area, original_area)

        if aspect_ratio_diff < min_aspect_ratio_diff or (aspect_ratio_diff == min_aspect_ratio_diff and area_ratio < min_area_ratio):
            min_aspect_ratio_diff = aspect_ratio_diff
            min_area_ratio = area_ratio
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution, keep_ratio=False):
    """
    Resize and pad an image to a target resolution

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    if keep_ratio:
        # maintaining aspect ratio
        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized_image, (paste_x, paste_y))
    else:
        # not maintaining aspect ratio
        new_image = image.resize((target_width, target_height))
        
    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width1, height1 = select_best_resolution(image_size, possible_resolutions)
    width2, height2 = select_best_resolution_v2(image_size, possible_resolutions)
    if width1*height1 > width2*height2:
        width, height = width2, height2
    else:
        width, height = width1, height1
    return width // patch_size, height // patch_size



def process_anyres_image(image, image_transform, grid_pinpoints, base_image_size):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        image_transform: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    # best_resolution = select_best_resolution(image.size, possible_resolutions)
    width1, height1 = select_best_resolution(image.size, possible_resolutions)
    width2, height2 = select_best_resolution_v2(image.size, possible_resolutions)
    if width1*height1 > width2*height2:
        width, height = width2, height2
    else:
        width, height = width1, height1
    best_resolution = [width, height]

    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, base_image_size)

    image_original_resize = image.resize((base_image_size, base_image_size))

    image_patches =  patches + [image_original_resize] # add the original image as the last patch
    image_patches = [image_transform(image_patch)
                     for image_patch in image_patches]
    
    # generate tensor 
    # x_idnex y_index
    # x_index [[0, 1, 2],
                # [0, 1, 2],
                # [0, 1, 2]]]
    # y_index [[0, 0, 0],
                # [1, 1, 1],
                # [2, 2, 2]]
    patch_grid = (best_resolution[0]//base_image_size, best_resolution[1]//base_image_size)
    x_index = (torch.arange(patch_grid[0]).repeat(patch_grid[1], 1)  + 0.5)/patch_grid[0]
    y_index = (torch.arange(patch_grid[1]).unsqueeze(1).repeat(1, patch_grid[0]) + 0.5)/patch_grid[1]
    patch_pos = torch.stack([x_index, y_index], dim=-1).flatten(0, 1) # h*w, 2

    origin_pos = torch.tensor([[0.5, 0.5]])
    patch_pos  = torch.cat([patch_pos, origin_pos], dim=0) # h*w+1, 2

    return torch.stack(image_patches, dim=0), patch_pos


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def anyres_data_collate(batch, tokenizer, dataset_name=None):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images', 'images_patch_length', 'patch_position', 'image_size']:
                results[key] = torch.cat(cur, dim=0)
            else:
                if key in ['input_ids']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=tokenizer.pad_token_id)
                elif key in ['attention_mask']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=0)
                elif key in ['labels']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=-100)
                elif key in ['ids_gen_mask', 'ids_cmp_mask']:
                    results[key] = torch.nn.utils.rnn.pad_sequence(cur, batch_first=True, padding_value=False)

                else:
                    results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    # move to cuda
    # for key in results:
    #     if isinstance(results[key], torch.Tensor):
    #         results[key] = results[key].cuda()
    results['dataset_name'] = dataset_name

    return results


def anyres_data_collate_old(batch, dataset_name=None):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images', 'images_patch_length', 'patch_position', 'image_size']:
                results[key] = torch.cat(cur, dim=0)
            else:
                results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    results['dataset_name'] = dataset_name

    return results
