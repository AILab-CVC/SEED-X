import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import torch.distributed as dist
import os
import random
from braceexpand import braceexpand
import hydra

from .any_res import process_anyres_image, anyres_data_collate, anyres_data_collate_old

import pyrootutils
try:
    import fitz
except:
    fitz = None

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

dynamic_padding = False

BOI_TOKEN = '<img>'
BOP_TOKEN = '<patch>'
EOI_TOKEN = '</img>'
EOP_TOKEN = '</patch>'
IMG_TOKEN = '<img_{:05d}>'

gen_prompt_response = [
    "Here is a picture.",
    "I have designed an image.",
    "Here is a photo.",
    "I have generated an image.",
    "Here's a painting.",
    "Here's a drawing.",
    "Enjoy this illustration.",
    "Take a look at this image.",
    "Here is a picture.",
    "I have created a photo.",
    "Enjoy this photo.",
    "I have generated a picture.",
    "Here is a photograph.",
    "Here's an image.",
    "Certainly, here's an image.",
    "Absolutely, here is a painting.",
    "Sure, here is a picture.",
    "Of course, here is a photo.",
    "Certainly, please enjoy this picture.",
    "Sure, please enjoy this illustration.",
    "",
]

def build_multi_datapipes(datapipes, tokenizer=None, image_transform=None, sample_weights=None):
    # assert concat_type in ['concat', 'mux_longest', 'sample']
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [
        hydra.utils.instantiate(datapipe, tokenizer=tokenizer, image_transform=image_transform) for datapipe in datapipes
    ]

    datasets_to_weights_dict = {}
    for dataset, sample_weight in zip(datapipes, sample_weights):
        datasets_to_weights_dict[dataset] = sample_weight
    datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict, seed=42 + dist.get_rank())

    return datapipe


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    if 'metadata' not in unwarpped:
        unwarpped['metadata'] = '{}'
    # if '__key__' in unwarpped:
    #     unwarpped['__key__'] = unwarpped['__key__'].split('/')[-1]
    return unwarpped


# def filter_data_with_similarity(item, similarity_thr=0.2, min_resolution=180, min_aspect_ratio=0.666):
def filter_data_with_similarity(item, similarity_thr=0.2, assure_text=True):
    if ('images' not in item):
        # print(item['__key__'])
        # print('filtered because no images')
        return False
    elif (not item.get('filter_flag', True)):
        # print(item['__key__'])
        # print('filtered because filter flag.')
        return False
    elif assure_text and ('text' not in item):
        # print(item['__key__'])
        # print('filtered because assure_text')
        return False
    else:
        metadata = json.loads(item['metadata'])

        if 'all_similarities' in metadata:
            similarity = max(metadata['all_similarities'])
        elif 'similarity' in metadata:
            similarity = metadata['similarity']
        elif 'score' in metadata:
            similarity = metadata['score']
        elif 'SCORE' in metadata:
            similarity = metadata['SCORE']
        else:
            similarity = None

        if similarity is not None:
            if similarity < similarity_thr:
                # print(item['__key__'])
                # print('filtered because similarity')
                return False

        return True


def select(sample):
    return {
        'input_ids': sample['input_ids'],
        'attention_mask': sample['attention_mask'],
        'labels': sample['labels'],
        'ids_gen_mask': sample['ids_gen_mask'],
        'ids_cmp_mask': sample['ids_cmp_mask'],
        'embeds_gen_mask': sample['embeds_gen_mask'],
        'embeds_cmp_mask': sample['embeds_cmp_mask'],
        'images': sample['images']
    }



def filter_data_with_image_ids(item):
    if ('images' not in item):
        return False
    elif 'input_ids' not in item:
        return False
    else:
        return True



def decode_llava_data(item,
                      image_dir,
                      tokenizer,
                      image_transform=None,
                      max_length=128,
                      min_resolution=400,
                      instruction_prompt='[INST] {instruction} [/INST]\n',
                      turn_sep='\n',
                      system_message='',
                      min_aspect_ratio=0.666,
                      num_img_in_tokens=64,
                      num_img_out_tokens=64,
                      multi_resolution=False,
                      resolution_grids=None,
                      base_resolution=224,
                      grid_pinpoints=None):
    
    key, value = item

    if value.get('data', None) is None:
        return {}

    if 'image' in value and 'null' not in value['image'] and value['image'] != '' and value['image'] != 'none':
        image_path = os.path.join(image_dir, value['image'].lstrip('/'))

        try:
            if image_path.endswith('pdf'):
                if fitz is None:
                    print('You need to install fitz to load pdf images by "pip3 install pymupdf".')
                    raise Exception('fitz is not installed.')
                pages = fitz.open(image_path)
                page = pages[0]
                zoom_x = 1  
                zoom_y = 1  
                matrix = fitz.Matrix(zoom_x, zoom_y)
                pix = page.get_pixmap(matrix=matrix)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                image = Image.open(image_path).convert('RGB')


            if image_transform is not None:
                if multi_resolution:
                    img_size = image.size
                    image, patch_pos = process_anyres_image(image, image_transform, grid_pinpoints, base_resolution)
                    images_patch_length = torch.tensor([len(patch_pos)], dtype=torch.long)
                    image_size = torch.tensor([img_size], dtype=torch.long)
                    embeds_gen_mask = [False] * len(patch_pos)
                    embeds_cmp_mask = [True] * len(patch_pos)

                else:

                    image = image_transform(image)
                    embeds_gen_mask = False
                    embeds_cmp_mask = True
        except Exception as e:
            print('Error while decode image: ', e)
            return {}
    else:
        image = None
        embeds_gen_mask = None
        embeds_cmp_mask = None
        if multi_resolution:
            images_patch_length = None
            image_size = None
            patch_pos = None

    input_ids = []
    labels = []
    input_text = ''

    if system_message != '':
        if not system_message.endswith('\n'):
            system_message += '\n'
        input_text += system_message
        item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    ground_response = False
    for idx, content in enumerate(value['data']):
        # USER
        if idx % 2 == 0:
            if idx == 0:
                if image is not None:
                    if multi_resolution:
                        image_tokens = ''
                        for patch_legnth in images_patch_length.tolist():
                            for _ in range(patch_legnth-1):
                                image_tokens += BOP_TOKEN + ''.join([IMG_TOKEN.format(int(item))
                                                                for item in range(num_img_in_tokens)]) + EOP_TOKEN
                            image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item))
                                                            for item in range(num_img_in_tokens)]) + EOI_TOKEN

                    else:
                        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item))
                                                        for item in range(num_img_in_tokens)]) + EOI_TOKEN
                else:
                    image_tokens = ''

                image_in_start = np.random.uniform(0, 1) < 0.5
                if image_in_start:
                    instruction = image_tokens + content
                else:
                    instruction = content + image_tokens

                text = instruction_prompt.format_map({'instruction': instruction})

            else:
                text = turn_sep + instruction_prompt.format_map({'instruction': content})
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)
        # ASSISTANT
        else:
            text = content
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = item_ids
            if '<box_start>' in content and '<box_end>' in content:
                ground_response = True

        input_text += text
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
    ids_cmp_mask = [False] * len(input_ids)
    ids_gen_mask = [False] * len(input_ids)

    if image is not None:
        boi_idx = input_ids.index(boi_token_id)
        eoi_idx = input_ids.index(eoi_token_id)

        if eoi_idx >= max_length:
            return {}

    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        ids_cmp_mask = ids_cmp_mask[:max_length]
        ids_gen_mask = ids_gen_mask[:max_length]
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask) if embeds_cmp_mask is not None else None
    embeds_gen_mask = torch.tensor(embeds_gen_mask) if embeds_gen_mask is not None else None

    if image is not None:
        ids_cmp_mask[boi_idx + 1:eoi_idx] = True
    
    if multi_resolution:
        bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
        eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]
        bop_indices = torch.where(input_ids == bop_token_id)
        eop_indices = torch.where(input_ids == eop_token_id)

        for bop_idx, eop_idx in zip(bop_indices[0], eop_indices[0]):
            ids_cmp_mask[bop_idx + 1:eop_idx] = True

    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'images': image,
        'text': input_text,
    }

    if multi_resolution:
        ret.update({
            'images_patch_length': images_patch_length,
            'patch_position': patch_pos,
            'image_size': image_size,
        })
    
    return ret



def llava_collate(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results

def llava_collate_new(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results


def build_llava_jsonl_datapipes(data_dir,
                                image_dir,
                                tokenizer=None,
                                max_length=77,
                                batch_size=None,
                                min_resolution=180,
                                image_transform=None,
                                instruction_prompt='[INST] {instruction} [INST]\n',
                                turn_sep='\n',
                                system_message='',
                                min_aspect_ratio=0.666,
                                num_img_in_tokens=64,
                                num_img_out_tokens=64,
                                cycle_count=None,
                                multi_resolution=False,
                                resolution_grids=None,
                                base_resolution=224,
                                dataset_name=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    grid_pinpoints = []
    if multi_resolution:
        resolution_grids = list(resolution_grids)
        
        for scale in resolution_grids:
            s1, s2 = scale.split('x')
            grid_pinpoints.append([int(s1)*base_resolution, int(s2)*base_resolution])
        

    decode_partial = functools.partial(decode_llava_data,
                                       image_dir=image_dir,
                                       tokenizer=tokenizer,
                                       image_transform=image_transform,
                                       max_length=max_length,
                                       instruction_prompt=instruction_prompt,
                                       turn_sep=turn_sep,
                                       system_message=system_message,
                                       min_resolution=min_resolution,
                                       min_aspect_ratio=min_aspect_ratio,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens,
                                       multi_resolution=multi_resolution,
                                       resolution_grids=resolution_grids,
                                       base_resolution=base_resolution,
                                       grid_pinpoints=grid_pinpoints)

    filter_partial = functools.partial(filter_data_with_image_ids)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_partial)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        if dynamic_padding:
            collate_func = functools.partial(anyres_data_collate, tokenizer=tokenizer, dataset_name=dataset_name)
        else:
            collate_func = functools.partial(anyres_data_collate_old, dataset_name=dataset_name)
        datapipe = datapipe.collate(collate_fn=collate_func if multi_resolution else llava_collate)

    return datapipe



def decode_single_turn_edit_data(item,
                                 image_dir,
                                 tokenizer,
                                 image_transform=None,
                                 max_length=128,
                                 min_resolution=400,
                                 instruction_prompt='[INST] {instruction} [/INST]\n',
                                 turn_sep='\n',
                                 system_message='',
                                 min_aspect_ratio=0.666,
                                 prompt_drop_ratio=0.0,
                                 use_polite_response=True,
                                 num_img_in_tokens=64,
                                 num_img_out_tokens=64,
                                 multi_resolution=False,
                                 resolution_grids=None,
                                 base_resolution=224,
                                 grid_pinpoints=None):
    key, value = item
    if 'source_image' not in value or 'target_image' not in value or 'instruction' not in value:
        return {}
    try:
        source_image_path = os.path.join(image_dir, value['source_image'])
        target_image_path = os.path.join(image_dir, value['target_image'])

        source_image = Image.open(source_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')
        width, height = source_image.size

        aspect_ratio = height / width
        if height < min_resolution or width < min_resolution:
            #print(f'filtered because resolution: ({width},{height})')
            return {}
        if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
            #print(f'filtered because aspect ratio: ({width},{height})')
            return {}

        if multi_resolution:
            images = []
            embeds_cmp_mask = []
            embeds_gen_mask = []
            images_patch_length = []
            image_size = []
            patch_position = []

            img_size = source_image.size
            image, patch_pos = process_anyres_image(source_image, image_transform, grid_pinpoints, base_resolution)
            images_patch_length.append(len(patch_pos))
            image_size.append(img_size)
            patch_position.append(patch_pos)
            images.append(image)
            embeds_cmp_mask.extend([True]*len(patch_pos))
            embeds_gen_mask.extend([False]*len(patch_pos))

            image_tokens = ''
            for _ in range(len(patch_pos)-1):
                image_tokens += BOP_TOKEN + ''.join([IMG_TOKEN.format(int(item))
                                                    for item in range(num_img_in_tokens)]) + EOP_TOKEN
            image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item))
                                                for item in range(num_img_in_tokens)]) + EOI_TOKEN
            
            img_size = target_image.size
            image, patch_pos = process_anyres_image(target_image, image_transform, grid_pinpoints, base_resolution)
            images_patch_length.append(len(patch_pos))
            image_size.append(img_size)
            patch_position.append(patch_pos)
            images.append(image)
            images = torch.cat(images, dim=0)

            embeds_cmp_mask.extend([False]*len(patch_pos))
            embeds_gen_mask.extend([False]*(len(patch_pos)-1) + [True])
            
        else:
            images = [source_image, target_image]
            if image_transform is not None:
                source_image = image_transform(source_image)
                target_image = image_transform(target_image)
                
                images = torch.stack([source_image, target_image], dim=0)

        input_ids = []
        labels = []
        input_text = ''

        if system_message != '':
            if not system_message.endswith('\n'):
                system_message += '\n'
            input_text += system_message
            item_ids = tokenizer.encode(system_message, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)
            input_ids.extend(item_ids)
            labels.extend(item_labels)

        if 'instruction_new' in value and 'response' in value:
            instruction = value['instruction_new']
            response = value['response']
        else:
            instruction = value['instruction']
            response = random.choice(gen_prompt_response)

        drop_prompt = np.random.uniform(0, 1) < prompt_drop_ratio
        if drop_prompt or instruction is None:
            instruction = ''

        if not use_polite_response:
            response = ''

        image_gen_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)]) + EOI_TOKEN

        image_in_start = np.random.uniform(0, 1) < 0.5
        if image_in_start:
            instruction = instruction_prompt.format_map({'instruction': image_tokens + instruction})
        else:
            instruction = instruction_prompt.format_map({'instruction': instruction + image_tokens})

        response = response + image_gen_tokens

        item_ids = tokenizer.encode(instruction, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_text += instruction
        input_ids.extend(item_ids)
        labels.extend(item_labels)

        item_ids = tokenizer.encode(response, add_special_tokens=False)
        item_labels = item_ids
        input_text += response
        input_ids.extend(item_ids)
        labels.extend(item_labels)

        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
        ids_cmp_mask = [False] * len(input_ids)
        ids_gen_mask = [False] * len(input_ids)

        if not multi_resolution:
            embeds_cmp_mask = [True, False]
            embeds_gen_mask = [False, True]

        if len(input_ids) >= max_length:
            print('An edit sample has been removed because of max length.', len(input_ids))
            return {}
        else:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
            ids_cmp_mask = ids_cmp_mask + [False] * padding_length
            ids_gen_mask = ids_gen_mask + [False] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
        ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
        embeds_cmp_mask = torch.tensor(embeds_cmp_mask) if embeds_cmp_mask is not None else None
        embeds_gen_mask = torch.tensor(embeds_gen_mask) if embeds_gen_mask is not None else None

        if multi_resolution:
            bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
            eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]
            boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))
            eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))
            for boi_idx, eoi_idx in zip(boi_indices[0][:-1], eoi_indices[0][:-1]):
                ids_cmp_mask[boi_idx + 1:eoi_idx] = True
                
            ids_gen_mask[boi_indices[0][-1] + 1:eoi_indices[0][-1]] = True
            labels[boi_indices[0][-1] + 1:eoi_indices[0][-1] + 1] = -100
        else:
            boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
            eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()
            ids_cmp_mask[boi_idx[0] + 1:eoi_idx[0]] = True
            ids_gen_mask[boi_idx[1] + 1:eoi_idx[1]] = True
            labels[boi_idx[1] + 1:eoi_idx[1] + 1] = -100

    except Exception as e:
        print('Error while decode image: ', e)
        return {}
    
    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'images': images,
        'text': input_text,
    }
    if multi_resolution:
        ret.update({
            'images_patch_length': torch.tensor(images_patch_length, dtype=torch.long),
            'patch_position': torch.cat(patch_position, dim=0),
            'image_size': torch.tensor(image_size, dtype=torch.long),
        })

    return ret


def single_turn_edit_collate(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images']:
                results[key] = torch.cat(cur, dim=0)
            else:
                results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results


def build_single_turn_edit_datapipes(data_dir,
                                     image_dir,
                                     tokenizer=None,
                                     max_length=77,
                                     batch_size=None,
                                     min_resolution=180,
                                     image_transform=None,
                                     instruction_prompt='[INST] {instruction} [INST]\n',
                                     turn_sep='\n',
                                     system_message='',
                                     min_aspect_ratio=0.666,
                                     prompt_drop_ratio=0.0,
                                     use_polite_response=True,
                                     num_img_in_tokens=64,
                                     num_img_out_tokens=64,
                                     cycle_count=None,
                                     multi_resolution=False,
                                     resolution_grids=None,
                                     base_resolution=224,
                                     dataset_name=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    grid_pinpoints = []
    if multi_resolution:
        resolution_grids = list(resolution_grids)
        
        for scale in resolution_grids:
            s1, s2 = scale.split('x')
            grid_pinpoints.append([int(s1)*base_resolution, int(s2)*base_resolution])

    decode_partial = functools.partial(decode_single_turn_edit_data,
                                       image_dir=image_dir,
                                       tokenizer=tokenizer,
                                       image_transform=image_transform,
                                       max_length=max_length,
                                       instruction_prompt=instruction_prompt,
                                       turn_sep=turn_sep,
                                       system_message=system_message,
                                       min_resolution=min_resolution,
                                       min_aspect_ratio=min_aspect_ratio,
                                       prompt_drop_ratio=prompt_drop_ratio,
                                       use_polite_response=use_polite_response,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens,
                                       multi_resolution=multi_resolution,
                                       resolution_grids=resolution_grids,
                                       base_resolution=base_resolution,
                                       grid_pinpoints=grid_pinpoints)

    filter_partial = functools.partial(filter_data_with_image_ids)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.shuffle(buffer_size=512)
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_partial)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        if dynamic_padding:
            collate_func = functools.partial(anyres_data_collate, tokenizer=tokenizer, dataset_name=dataset_name)
        else:
            collate_func = functools.partial(anyres_data_collate_old, dataset_name=dataset_name)
        datapipe = datapipe.collate(collate_fn=collate_func if multi_resolution else single_turn_edit_collate)
    return datapipe
