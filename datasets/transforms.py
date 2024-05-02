# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import cv2

def crop(image, target, region):
    rg_ymin, rg_xmin, rg_h, rg_w = region
    rg_xmax = rg_xmin + rg_w
    rg_ymax = rg_ymin + rg_h 


    boxes = target['boxes'].clone()

    pre_keep = torch.zeros((boxes.shape[0]), dtype=torch.bool)
    while True:
        ov_xmin = torch.clamp(boxes[:, 0], min=rg_xmin)
        ov_ymin = torch.clamp(boxes[:, 1], min=rg_ymin)
        ov_xmax = torch.clamp(boxes[:, 2], max=rg_xmax)
        ov_ymax = torch.clamp(boxes[:, 3], max=rg_ymax)
        ov_h = ov_ymax - ov_ymin
        ov_w = ov_xmax - ov_xmin 
        keep = torch.bitwise_and(ov_w>0, ov_h>0)
        if (keep == False).all():
            return None, None
        if keep.equal(pre_keep):
            break

        keep_boxes = boxes[keep]
        img_h, img_w = target["size"]
        rg_xmin = rg_xmin if rg_xmin < min(keep_boxes[:, 0]) else int(max(min(keep_boxes[:, 0]), 0))
        rg_ymin = rg_ymin if rg_ymin < min(keep_boxes[:, 1]) else int(max(min(keep_boxes[:, 1]), 0))
        rg_xmax = rg_xmax if rg_xmax > max(keep_boxes[:, 2]) else int(min(max(keep_boxes[:, 2]), img_w-1))
        rg_ymax = rg_ymax if rg_ymax > max(keep_boxes[:, 3]) else int(min(max(keep_boxes[:, 3]), img_h-1))

        pre_keep = keep.clone()

    region = (rg_ymin, rg_xmin, rg_ymax-rg_ymin, rg_xmax-rg_xmin)
    cropped_image = F.crop(image, *region)

    target['size'] = torch.as_tensor([rg_ymax-rg_ymin, rg_xmax-rg_xmin])
    fields = ['labels', 'area', 'iscrowd', 'rec']
    if 'boxes' in target:
        boxes = target['boxes']
        cropped_boxes = boxes - torch.as_tensor([rg_xmin, rg_ymin]*2)
        target['boxes'] = cropped_boxes
        fields.append('boxes')
    
    if 'masks' in target:
        target['masks'] = target['masks'][:, rg_ymin:rg_ymax, rg_xmin:rg_xmax]
        fields.append('masks')

    if 'bezier_pts' in target:
        bezier_pts = target['bezier_pts']
        cropped_beizer_pts = bezier_pts - torch.as_tensor([rg_xmin, rg_ymin]*8)
        target['bezier_pts'] = cropped_beizer_pts
        fields.append('bezier_pts')
    
    for field in fields:
        target[field] = target[field][keep]

    return cropped_image, target

def crop_(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region
    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "bezier_pts" in target:
        bezier_pts = target['bezier_pts']
        scaled_bezier_pts = bezier_pts * torch.as_tensor([ratio_width, ratio_height] * 8)

        target['bezier_pts'] = scaled_bezier_pts

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, is_ratio: bool, prob: float):
        self.min_size = min_size
        self.max_size = max_size
        self.is_ratio = is_ratio
        self.prob = prob

    def __call__(self, img: PIL.Image.Image, target: dict):
        if random.random() < self.prob and len(target['boxes']) > 0:
            max_try = 100
            for _ in range(max_try):
                if not self.is_ratio:
                    w = random.randint(self.min_size, min(img.width, self.max_size))
                    h = random.randint(self.min_size, min(img.height, self.max_size))
                else:
                    w = int(img.width * random.uniform(self.min_size, self.max_size))
                    h = int(img.height * random.uniform(self.min_size, self.max_size))
                region = T.RandomCrop.get_params(img, [h, w])
                img_, target_ = crop(img.copy(), target.copy(), region)
                if not img_ is None:
                    return img_, target_
            print('Can not be cropped')
        return img, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "bezier_pts" in target:
            target['bezier_pts'] = target['bezier_pts'] / torch.tensor([w, h]*8, dtype=torch.float32)
        return image, target


class RandomRotate(object):
    def __init__(self, max_angle, prob):
        self.max_angle = max_angle
        self.prob = prob 
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image_w, image_h = image.size
            center_pt = (image_w//2, image_h//2)
            rotation_matrix = cv2.getRotationMatrix2D(center_pt, angle, 1)
            image = image.rotate(angle, expand=True)

            cur_w, cur_h = image.size
            target['size'] = torch.as_tensor([cur_h, cur_w])
            pad_w = (cur_w - image_w) / 2; pad_h = (cur_h - image_h) / 2

            # boxes = target['boxes'].numpy().copy()
            # boxes = boxes.reshape(-1, 2, 2)
            # boxes = np.pad(boxes, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
            # boxes = np.dot(boxes, rotation_matrix.transpose())
            # boxes[:, :, 0] += pad_w; boxes[:, :, 1] += pad_h
            # boxes = boxes.reshape(-1, 4)
            # target['boxes'] = torch.from_numpy(boxes).to(target['boxes'])

            bezier_pts = target['bezier_pts'].numpy().copy()
            bezier_pts = bezier_pts.reshape(-1, 8, 2)
            bezier_pts = np.pad(bezier_pts, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
            bezier_pts = np.dot(bezier_pts, rotation_matrix.transpose())
            bezier_pts[:, :, 0] += pad_w; bezier_pts[:, :, 1] += pad_h
            bezier_pts = bezier_pts.reshape(-1, 16)
            target['bezier_pts'] = torch.from_numpy(bezier_pts).to(target['bezier_pts'])

            boxes = []
            tmp_beziers = bezier_pts
            for tmp_bezier in tmp_beziers:
                tmp_bezier = _bezier_to_poly(tmp_bezier)
                box = _quad2minrect(tmp_bezier.reshape(-1))
                boxes.append(box)
            # from PIL import ImageDraw
            # draw = ImageDraw.ImageDraw(image)
            # for box in boxes:
            #     draw.rectangle(((box[0],box[1]),(box[2],box[3])))
            target['boxes'] = torch.from_numpy(np.array(boxes)).to(target['boxes'])
        # from PIL import ImageDraw
        # draw = ImageDraw.ImageDraw(image)
        # boxes = target['boxes']
        # for box in boxes:
        #     draw.rectangle(((box[0],box[1]),(box[2],box[3])))
        #     draw.rectangle(((box[0],box[1]),(box[2],box[3])))
        # image.save('test.jpg')
        # import pdb;pdb.set_trace()
        return image, target 


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def _bezier_to_poly(bezier):
    # bezier to polygon
    bezier = np.array(bezier)
    u = np.linspace(0, 1, 8)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points

def _quad2minrect(boxes):
    ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.array([min(boxes[::2]),min(boxes[1::2]), max(boxes[::2]), max(boxes[1::2])])