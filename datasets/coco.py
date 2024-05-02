# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

__all__ = ['build']


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

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_set=None, aux_target_hacks=None, dataset_name=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
        self.image_set = image_set
        if 'ctw1500' in dataset_name:
            self.prompt = 1
        else:
            self.prompt = 0

    def __getitem__(self, idx):

        img, target = super(CocoDetection, self).__getitem__(idx)
        if self.image_set=='train':
            while not len(target):
                print("Error idx: {}".format(idx))
                idx = random.randint(0, self.__len__()-1)
                img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self.image_set=='train':
            while not len(target["boxes"]):
                print("Error idx: {}".format(idx))
                idx = random.randint(0, self.__len__()-1)
                img, target = super(CocoDetection, self).__getitem__(idx)
                image_id = self.ids[idx]
                target = {'image_id': image_id, 'annotations': target}
                img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target['prompt'] = torch.tensor(self.prompt)
        return img, target



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        # image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        rec = [obj["rec"] for obj in anno]
        rec = torch.as_tensor(rec, dtype=torch.int32)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # poly = [_bezier_to_poly(obj["bezier_pts"]) for obj in anno]
        # poly = torch.as_tensor(poly, dtype=torch.float32)
        # poly = poly.reshape(-1,32)
        bezier_pts = [obj['bezier_pts'] for obj in anno]
        bezier_pts = torch.tensor(bezier_pts, dtype=torch.float32).reshape(-1, 16)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep][:100]
        classes = classes[keep][:100]
        rec = rec[keep][:100]
        bezier_pts = bezier_pts[keep][:100]
        if self.return_masks:
            masks = masks[keep][:100]
        if keypoints is not None:
            keypoints = keypoints[keep][:100]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["bezier_pts"] = bezier_pts
        target["rec"] = rec
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep][:100]
        target["iscrowd"] = iscrowd[keep][:100]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

class MixDataset(object):
    def __init__(self, datasets, ratios):
        self.datasets = datasets
        self.ratios = ratios
        self.lengths = []
        for dataset in self.datasets:
            self.lengths.append(len(dataset))
        self.lengths = np.array(self.lengths)
        self.seperate_inds = []
        s = 0
        for i in self.ratios[:-1]:
            s += i
            self.seperate_inds.append(s)

    def __len__(self):
       return self.lengths.sum()
       
    def __getitem__(self, item):
        i = np.random.rand()
        ind = bisect.bisect_right(self.seperate_inds, i)
        b_ind = np.random.randint(self.lengths[ind])
        return self.datasets[ind][b_ind]

def make_coco_transforms(image_set, max_size_train, min_size_train, max_size_test, min_size_test,
                         crop_min_ratio, crop_max_ratio, crop_prob, rotate_max_angle, rotate_prob,
                         brightness, contrast, saturation, hue, distortion_prob):

    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomSizeCrop(crop_min_ratio, crop_max_ratio, True, crop_prob))
        transforms.append(T.RandomRotate(rotate_max_angle, rotate_prob))
        transforms.append(T.RandomResize(min_size_train, max_size_train))
        transforms.append(T.RandomDistortion(brightness, contrast, saturation, hue, distortion_prob))
    if image_set == 'val':
        transforms.append(T.RandomResize([min_size_test], max_size_test))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    return T.Compose(transforms)

def build(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'totaltext_train':
            img_folder = root / "totaltext" / "train_images"; ann_file = root / "totaltext" / "train.json"
        elif dataset_name == 'totaltext_val':
            img_folder = root / "totaltext" / "test_images"; ann_file = root / "totaltext" / "test.json"
        elif dataset_name == 'mlt_train':
            img_folder = root / "mlt2017" / "MLT_train_images"; ann_file = root / "mlt2017" / "train.json"
        elif dataset_name == 'ctw1500_train':
            img_folder = root / "ctw1500" / "train_images"; ann_file = root / "ctw1500" / "annotations" / "train_96voc.json"
        elif dataset_name == 'ctw1500_val':
            img_folder = root / "ctw1500" / "test_images"; ann_file = root / "ctw1500" / "annotations" / "test.json"
        elif dataset_name == 'syntext1_train':
            img_folder = root / "syntext1" / "syntext_word_eng"; ann_file = root / "syntext1" / "train.json"
        elif dataset_name == 'syntext2_train':
            img_folder = root / "syntext2" / "emcs_imgs"; ann_file = root / "syntext2" / "train.json"
        elif dataset_name == 'ic13_train':
            img_folder = root / "icdar2013" / "train_images"; ann_file = root / "icdar2013" / "ic13_abc.json"
        elif dataset_name == 'ic15_train':
            img_folder = root / "icdar2015" / "train_images"; ann_file = root / "icdar2015" / "ic15_train.json"
        elif dataset_name == 'ic13_val':
            img_folder = root / "icdar2013" / "ic13_Test_Images"; ann_file = root / "icdar2013" / "icdar_2013_ist_test.json"
        elif dataset_name == 'icdar2015_val':
            img_folder = root / "icdar2015" / "test_images"; ann_file = root / "icdar2015" / "ic15_test.json"
        elif dataset_name == 'art':
            img_folder = root / "ArT" / "rename_artimg_train"; ann_file = root / "ArT" / "annotations" / "abcnet_art_train.json"
        elif dataset_name == 'lsvt':
            img_folder = root / "LSVT" / "rename_lsvtimg_train"; ann_file = root / "LSVT" / "annotations" / "abcnet_lsvt_train.json"
        elif dataset_name == 'rects':
            img_folder = root / "ReCTS" / "ReCTS_train_images"; ann_file = root / "ReCTS" / "annotations" / "rects_train.json"
        elif dataset_name == 'rects_val':
            img_folder = root / "ReCTS" / "ReCTS_test_images"; ann_file = root / "ReCTS" / "annotations" / "rects_val.json"
        elif dataset_name == 'rects_test':
            img_folder = root / "ReCTS_test_images"; ann_file = root / "annotations" / "rects_test.json"
        elif dataset_name == 'chnsyntext':
            img_folder = root / "syntext" / "syn_130k_images"; ann_file = root / "syntext" / "chn_syntext.json"
        elif dataset_name == 'msra_td500':
            img_folder = root / "ABCNetV2_td500" / "train_images"; ann_file = root / "ABCNetV2_td500" / "td500.json"
        elif dataset_name == 'msra_td500_test':
            img_folder = root / "ABCNetV2_td500" / "test_images"; ann_file = root / "ABCNetV2_td500" / "td500_test.json"
        elif dataset_name == 'arabic':
            img_folder = root / "Arabic"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Arabic.json"
        elif dataset_name == 'bangla':
            img_folder = root / "Bangla"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Bangla.json"
        elif dataset_name == 'chinese':
            img_folder = root / "Chinese"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Chinese.json"
        elif dataset_name == 'hindi':
            img_folder = root / "Hindi"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Hindi.json"
        elif dataset_name == 'japanese':
            img_folder = root / "Japanese"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Japanese.json"
        elif dataset_name == 'korean':
            img_folder = root / "Korean"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Korean.json"
        elif dataset_name == 'latin':
            img_folder = root / "Latin"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_Latin.json"
        elif dataset_name == 'mlt2019':
            img_folder = root / "icdar2019_mlt_images"; ann_file = root / "icdar2019mlt_with_synth_coco_fomat" / "abc_icdar_2019_mlt.json"
        elif dataset_name == 'mlt2019_test':
            img_folder = root / "icdar2019mlt" / "MLT2019_test"; ann_file = root / "icdar2019mlt"/ "annotations" / "icdar_2019_mlt_test.json"
        elif dataset_name == 'hust_art':
            img_folder = root / "HUST-ART" / "train_images"; ann_file = root / "HUST" / "hust_art_train.json"
        elif dataset_name == 'hust_ast':
            img_folder = root / "HUST-AST" / "images"; ann_file = root / "HUST" / "hust_ast.json"
        elif dataset_name == 'vintext':
            img_folder = root / "vintext" / "train_images"; ann_file = root / "vintext" / "train.json"
        elif dataset_name == 'vintext_val':
            img_folder = root / "vintext" / "val_image"; ann_file = root / "vintext" / "val.json"
        elif dataset_name == 'vintext_test':
            img_folder = root / "vintext" / "test_image"; ann_file = root / "vintext" / "test.json"
        elif dataset_name == 'textocr_96':
            img_folder = root / "textocr" / "train_images"; ann_file = root / "textocr" / "textocr_train_word_96.json"
        elif dataset_name == 'ic13_video':
            img_folder = '/data/hmx/ic13_video_new'; ann_file = '/data/hmx/ic13video_test.json'
        else:
            raise NotImplementedError

        transforms = make_coco_transforms(image_set, args.max_size_train, args.min_size_train,
              args.max_size_test, args.min_size_test, args.crop_min_ratio, args.crop_max_ratio,
              args.crop_prob, args.rotate_max_angle, args.rotate_prob, args.brightness, args.contrast,
              args.saturation, args.hue, args.distortion_prob)
        dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks, image_set=image_set, dataset_name=dataset_name)
        datasets.append(dataset)
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    return dataset

