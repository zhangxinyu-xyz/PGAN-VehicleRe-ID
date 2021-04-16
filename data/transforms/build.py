# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

def build_transforms(cfg, is_train=True):
    from .transforms import RandomErasing
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def build_transforms_mask(cfg, is_train=True):
    from .transforms_mask import RandomErasing_mask, RectScale_mask, RandomHorizontalFlip_mask, ToTensor_mask, Normalize_mask, Compose_mask
    normalize_transform = Normalize_mask(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = Compose_mask([
            RectScale_mask(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
            RandomHorizontalFlip_mask(p=cfg.INPUT.PROB),
            ToTensor_mask(),
            normalize_transform,
            RandomErasing_mask(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = Compose_mask([
            RectScale_mask(cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]),
            ToTensor_mask(),
            normalize_transform
        ])
    
    return transform