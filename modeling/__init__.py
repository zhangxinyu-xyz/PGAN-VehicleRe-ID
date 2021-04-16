# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                     cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                     pgan=cfg.MODEL.PGAN,
                     attention_dim=cfg.MODEL.ATTENTION_DIM,
                     prop_num=cfg.DATASETS.PROPOSAL_NUM,
                     reduction=cfg.MODEL.REDUCTION,
                     multi_nums=cfg.MODEL.MULTI_NUMS,
                     embed_num=cfg.MODEL.EMBED_NUM,
                     temp=cfg.MODEL.TEMPERATURE
                     )
    return model
