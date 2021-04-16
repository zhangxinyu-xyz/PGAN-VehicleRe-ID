# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':  # Ignore: we do not use label smooth
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            idloss = F.cross_entropy(score, target)
            return [idloss, idloss, None, None, None, None]
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            triloss = triplet(feat, target)[0]
            return [triloss, None, triloss, None, None, None]
    ##### total loss, idloss, triloss, clusterloss, centerloss, rangeloss #####
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(features, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                idloss = 0
                triloss = 0
                prop_loss = 0
                base_loss = 0
                for layer, output in features.items():
                    idloss += F.cross_entropy(output['cls'], target)
                    triloss += triplet(output['global'], target)[0]
                    if cfg.MODEL.PGAN == 'yes':
                        prop_loss += triplet(output['prop'], target)[0]
                        base_loss += triplet(output['base'], target)[0]

                return [cfg.SOLVER.ID_LOSS_WEIGHT * idloss + triloss * cfg.SOLVER.TRIPLET_LOSS_WEIGHT + \
                        prop_loss + base_loss, idloss, triloss, None, None, None, prop_loss, base_loss]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


##### Ignore: We do not use center loss #####
def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, '
              'range_center,triplet_center, triplet_range_center '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    ##### total loss, idloss, triloss, clusterloss, centerloss, rangeloss #####
    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                idloss = xent(score, target)
                centerloss = center_criterion(feat, target)
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT, idloss, None, None, centerloss, None] # change by xinyu, open label smooth
            else:
                idloss = F.cross_entropy(score, target)
                centerloss = center_criterion(feat, target)
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT, idloss, None, None, centerloss, None] # change by xinyu, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                idloss = xent(score, target)
                centerloss = center_criterion(feat, target)
                rangeloss = range_criterion(feat, target)[0]
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT + rangeloss * cfg.SOLVER.RANGE_LOSS_WEIGHT,\
                       idloss, None, None, centerloss, rangeloss] # change by xinyu, open label smooth
            else:
                idloss = F.cross_entropy(score, target)
                centerloss = center_criterion(feat, target)
                rangeloss = range_criterion(feat, target)[0]
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT + rangeloss * cfg.SOLVER.RANGE_LOSS_WEIGHT,\
                       idloss, None, None, centerloss, rangeloss] # change by xinyu, open label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                idloss = xent(score, target)
                triloss = triplet(feat, target)[0]
                centerloss = center_criterion(feat, target)
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT + triloss * cfg.SOLVER.TRIPLET_LOSS_WEIGHT,\
                       idloss, triloss, None, centerloss, None] # change by xinyu, open label smooth
            else:
                idloss = F.cross_entropy(score, target)
                triloss = triplet(feat, target)[0]
                centerloss = center_criterion(feat, target)
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT + triloss * cfg.SOLVER.TRIPLET_LOSS_WEIGHT,\
                       idloss, triloss, None, centerloss, None] # change by xinyu, open label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                idloss = xent(score, target)
                triloss = triplet(feat, target)[0]
                centerloss = center_criterion(feat, target)
                rangeloss = range_criterion(feat, target)[0]
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT + \
                       triloss * cfg.SOLVER.TRIPLET_LOSS_WEIGHT + rangeloss * cfg.SOLVER.RANGE_LOSS_WEIGHT,\
                       idloss, triloss, None, centerloss, rangeloss] # change by xinyu, open label smooth
            else:
                idloss = F.cross_entropy(score, target)
                triloss = triplet(feat, target)[0]
                centerloss = center_criterion(feat, target)
                rangeloss = range_criterion(feat, target)[0]
                return [idloss + centerloss * cfg.SOLVER.CENTER_LOSS_WEIGHT + \
                       triloss * cfg.SOLVER.TRIPLET_LOSS_WEIGHT + rangeloss * cfg.SOLVER.RANGE_LOSS_WEIGHT,\
                       idloss, triloss, None, centerloss, rangeloss] # change by xinyu, open label smooth

        else:
            print('expected METRIC_LOSS_TYPE with center should be center,'
                  ' range_center, triplet_center, triplet_range_center '
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion
