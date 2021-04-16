# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
import numpy as np
import os

from utils.reid_metric import R1_mAP, R1_mAP_reranking

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            if len(batch) == 4:
                data, pids, camids, img_paths = batch
                proposal = None
            else:
                data, proposal, pids, camids, img_paths = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            proposal = proposal.to(device) if torch.cuda.device_count() >= 1 and proposal is not None else proposal
            features = model((data, proposal))
            return features['layer4']['feat'], pids, camids, img_paths

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def write_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')

def get_index(distmat, data_loader, output_dir, num_query):
    indices = np.argsort(distmat, axis=1)
    for idx, (name, pid, camid) in enumerate(data_loader.dataset.dataset[0:num_query]):
        imgname = os.path.basename(name)
        sort_index = indices[idx][0:100]
        save_path = os.path.join(output_dir, imgname.replace('jpg', 'txt'))
        write_txt(save_path, sort_index)

def generate_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')


def inference(
        cfg,
        model,
        val_loader,
        num_query,
):
    device = cfg.MODEL.DEVICE
    log_period = cfg.SOLVER.LOG_PERIOD
    output_dir = cfg.OUTPUT_DIR

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, remove_camera=True,
                             extract_feat=True)}, device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, remove_camera=True,
                                       extract_feat=True)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iteration(engine):
        iter = (engine.state.iteration - 1) % len(val_loader) + 1
        if iter % log_period == 0:
            logger.info("Extract Features. Iteration[{}/{}]"
                        .format(iter, len(val_loader)))

    evaluator.run(val_loader)
    distmat, cmc, mAP = evaluator.state.metrics['r1_mAP']

    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return mAP, cmc[0], cmc[4]
