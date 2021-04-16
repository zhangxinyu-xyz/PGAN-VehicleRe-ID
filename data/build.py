# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn, train_collate_fn_withproposal, val_collate_fn_withproposal
from .datasets import init_dataset, ImageDataset, ImageProposalDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms, build_transforms_mask


def make_data_loader(cfg):

    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_transforms = build_transforms_mask(cfg, is_train=True)
    val_transforms = build_transforms_mask(cfg, is_train=False)
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, folder=cfg.DATASETS.FOLDER, index=cfg.DATASETS.INDEX)
    else:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, folder=cfg.DATASETS.FOLDER, index=cfg.DATASETS.INDEX)

    num_classes = dataset.num_train_pids
    train_set = ImageProposalDataset(dataset.train, transform=train_transforms,
                                     proposal_path=cfg.DATASETS.PROPOSAL_DIR,
                                     padding=cfg.INPUT.PADDING,
                                     proposal_num=cfg.DATASETS.PROPOSAL_NUM, istrain=True)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn_withproposal, drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn_withproposal, drop_last=False
        )
    val_set = ImageProposalDataset(dataset.query + dataset.gallery, transform=val_transforms, proposal_path=cfg.DATASETS.PROPOSAL_DIR, proposal_num=cfg.DATASETS.PROPOSAL_NUM)
    num_query = len(dataset.query)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_withproposal, drop_last=False
    )
    return train_loader, val_loader, num_query, num_classes

def make_test_data_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_transforms = build_transforms_mask(cfg, is_train=True)
    val_transforms = build_transforms_mask(cfg, is_train=False)
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, folder=cfg.DATASETS.FOLDER, index=cfg.DATASETS.INDEX)
    else:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, folder=cfg.DATASETS.FOLDER, index=cfg.DATASETS.INDEX)

    num_classes = dataset.num_train_pids
    train_set = ImageProposalDataset(dataset.train, transform=val_transforms,
                                     proposal_path=cfg.DATASETS.PROPOSAL_DIR,
                                     padding=cfg.INPUT.PADDING,
                                     proposal_num=cfg.DATASETS.PROPOSAL_NUM, istrain=True)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=train_collate_fn_withproposal, drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn_withproposal, drop_last=False
        )
    val_set = ImageProposalDataset(dataset.query + dataset.gallery, transform=val_transforms,
                                   proposal_path=cfg.DATASETS.PROPOSAL_DIR,
                                   proposal_num=cfg.DATASETS.PROPOSAL_NUM)
    num_query = len(dataset.query)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_withproposal, drop_last=False
    )
    return train_loader, val_loader, num_query, num_classes



