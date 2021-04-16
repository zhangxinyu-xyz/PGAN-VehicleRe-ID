# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dataset_loader import ImageDataset, ImageProposalDataset

from .veri import VeRi
from .vehicleid import VehicleID
from .vric import Vric
from .veriwild import VeriWild


__factory = {
    'veri': VeRi,
    'vehicleid': VehicleID,
    'vric': Vric,
    'veriwild': VeriWild,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
