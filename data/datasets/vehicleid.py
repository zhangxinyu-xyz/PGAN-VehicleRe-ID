import os.path as osp

from .bases import BaseImageDataset
from .utils import *
from collections import defaultdict
import random

class VehicleID(BaseImageDataset):
    """
    VehicleID

    Reference:
    @inproceedings{liu2016deep,
    title={Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles},
    author={Liu, Hongye and Tian, Yonghong and Wang, Yaowei and Pang, Lu and Huang, Tiejun},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={2167--2175},
    year={2016}}

    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    """
    dataset_dir = 'PKU-VehicleID'
    container = {'small': 800, 'medium': 1600, 'large': 2400}

    def __init__(self, root='./data/', verbose=True, add_mask=False, folder='large', index=0, **kwargs):
        super(VehicleID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image')
        self.query_dir = osp.join(self.dataset_dir, 'image')
        self.gallery_dir = osp.join(self.dataset_dir, 'image')
        self._check_before_run()

        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        self.folders = folder
        self.test_size = self.container[folder]
        self.index = index

        if self.folders == 'small':
            self.test_list = osp.join(self.split_dir, 'test_list_800.txt')
            self.test_split_list = osp.join(self.split_dir, 'test_list_800_gallery_{}.txt'.format(self.index))
            self.query_split_list = osp.join(self.split_dir, 'test_list_800_query_{}.txt'.format(self.index))
        elif self.folders == 'medium':
            self.test_list = osp.join(self.split_dir, 'test_list_1600.txt')
            self.test_split_list = osp.join(self.split_dir, 'test_list_1600_gallery_{}.txt'.format(self.index))
            self.query_split_list = osp.join(self.split_dir, 'test_list_1600_query_{}.txt'.format(self.index))
        elif self.folders == 'large':
            self.test_list = osp.join(self.split_dir, 'test_list_2400.txt')
            self.test_split_list = osp.join(self.split_dir, 'test_list_2400_gallery_{}.txt'.format(self.index))
            self.query_split_list = osp.join(self.split_dir, 'test_list_2400_query_{}.txt'.format(self.index))

        print(self.test_list)

        if (not os.path.exists(self.test_split_list)) or (not os.path.exists(self.query_split_list)):
            train, query, gallery = self.process_split(relabel=True)
        else:
            train = self.process_txt(self.train_list, self.train_dir, relabel=True, add_mask=add_mask)
            gallery = self.process_txt(self.test_split_list, self.gallery_dir, relabel=False)
            query = self.process_txt(self.query_split_list, self.query_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print('=> VehicleID loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None, image_path=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = int(name)  # don't have camid information use itself name for all
            img_path = osp.join(image_path, name+'.jpg')
            output.append((img_path, pid, camid))
        return output

    def process_split(self, relabel=False):
        # read train paths
        train_pid_dict = defaultdict(list)

        # 'train_list.txt' format:
        # the first number is the number of image
        # the second number is the id of vehicle
        with open(self.train_list) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                name, pid = data.split(' ')
                pid = int(pid)
                train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)
        assert num_train_pids == 13164, 'There should be 13164 vehicles for training,' \
                                        ' but but got {}, please check the data'\
                                        .format(num_train_pids)
        print('num of train ids: {}'.format(num_train_pids))
        test_pid_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, pid = data.split(' ')
                pid = int(pid)
                test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)
        assert num_test_pids == self.test_size, 'There should be {} vehicles for testing,' \
                                                ' but but got {}, please check the data'\
                                                .format(self.test_size, num_test_pids)

        train_data = []
        query_data = []
        gallery_data = []

        # for train ids, all images are used in the train set.
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)

        # for each test id, random choose one image for gallery
        # and the other ones for query.
        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            sample = random.choice(imginfo)
            imginfo.remove(sample)
            query_data.extend(imginfo)
            gallery_data.append(sample)

        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None
        for key, value in train_pid2label.items():
            print('{key}:{value}'.format(key=key, value=value))

        train = self.parse_img_pids(train_data, pid2label=train_pid2label, image_path=self.train_dir)
        query = self.parse_img_pids(query_data, image_path=self.query_dir)
        gallery = self.parse_img_pids(gallery_data, image_path=self.gallery_dir)

        #### save test and query list
        with open(self.query_split_list, 'w') as f:
            for name, pid in query_data:
                f.write('{} {}\n'.format(name, pid))
        with open(self.test_split_list, 'w') as f:
            for name, pid in gallery_data:
                f.write('{} {}\n'.format(name, pid))

        return train, query, gallery

    def process_txt(self, txt_path, image_path, relabel=False, add_mask=False):
        with open(txt_path, 'r') as f:
            _image_list = []
            _vid_label_list = []
            _camera_label_list = []
            all_pids = {}
            for line in f.readlines():
                line = line.strip().split((' '))
                image, vid = line[0], int(line[1])
                camera = int(image)
                image = '{}.jpg'.format(image)
                if relabel:
                    if vid not in all_pids.keys():
                        all_pids[vid] = len(all_pids)
                else:
                    if vid not in all_pids.keys():
                        all_pids[vid] = vid

                vid = all_pids[vid]
                _image_list.append(image)
                _vid_label_list.append(vid)
                _camera_label_list.append(camera)

        _model_label_list = [0] * len(_image_list)
        _color_label_list = [0] * len(_image_list)

        dataset = [(osp.join(image_path, img_file), vid, cam_id) for img_file, vid, cam_id in
                   zip(_image_list, _vid_label_list, _camera_label_list)]

        return dataset

