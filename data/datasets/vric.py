import xmltodict
import os.path as osp

from .bases import BaseImageDataset

class Vric(BaseImageDataset):
    """
    VeRi: https://github.com/VehicleReId/VeRidataset

    Dataset statistics:
    # 50,000 images of 776 vehicles captured by 20 cameras covering an 1.0 km^2 area in 24 hours
    """
    dataset_dir = 'VRIC'
    
    def __init__(self, root='./data/', verbose=True, add_mask=False, num_instance=4, **kwargs):
        super(Vric, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_images')
        self.query_dir = osp.join(self.dataset_dir, 'probe_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_images')
        
        self._check_before_run()
        
        train = self._process_txt(osp.join(self.dataset_dir, 'vric_train.txt'), self.train_dir, relabel=True,
                                  add_mask=add_mask, num_instance=num_instance)
        gallery = self._process_txt(osp.join(self.dataset_dir, 'vric_gallery.txt'), self.gallery_dir, relabel=False)
        query = self._process_txt(osp.join(self.dataset_dir, 'vric_probe.txt'), self.query_dir, relabel=False)
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        if verbose:
            print("=> VeRi loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
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
    
    def _process_xml(self, xml_path, image_path, relabel=False, add_mask=False):
        with open(xml_path, 'rb') as f:
            xml = xmltodict.parse(f)
        all_items = xml['TrainingImages']['Items']['Item']
        
        _image_list = [item['@imageName'] for item in all_items]
        _vid_label_list = [int(item['@vehicleID']) - 1 for item in all_items]
        _model_label_list = [int(item['@typeID']) - 1 for item in all_items]
        _color_label_list = [int(item['@colorID']) - 1 for item in all_items]
        _camera_label_list = [int(item['@cameraID'][1:]) - 1 for item in all_items]
        
        if relabel:
            new_vids = {vid: new_vid for new_vid, vid in enumerate(sorted(set(_vid_label_list)))}
            _vid_label_list = [new_vids[vid] for vid in _vid_label_list]
        
        dataset = [(osp.join(image_path, img_file), vid, cam_id) for img_file, vid, cam_id in
                   zip(_image_list, _vid_label_list, _camera_label_list)]
        
        return dataset
    
    def _process_txt(self, txt_path, image_path, relabel=False, add_mask=False, num_instance=None):
        _image_list = []
        _vid_label_list = []
        _camera_label_list = []
        all_pids = {}
        all_names = {}
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split(' ')
                img_name, vid_label, camera_label = line[0], int(line[1]) - 1, int(line[2]) - 1
                if vid_label not in all_names.keys():
                    all_names[vid_label] = []
                all_names[vid_label].append((img_name, camera_label))
                
        for vid_label, names in all_names.items():
            if num_instance is not None and 'train' not in txt_path:
                if len(names) < num_instance:
                    continue
                else:
                    for img_name, camera_label in names:
                        if relabel:
                            if vid_label not in all_pids.keys():
                                all_pids[vid_label] = len(all_pids)
                        else:
                            if vid_label not in all_pids.keys():
                                all_pids[vid_label] = vid_label
                        vid = all_pids[vid_label]
                        _image_list.append(img_name)
                        _vid_label_list.append(vid)
                        _camera_label_list.append(camera_label)
            else:
                for img_name, camera_label in names:
                    if relabel:
                        if vid_label not in all_pids.keys():
                            all_pids[vid_label] = len(all_pids)
                    else:
                        if vid_label not in all_pids.keys():
                            all_pids[vid_label] = vid_label
                    vid = all_pids[vid_label]
                    _image_list.append(img_name)
                    _vid_label_list.append(vid)
                    _camera_label_list.append(camera_label)

        _model_label_list = [0] * len(_image_list)
        _color_label_list = [0] * len(_image_list)
        
        dataset = [(osp.join(image_path, img_file), vid, cam_id) for img_file, vid, cam_id in
                   zip(_image_list, _vid_label_list, _camera_label_list)]
        
        return dataset