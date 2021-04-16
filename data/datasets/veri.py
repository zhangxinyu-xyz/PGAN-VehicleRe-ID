import xmltodict
import os.path as osp
from .bases import BaseImageDataset

class VeRi(BaseImageDataset):
    """
    VeRi: https://github.com/VehicleReId/VeRidataset

    Dataset statistics:
    # 50,000 images of 776 vehicles captured by 20 cameras covering an 1.0 km^2 area in 24 hours
    """
    dataset_dir = 'VeRi'
    def __init__(self, root='./data/', verbose=True, add_mask=False, **kwargs):
        super(VeRi, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self._process_xml(osp.join(self.dataset_dir,'train_label.xml'), self.train_dir, relabel=True, add_mask=add_mask)
        gallery = self._process_xml(osp.join(self.dataset_dir,'test_label.xml'), self.gallery_dir, relabel=False)
        query = self._process_txt(osp.join(self.dataset_dir,'name_query.txt'), self.query_dir, relabel=False)

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
        with open(xml_path,'rb') as f:
            xml = xmltodict.parse(f)
        all_items = xml['TrainingImages']['Items']['Item']

        _image_list = [ item['@imageName'] for item in all_items]
        _vid_label_list = [ int(item['@vehicleID'])-1 for item in all_items]
        _model_label_list = [ int(item['@typeID'])-1 for item in all_items]
        _color_label_list = [ int(item['@colorID'])-1 for item in all_items]
        _camera_label_list = [ int(item['@cameraID'][1:])-1 for item in all_items]
        
        if relabel:
            new_vids = {vid:new_vid for new_vid,vid in enumerate(sorted(set(_vid_label_list)))}
            _vid_label_list = [ new_vids[vid] for vid in _vid_label_list ]

        dataset = [ (osp.join(image_path, img_file), vid, cam_id) for img_file, vid, cam_id in zip(_image_list, _vid_label_list, _camera_label_list) ]

        return dataset

    def _process_txt(self, txt_path, image_path, relabel=False, add_mask=False):
        with open(txt_path, 'r') as f:
            _image_list = [ line.strip() for line in f.readlines()]
        
        _vid_label_list = [ int(img_file[:4]) - 1 for img_file in _image_list]
        _camera_label_list = [ int(img_file[6:9]) - 1 for img_file in _image_list]
        _model_label_list = [0] * len(_image_list)
        _color_label_list = [0] * len(_image_list)

        dataset = [ (osp.join(image_path, img_file), vid, cam_id) for img_file, vid, cam_id in zip(_image_list, _vid_label_list, _camera_label_list) ]

        return dataset
