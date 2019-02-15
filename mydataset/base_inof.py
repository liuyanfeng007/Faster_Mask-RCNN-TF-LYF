import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
from lib.utils.config import cfg

class base_inof():
    def __init__(self,base_path = ""):
        #对应图片名字一览的文件
        self._image_set = "val"
        #根目录
        self._base_path = base_path
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

        #文件一览
        self.image_index = []
        #取得图片一栏
        self.__get_files()
        self.__roidb = [self.__load_file_data(index) for index in self.image_index]

    @property
    def image_lenght(self):
        return len(self.image_index)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def roidb(self):
        return self.__roidb

    @roidb.setter
    def roidb(self,value):
        self.__roidb = value

    def get_roidb(self,index = 0):
        return self.__roidb[index]

    def __get_files(self):
        if cfg.FLAGS.mask == 1:
            image_set_file = os.path.join(self._base_path, 'ImageSets', 'Segmentation', self._image_set + '.txt')
        else:
            image_set_file = os.path.join(self._base_path, 'ImageSets', 'Main', self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            self.image_index = [x.strip() for x in f.readlines()]

    #读取文件配置信息
    def __load_file_data(self,file):
        filename = os.path.join(self._base_path, 'Annotations', file + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        if not self.config['use_diff']:
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        image = os.path.join(self._base_path, 'JPEGImages',file+'.jpg')
        return {'boxes': boxes,'gt_classes': gt_classes,'gt_overlaps': overlaps,'flipped': False,'seg_areas': seg_areas,'image':image}

if __name__ == '__main__':
    pass