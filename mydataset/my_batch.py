from lib.utils.config import cfg
import numpy.random as npr
import numpy as np
import cv2
from lib.utils.config import cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob,prep_im_for_blob_mask

import os
import os.path as osp

import pickle
import numpy as np
import PIL

from mydataset.base_inof import base_inof

base_path = cfg.BASE_PATH
class my_batch():
    def __init__(self,name = "Faster"):
        self.name = name
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            self.data_inof = roidb
        else:
            self.data_inof = base_inof(cfg.BASE_PATH)
            with open(cache_file, 'wb') as fid:
                pickle.dump(self.data_inof, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        self.current_i = -1
        self.prepare_roidb()
    @property
    def cache_path(self):
        #cache_path = osp.abspath(osp.join(cfg.FLAGS2["data_dir"], 'cache'))
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def __get_index(self):
        self.current_i += 1
        if self.current_i >=  self.data_inof.image_lenght:
            self.current_i = 0
        return self.current_i

    def get_next(self):
        num_images = 1
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)

        assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.TRAIN.BATCH_SIZE)
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        #fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        roidb = self.data_inof.get_roidb(self.__get_index())
        im_blob, im_scales = self.get_image_blob(roidb, random_scale_inds)
        # 画像
        blobs = {'data': im_blob}
        masks = []
        #to get data of mask
        if cfg.FLAGS.mask == 1:
            masks = self._get_mask_blob(roidb, random_scale_inds, im_blob.shape[1], im_blob.shape[2])
        if cfg.TRAIN.HAS_RPN:

            # gt boxes: (x1, y1, x2, y2, cls)
            gt_inds = np.where(roidb['gt_classes'] != 0)[0]
            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            gt_boxes[:, 0:4] = roidb['boxes'][gt_inds, :] * im_scales[0]
            gt_boxes[:, 4] = roidb['gt_classes'][gt_inds]
            blobs['gt_boxes'] = gt_boxes
            blobs['gt_masks'] = masks
            blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],dtype=np.float32)
        return blobs
    def _get_mask_blob(self,roidb,scale_inds,widh,hight):
        num_images = len(roidb)
        from PIL import Image
        inof = cv2.imread(roidb['image'].replace("JPEGImages","SegmentationClass").replace(".jpg",".png"))
        im = Image.open(roidb['image'].replace("JPEGImages","SegmentationClass").replace(".jpg",".png"))
        target_size = cfg.TRAIN.SCALES[scale_inds[0]]
        nBoxs = roidb['boxes'].shape[0]
        masks = np.zeros([nBoxs,widh,hight], dtype=np.uint8)
        for ii in range(nBoxs):
            mask = np.zeros([inof.shape[0], inof.shape[1]], dtype=np.uint8)
            x1,y1,x2,y2 = roidb['boxes'][ii,0:4]
            index = roidb['gt_classes'][ii]
            for i in range(x1,x2):
                for j in range(y1,y2):
                    at_pixel = im.getpixel((i, j))
                    if at_pixel == index:
                        mask[j,i] = at_pixel
            if roidb['flipped']:
                mask = mask[:, ::-1]
            masks[ii] = prep_im_for_blob_mask(mask,target_size, cfg.TRAIN.MAX_SIZE)
        return masks
    def get_image_blob(self,roidb, scale_inds):
        processed_ims = []
        im_scales = []
        im = cv2.imread(roidb['image'])
        print(roidb['image'])
        if roidb['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[0]]
        im, im_scale = self.prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)

        return blob, im_scales
    def prep_im_for_blob(self,im, pixel_means, target_size, max_size):
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    def prepare_roidb(self):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        roidb = self.data_inof.roidb
        for i in range(len(self.data_inof.image_index)):
            #roidb[i]['image'] = imdb.image_path_at(i)
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            self.data_inof.roidb[i]['max_classes'] = max_classes
            self.data_inof.roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)

if __name__ == '__main__':
    my_data = my_batch()
    bloss = my_data.get_next()
    pass