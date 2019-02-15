import os
import tensorflow as tf
from tensorflow.python.client import timeline
import time

from lib.utils.config import cfg
from lib.layer.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer
from lib.layer.SmoothL1LossLayer.SmoothL1Loss import _modified_smooth_l1
import numpy as np
from mydataset.my_batch import my_batch

import lib.layer.gt_data_layer.roidb as gdl_roidb
import lib.layer.roi_data_layer.roidb as rdl_roidb

class Do_Train(object):
    def __init__(self, sess, saver, model, imdb, roidb, output_dir, pretrained_model=None):
        self.model = model
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        print('Computing bounding-box regression targets...')
       #True
        if cfg.TRAIN.BBOX_REG:
            #不同类的均值与方差，返回格式means.ravel(), stds.ravel()
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print('done')
        # For checkpoint
        self.saver = saver
    def load_snapshot(self, saver,sess):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            return 0
        checkpoint = tf.train.latest_checkpoint(self.output_dir)
        if checkpoint == None:
            return 0
        else:
            start = np.longlong(checkpoint.split("_")[-1].replace(".ckpt",""))
            saver.restore(sess, checkpoint)
            return start
    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        #net = self.model
        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in self.model.layers.keys():
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(self.model.bbox_weights_assign, feed_dict={self.model.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(self.model.bbox_bias_assign, feed_dict={self.model.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +'_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in self.model.layers.keys():
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(self.model.bbox_weights_assign, feed_dict={self.model.bbox_weights: orig_0})
                sess.run(self.model.bbox_bias_assign, feed_dict={self.model.bbox_biases: orig_1})

    def train_faster_model(self, sess, max_iters):
        """Network training loop."""
        #返回一个RoIDataLayer类对象，内容self._roidb ,self._num_classes ,self._perm,self._cur
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # initialize variables
        sess.run(tf.global_variables_initializer())

        #start = self.load_snapshot(self.saver,sess)

        #如果有预训练模型，则加载
        if self.pretrained_model is not None:
            print ('Loading pretrained model ''weights from {:s}').format(self.pretrained_model)
            self.model.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1

        #记录当前时间
        timer = Timer()
        start = 0

        #在最大循环次数内
        for iter in range(start,max_iters):
            # get one batch
            #得到一个batch信息
            blobs = data_layer.forward()

            # Make one SGD update
            #给定placehold信息
            feed_dict={self.model.data: blobs['data'], self.model.im_info: blobs['im_info'], self.model.keep_prob: 0.5, self.model.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            #False
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([self.model.rpn_cross_entropy, self.model.rpn_loss_box, self.model.cross_entropy, self.model.loss_box, self.model.train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)

            timer.toc()
            #False
            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(np.longlong(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, self.model.lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

    def train_mask_model(self, sess, max_iters):
        """Network training loop."""
        #返回一个RoIDataLayer类对象，内容self._roidb ,self._num_classes ,self._perm,self._cur
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        #blobs = data_layer.forward()
        # initialize variables
        sess.run(tf.global_variables_initializer())

        start = self.load_snapshot(self.saver,sess)
        #如果有预训练模型，则加载
        #if self.pretrained_model is not None:
        #    print ('Loading pretrained model ''weights from {:s}').format(self.pretrained_model)
        #    self.model.load(self.pretrained_model, sess, self.saver, True)

        #
        last_snapshot_iter = -1

        #记录当前时间
        timer = Timer()

        #在最大循环次数内
        for iter in range(start,max_iters):
            # get one batch
            #得到一个batch信息
            blobs = data_layer.forward()

            # Make one SGD update
            #给定placehold信息
            feed_dict={self.model.data: blobs['data'], self.model.im_info: blobs['im_info'], self.model.keep_prob: 0.5, self.model.gt_boxes: blobs['gt_boxes'],self.model.get_masks: blobs['gt_masks']}

            run_options = None
            run_metadata = None
            #False
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, mask_loss_value,_ = sess.run([self.model.rpn_cross_entropy, self.model.rpn_loss_box, self.model.cross_entropy, self.model.loss_box, self.model.mask_loss ,self.model.train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)

            timer.toc()
            #False
            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(np.longlong(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mask_loss:%.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value+mask_loss_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value,mask_loss_value, self.model.lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)
        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb
#返回一个RoIDataLayer类对象，内容self._roidb ,self._num_classes ,self._perm,self._cur
def get_data_layer(roidb, num_classes):
    """return a data layer."""
    #False
    if cfg.TRAIN.HAS_RPN:
        #False
        if cfg.IS_MULTISCALE:
            #layer = GtDataLayer(roidb)
            layer = None
            pass
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer
def filter_roidb(roidb):
    #筛选掉没有前景也没有背景的rois
    """Remove roidb entries that have no usable RoIs."""
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        #overlaps就是一个one-hot编码，有分类物体的就在该分类位置上置1（包括背景），所以可以通过一下函数找到有只是一个前景背景物体的图片
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &(overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        #如果至少有一个前景或者背景即返回True
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid
    #roidb列表中元素（字典）的长度，即有多少个图片信息
    num = len(roidb)
    #记录筛选后的roidb
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    #筛选后的roid数目
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,num, num_after))
    return filtered_roidb

#network为VGGnet_train一个对象，imdb为pascal_voc对象，roidb为一个列表
def train_faster_model(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    #筛选roidb（至少有一个前景或者背景的图片）
    roidb = filter_roidb(roidb)
    #对参数进行保存，100次迭代更新一次
    saver = tf.train.Saver(max_to_keep=100)
    #建立对话，对于tf.ConfigProto有以下选项
    #log_device_placement=True : 是否打印设备分配日志
    #allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #Do_Train,添加了self.saver, self.model, self.imdb, self.roidb,self.output_dir, self.pretrained_model，
        #以及roidb['bbox_targets'](标准化后的), self.bbox_means, self.bbox_stds信息
        Do_ = Do_Train(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print('Solving...')
        Do_.train_faster_model(sess, max_iters)
        print('done solving')

#network为VGGnet_train一个对象，imdb为pascal_voc对象，roidb为一个列表
def train_mask_model(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    #筛选roidb（至少有一个前景或者背景的图片）
    roidb = filter_roidb(roidb)
    #对参数进行保存，100次迭代更新一次
    saver = tf.train.Saver(max_to_keep=100)
    #建立对话，对于tf.ConfigProto有以下选项
    #log_device_placement=True : 是否打印设备分配日志
    #allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #Do_Train,添加了self.saver, self.model, self.imdb, self.roidb,self.output_dir, self.pretrained_model，
        #以及roidb['bbox_targets'](标准化后的), self.bbox_means, self.bbox_stds信息
        Do_ = Do_Train(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print('Solving...')
        Do_.train_mask_model(sess, max_iters)
        print('done solving')

#network为VGGnet_train一个对象，imdb为pascal_voc对象，roidb为一个列表
def my_train_faster_model(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    #筛选roidb（至少有一个前景或者背景的图片）
    roidb = filter_roidb(roidb)
    #对参数进行保存，100次迭代更新一次
    saver = tf.train.Saver(max_to_keep=100)
    #建立对话，对于tf.ConfigProto有以下选项
    #log_device_placement=True : 是否打印设备分配日志
    #allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #Do_Train,添加了self.saver, self.model, self.imdb, self.roidb,self.output_dir, self.pretrained_model，
        #以及roidb['bbox_targets'](标准化后的), self.bbox_means, self.bbox_stds信息
        Do_ = Do_Train(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print('Solving...')
        Do_.train_faster_model(sess, max_iters)
        print('done solving')

