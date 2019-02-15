import tensorflow as tf
import numpy as np

from lib.layer.rpn_layer.proposal_layer_tf import proposal_layer as proposal_layer_py
from lib.layer.rpn_layer.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from lib.layer.rpn_layer.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
import tensorflow.contrib.slim as slim

from lib.layer.masklayer.roi import decode
from lib.layer.masklayer.mask import encode

DEFAULT_PADDING = 'SAME'
class model():
    def __init__(self):
        #中間変量保存
        self.current = []
        self.layers = {}
        pass
    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model "+subkey+ " to "+key)
                        except ValueError:
                            print("ignore "+key)
                            if not ignore_missing:
                                raise
    #取得各个层的数据，根据名字
    def get_dt(self,*names):
        assert len(names)!=0
        self.current = []
        for name in names:
            try:
                date = self.layers[name]
            except KeyError:
                print(self.layers.keys())
                raise KeyError('Unknown layer name fed: %s' % name)
            self.current.append(date)
        return self
    #k_h,核の高さ k_w核の広さ, height画像の高さ, stride_h, stride_w【歩幅】
    def conv(self,k_h, k_w, height, stride_h, stride_w,name,active='relu',padding=DEFAULT_PADDING,group=1,trainable=True):
        self.validate_padding(padding)
        input = self.current[0]
        pic_h = input.shape[-1].value
        assert pic_h%group == 0
        assert height%group == 0
        # 定义函数卷积函数
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride_h, stride_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            # 采取截断是正态初始化权重，这只是一种initializer方法，mean=0,stddev=0.01
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            # 这也只是定义initializer的方法，初始化为0
            init_biases = tf.constant_initializer(0.0)
            kernel = self._weight_variable([k_h, k_w, pic_h / group, height],initializer=init_weights ,trainable=trainable,name='weights')
            # 偏差定义　出力高
            biases = self._bias_variable([height],initializer=init_biases ,trainable=trainable,name='biases')
            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
                pass
        bias = tf.nn.bias_add(conv, biases)

        if active == "relu":
            bias =tf.nn.relu(bias, name=scope.name)
        self.layers[name] = bias
        return self

    def anchor_target_layer(self, _feat_stride, anchor_scales, name):
        input = self.current[0:4]
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        #with tf.variable_scope(name) as scope:
        #动态调用方法的函数　方法名字　入力数据　出力数据格式
        rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])
        #
        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')
        self.layers[name] = rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights
        return self
    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []
        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
                continue
            if v.name == 'vgg_16/fc6/biases:0' or v.name == 'vgg_16/fc7/biases:0':
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore
    #k_h, 核の高さ k_w核の広さ,stride_h, stride_w【歩幅】
    def max_pool(self,k_h, k_w, stride_h, stride_w, name, padding=DEFAULT_PADDING):
        input= self.current[0]
        self.validate_padding(padding)
        max = tf.nn.max_pool(input,ksize=[1, k_h, k_w, 1],strides=[1, stride_h, stride_w, 1],padding=padding,name=name)
        self.layers[name] = max
        return self
    def avg_pool(self,k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        input= self.current[0]
        self.validate_padding(padding)
        avg =  tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)
        self.layers[name] = max

        return self
    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    #weightを初期化
    def _weight_variable(self, shape,initializer=tf.random_normal_initializer(mean=0., stddev=1., ), name='weights',trainable=True):
        return tf.get_variable(shape=shape, initializer=initializer, name=name,trainable=trainable)

    #biasを初期化
    def _bias_variable(self, shape,initializer = tf.constant_initializer(0.1), name='biases',trainable=True):
        return tf.get_variable(name=name, shape=shape, initializer=initializer,trainable=trainable)

    def reshape_layer(self, d, name):
        input= self.current[0]
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            # 还原回rpn_cls_score的信息位置格式
            self.layers[name] = tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],int(d), tf.cast(tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(input_shape[3], tf.float32),tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)
        else:
            # 假设rpn_cls_score.shape为[1,n,n,18],最后reshape成[1,9n,n,2]
            # 假如rpn_cls_score.shape为[1,3,3,18]，元素内容为range（3*3*18），最后得到的形状为[0,81],[1,82],[2,83]..意思为前81个元素（3*3*9）为bg，
            # 后81个元素对应fg，0与36对应着该featuremap的位置i，对应原图的可视野为前景或者背景的概率
            # 当然需要再一步softmax才能给出该可视野为fg与bg的概率
            self.layers[name] = tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],int(d), tf.cast(tf.cast(input_shape[1], tf.float32) * (tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)), tf.int32),input_shape[2]]), [0, 2, 3, 1],name=name)
        return self

    #取得好的图片位置信息和分裂
    def proposal_layer(self,_feat_stride, anchor_scales, cfg_key, name):
        #cfg_key为TRAIN
        input = self.current
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        #input[1] = self.current[1]
        #input[2] = self.current[2]

        # 就是返回blob，内容为[proposal引索(全0)，proposal]，shape（proposals.shape[0],5）,引索(全0)占一列，proposal占4列
        self.layers[name] = tf.reshape(tf.py_func(proposal_layer_py, [input[0], input[1], input[2], cfg_key, _feat_stride, anchor_scales],[tf.float32]), [-1, 5], name=name)
        return self

    #根据名称取得数据
    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer
    def proposal_target_layer(self, classes, name):
        input = self.current
        # input为'rpn_cls_score','gt_boxes','im_info','data'信息组成的一个列表，input[0]为rpn_cls_score信息
        if isinstance(input[0], tuple):
            # input[0]就是'rpn_cls_score'的输出
            input[0] = input[0][0]

        #with tf.variable_scope(name) as scope:
        # 产生筛选后的roi，对应labels，三个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），另两个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1)
        rois,labels,bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            tf.py_func(proposal_target_layer_py,[input[0], input[1],classes],[tf.float32, tf.float32,tf.float32, tf.float32,tf.float32])
        rois = tf.reshape(rois, [-1, 5], name='rois')
        # 要将tf.float32类型转换为tf.Tensor类型
        # tf.convert_to_tensor函数，以确保我们处理张量而不是其他类型
        labels = tf.convert_to_tensor(tf.cast(labels, tf.int32),name='labels')
        bbox_targets = tf.convert_to_tensor(bbox_targets,name='bbox_targets')
        bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
        bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

        self.layers[name] = rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
        return self
    def __transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        x0, y0, x1, y1 ,_ = tf.split(boxes, 5, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)
        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # 通过上面处理选择好的图片位置信息，在初始数据上取得相应数据 faster rcnn 池华
    def roi_pool(self,pooled_height, pooled_width, spatial_scale, name):
        input = self.current[0]
        #if isinstance(input[0], tuple):
        #    input[0] = input[0][0]
        #if isinstance(input[1], tuple):
        #    input[1] = input[1][0]
        #pass
        #self.layers[name] = roi_pool_op.roi_pool(input[0], input[1],pooled_height,pooled_width,spatial_scale,name=name)[0]
        ################################################################################################################
        # 初始数据信息
        rpn_conv = self.current[0]
        # 选择好的图片位置信息
        if self.trainable == True :
            rois = self.current[1][0]
        else:
            rois = self.current[1]
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            # 取得对应图片的位置信息
            bottom_shape = tf.shape(rpn_conv)
            _feat_stride = 16
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(_feat_stride)
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(_feat_stride)

            # 转换成比例
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            #x11 = tf.slice(rois, [0, 1], [-1, 1], name="x1")
            #y11 = tf.slice(rois, [0, 2], [-1, 1], name="y1")
            #x21 = tf.slice(rois, [0, 3], [-1, 1], name="x2")
            #y21 = tf.slice(rois, [0, 4], [-1, 1], name="y2")
            #self.out_box = tf.stop_gradient(tf.concat([x11, y11, x21, y21], axis=1))
            #　阶梯下降对象外
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))

            #通过上面处理选择好的图片位置信息，在初始数据上取得相应数据
            crops = tf.image.crop_and_resize(rpn_conv, bboxes, tf.to_int32(batch_ids),[pooled_height*2,pooled_width*2], name="crops")
            self.layers[name] = slim.max_pool2d(crops, [2, 2], padding='SAME')
        return self

    def fc(self, num_out,name,relu=True,trainable=True):
        input = self.current[0]
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()

            if input_shape.ndims == 4:
                # 数据纬度变化
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))
            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)
            weights = self._weight_variable([dim, num_out],initializer=init_weights,trainable=trainable,name='weights')
            biases = self._bias_variable([num_out], initializer=init_biases, trainable=trainable,name='biases')

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            self.layers[name] = op(feed_in, weights, biases, name=scope.name)
        return self

    # dropout防止过拟合
    def dropout(self,keep_prob, name):
        input= self.current[0]
        self.layers[name] = tf.nn.dropout(input, keep_prob, name=name)
        return self

    #多项逻辑斯特回归
    def softmax(self,name):
        input= self.current[0]
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            # 形状shape(1,9n,n,2)
            # 对前景背景处理softmax
            self.layers[name] = tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),[-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            self.layers[name] = tf.nn.softmax(input, name=name)
        return self
########################################################################################################################
#### mask
    def _filter_negative_samples(self,labels, tensors):
        """keeps only samples with none-negative labels
        Params:
        -----
        labels: of shape (N,)
        tensors: a list of tensors, each of shape (N, .., ..) the first axis is sample number

        Returns:
        -----
        tensors: filtered tensors
        """
        # return tensors
        keeps = tf.where(tf.greater_equal(labels, 0))
        keeps = tf.reshape(keeps, [-1])

        filtered = []
        for t in tensors:
            #tf.assert_equal(tf.shape(t)[0], tf.shape(labels)[0])
            f = tf.gather(t, keeps)
            filtered.append(f)

        return filtered

    # mask rcnn
    def roi_decoder(self,ih, iw,name,scope='ROIDecoder'):
        input = self.current[0]
        boxes = input[0]
        scores = input[1]
        rois =  input[2]
        with tf.name_scope(scope) as sc:
            final_boxes, classes, scores = tf.py_func(decode,[boxes, scores, rois, ih, iw],
                           [tf.float32, tf.int32, tf.float32])
            final_boxes = tf.convert_to_tensor(final_boxes, name='boxes')
            classes = tf.convert_to_tensor(tf.cast(classes, tf.int32), name='classes')
            scores = tf.convert_to_tensor(scores, name='scores')
            final_boxes = tf.reshape(final_boxes, (-1, 4))
        self.layers[name] = final_boxes,classes,scores
        return self

    # mask rcnn 用的池华
    def roi_align_pool(self,pooled_height, pooled_width, spatial_scale, name):
        input = self.current[0]
        # 初始数据信息
        rpn_conv = self.current[0]
        # 选择好的图片位置信息
        rois = self.current[1][0]
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            # 取得对应图片的位置信息
            bottom_shape = tf.shape(rpn_conv)
            _feat_stride = 16
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(_feat_stride)
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(_feat_stride)

            # 转换成比例
            x0 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y0 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x1 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y1 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            spacing_w = (x1 - x0) / tf.to_float(pooled_width)
            spacing_h = (y1 - y0) / tf.to_float(pooled_height)
            nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(width - 1)
            ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(height - 1)
            nw = spacing_w * tf.to_float(pooled_width - 1) / tf.to_float(width - 1)
            nh = spacing_h * tf.to_float(pooled_height - 1) / tf.to_float(height - 1)

            #　阶梯下降对象外
            bboxes = tf.stop_gradient(tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1))

            #通过上面处理选择好的图片位置信息，在初始数据上取得相应数据
            crops = tf.image.crop_and_resize(rpn_conv, bboxes, tf.to_int32(batch_ids), [pooled_height, pooled_width], name="crops")
            self.layers[name] = slim.avg_pool2d(crops, [2, 2], padding='SAME')
        return self

    # make head mask
    def make_head_mask(self,num_classes,name):
        input = self.current[0]
        m = input
        for _ in range(4):
            m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        # to 28 x 28
        m = slim.conv2d_transpose(m, 256, 2, stride=2, padding='VALID', activation_fn=tf.nn.relu)
        tf.add_to_collection('__TRANSPOSED__', m)
        self.layers[name] = slim.conv2d(m, num_classes, [1, 1], stride=1, padding='VALID', activation_fn=None)
        return self

    def mask_encoder(self,num_classes, mask_height, mask_width,name,scope='MaskEncoder'):
        input = self.current
        gt_masks = input[0]
        gt_boxes = input[1]
        rois = input[2][0]
        masks = input[3]
        with tf.name_scope(scope) as sc:
            labels, mask_targets, mask_inside_weights = \
                tf.py_func(encode,[gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width],[tf.float32, tf.int32, tf.float32])
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='classes')
            mask_targets = tf.convert_to_tensor(mask_targets, name='mask_targets')
            mask_inside_weights = tf.convert_to_tensor(mask_inside_weights, name='mask_inside_weights')
            labels = tf.reshape(labels, (-1,))
            mask_targets = tf.reshape(mask_targets, (-1, mask_height, mask_width, num_classes))
            mask_inside_weights = tf.reshape(mask_inside_weights, (-1, mask_height, mask_width, num_classes))
            labels, masks, mask_targets, mask_inside_weights = self._filter_negative_samples(tf.reshape(labels, [-1]), [
                                                                tf.reshape(labels, [-1]),
                                                                masks,
                                                                mask_targets,
                                                                mask_inside_weights,
                                                            ])
            self.layers[name] = labels, masks, mask_targets, mask_inside_weights
        return self
########################################################################################################################
    # 損失計算層
    #@abstractmethod
    #def _compute_cost(self):
    #    pass