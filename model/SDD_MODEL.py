import tensorflow as tf
from lib.layer.SmoothL1LossLayer.SmoothL1Loss import _modified_smooth_l1
from lib.utils.config import cfg

from .model import model
#分類の数
n_classes = 21
#何分の一【max_poolの処理結果】
_feat_stride = [16,]
#一つの点で、作成の四角枠
anchor_scales = [16]

class SDD_MODEL(model):
    def __init__(self, trainable=True):
        # [画像の具体データ]【バチ：１、行、列、高さ－ＲＧＢ】
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # [画像の数：１,高さ,広さ,３]
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # 対象の四角枠、分類
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        # 入力データをlayersに保存する
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        # 訓練
        self.trainable = trainable

        self.create_model()
        self.compute_loss()
        # create ops and placeholders for bbox normalization process
        # 正則化
        # 建立weights,biases变量，用tf.assign来更新
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())
            # tf.assign用来更新参数值
            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)
        pass

    def create_model(self):
        self.get_dt("data").conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
        self.get_dt("conv1_1").conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
        # 高さ、広さが２分の１になる
        self.get_dt("conv1_2").max_pool(2, 2, 2, 2, padding='VALID', name='pool1')

        self.get_dt("pool1").conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
        self.get_dt("conv2_1").conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
        # 高さ、広さが４分の１になる
        self.get_dt("conv2_2").max_pool(2, 2, 2, 2, padding='VALID', name='pool2')

        self.get_dt("pool2").conv(3, 3, 256, 1, 1, name='conv3_1', trainable=False)
        self.get_dt("conv3_1").conv(3, 3, 256, 1, 1, name='conv3_2', trainable=False)
        self.get_dt("conv3_2").conv(3, 3, 256, 1, 1, name='conv3_3', trainable=False)
        # 高さ、広さが８分の１になる
        self.get_dt("conv3_3").max_pool(2, 2, 2, 2, padding='VALID', name='pool3')

        self.get_dt("pool3").conv(3, 3, 512, 1, 1, name='conv4_1', trainable=False)
        self.get_dt("conv4_1").conv(3, 3, 512, 1, 1, name='conv4_2', trainable=False)
        self.get_dt("conv4_2").conv(3, 3, 512, 1, 1, name='conv4_3', trainable=False)
        # 高さ、広さが１６分の１になる
        self.get_dt("conv4_3").max_pool(2, 2, 2, 2, padding='VALID', name='pool4')

        self.get_dt("pool4").conv(3, 3, 512, 1, 1, name='conv5_1', trainable=False)
        self.get_dt("conv5_1").conv(3, 3, 512, 1, 1, name='conv5_2', trainable=False)
        self.get_dt("conv5_2").conv(3, 3, 512, 1, 1, name='conv5_3', trainable=False)
        # 经过上面错操作后，图片变成原大小的十六分之一　高五百一十二

        self.get_dt("conv5_3").conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
        ##SSD 各个图层输出 原图的1/16 1/32 1/64 1/128 高:三个位置*4 分类数21   3*4 加 3*21


    def compute_loss(self):
        pass