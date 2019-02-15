import tensorflow as tf

from .model import model
#分類の数
n_classes = 21
#何分の一【max_poolの処理結果】
_feat_stride = [16,]
#一つの点で、作成の四角枠
anchor_scales = [8, 16, 32]

class Mask_test_model(model):
    def __init__(self, trainable=False):
        #[画像の数：１,高さ,広さ,ＲＧＢ]
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        #[画像の数：１,高さ,広さ,３]
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        #入力データをlayersに保存する
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        #訓練
        self.trainable = trainable

        self.create_model()

        # create ops and placeholders for bbox normalization process
        #正則化
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
        #高さ、広さが２分の１になる
        self.get_dt("conv1_2").max_pool(2, 2, 2, 2, padding='VALID', name='pool1')

        self.get_dt("pool1").conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
        self.get_dt("conv2_1").conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
        #高さ、広さが４分の１になる
        self.get_dt("conv2_2").max_pool(2, 2, 2, 2, padding='VALID', name='pool2')

        self.get_dt("pool2").conv(3, 3, 256, 1, 1, name='conv3_1', trainable=False)
        self.get_dt("conv3_1").conv(3, 3,256, 1, 1, name='conv3_2', trainable=False)
        self.get_dt("conv3_2").conv(3, 3,256, 1, 1, name='conv3_3', trainable=False)
        #高さ、広さが８分の１になる
        self.get_dt("conv3_3").max_pool(2, 2, 2, 2, padding='VALID', name='pool3')

        self.get_dt("pool3").conv(3, 3, 512, 1, 1, name='conv4_1', trainable=False)
        self.get_dt("conv4_1").conv(3, 3,512, 1, 1, name='conv4_2', trainable=False)
        self.get_dt("conv4_2").conv(3, 3,512, 1, 1, name='conv4_3', trainable=False)
        #高さ、広さが１６分の１になる
        self.get_dt("conv4_3").max_pool(2, 2, 2, 2, padding='VALID', name='pool4')

        self.get_dt("pool4").conv(3, 3, 512, 1, 1, name='conv5_1', trainable=False)
        self.get_dt("conv5_1").conv(3, 3,512, 1, 1, name='conv5_2', trainable=False)
        self.get_dt("conv5_2").conv(3, 3,512, 1, 1, name='conv5_3', trainable=False)
        #经过上面错操作后，图片变成原大小的十六分之一　高五百一十二

        self.get_dt("conv5_3").conv(3, 3,512, 1, 1, name='rpn_conv/3x3')
        # anchor_scales为3个尺度，又有3个比例1：1,1：2,2：1,所以*3,又score为2个得分，所以*2
        self.get_dt("rpn_conv/3x3").conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', active = "", name='rpn_cls_score')

        # 回归bbox,存的是（dx,dy,dw,dh）
        #3个尺度，又有3个比例1：1, 1：2, 2：1, 所以 * 3, 又（dx,dy,dw,dh）为4个，所以 * 4
        self.get_dt("rpn_conv/3x3").conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID',  active = "", name='rpn_bbox_pred')

        # ========= RoI Proposal ============
        # 先reshape后softmax激活
        (self.get_dt('rpn_cls_score').reshape_layer(2, name='rpn_cls_score_reshape')# 形状shape(1,9n,n,2)
        .softmax(name='rpn_cls_prob')) # 形状shape(1,9n,n,2)
        # 再reshape
        (self.get_dt('rpn_cls_prob')# 形状shape(1,9n,n,2)
        .reshape_layer(len(anchor_scales) * 3 * 2,name='rpn_cls_prob_reshape'))
        # 形状shape(1,n,n,18),信息还原成'rpn_cls_score'，刚才两步reshape_layer操作：
        # 1、修改为softmax格式。2、还原rpn_cls_score信息位置格式，只不过内容变为sotfmax得分

        (self.get_dt('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         # 初始得到blob，内容为[proposal引索(全0)，proposal]，shape（proposals.shape[0](即暂存的proposals个数),5）
         # [proposal_layer](https://blog.csdn.net/u014256231/article/details/79698562)
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name = 'rpn_rois'))

        # ========= RCNN ============
        (self.get_dt('conv5_3', 'rpn_rois')
            .roi_pool(7, 7, 1.0 / 16, name='pool_5')
            .fc(4096, name='fc6')
            .fc(4096, name='fc7')
            .fc(n_classes, relu=False, name='cls_score')
            .softmax(name='cls_prob'))

        (self.feed('fc7').fc(n_classes * 4, relu=False, name='bbox_pred'))
       #取得处理mask
        (self.get_dt("pool_5").make_head_mask(n_classes,'mask'))