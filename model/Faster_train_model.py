import tensorflow as tf
from lib.layer.SmoothL1LossLayer.SmoothL1Loss import _modified_smooth_l1
from lib.utils.config import cfg

from .model import model
#分類の数
n_classes = 21
#何分の一【max_poolの処理結果】
_feat_stride = [16,]
#一つの点で、作成の四角枠
anchor_scales = [8, 16, 32]

class VOC_train_model(model):
    def __init__(self, trainable=True):

        #[画像の具体データ]【バチ：１、行、列、高さ－ＲＧＢ】
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        #[画像の数：１,高さ,広さ,３]
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        #対象の四角枠、分類
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        #入力データをlayersに保存する
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        #訓練
        self.trainable = trainable

        self.create_model()
        self.compute_loss()
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
        self.get_dt("data").conv(3, 3, 64, 1, 1, name='vgg_16/conv1/conv1_1', trainable=False)
        self.get_dt("vgg_16/conv1/conv1_1").conv(3, 3, 64, 1, 1, name='vgg_16/conv1/conv1_2', trainable=False)
        #高さ、広さが２分の１になる
        self.get_dt("vgg_16/conv1/conv1_2").max_pool(2, 2, 2, 2, padding='VALID', name='pool1')

        self.get_dt("pool1").conv(3, 3, 128, 1, 1, name='vgg_16/conv2/conv2_1', trainable=False)
        self.get_dt("vgg_16/conv2/conv2_1").conv(3, 3, 128, 1, 1, name='vgg_16/conv2/conv2_2', trainable=False)
        #高さ、広さが４分の１になる
        self.get_dt("vgg_16/conv2/conv2_2").max_pool(2, 2, 2, 2, padding='VALID', name='pool2')

        self.get_dt("pool2").conv(3, 3, 256, 1, 1, name='vgg_16/conv3/conv3_1', trainable=False)
        self.get_dt("vgg_16/conv3/conv3_1").conv(3, 3,256, 1, 1, name='vgg_16/conv3/conv3_2', trainable=False)
        self.get_dt("vgg_16/conv3/conv3_2").conv(3, 3,256, 1, 1, name='vgg_16/conv3/conv3_3', trainable=False)
        #高さ、広さが８分の１になる
        self.get_dt("vgg_16/conv3/conv3_3").max_pool(2, 2, 2, 2, padding='VALID', name='pool3')

        self.get_dt("pool3").conv(3, 3, 512, 1, 1, name='vgg_16/conv4/conv4_1', trainable=self.trainable)
        self.get_dt("vgg_16/conv4/conv4_1").conv(3, 3,512, 1, 1, name='vgg_16/conv4/conv4_2', trainable=self.trainable)
        self.get_dt("vgg_16/conv4/conv4_2").conv(3, 3,512, 1, 1, name='vgg_16/conv4/conv4_3', trainable=self.trainable)
        #高さ、広さが１６分の１になる
        self.get_dt("vgg_16/conv4/conv4_3").max_pool(2, 2, 2, 2, padding='VALID', name='pool4')

        self.get_dt("pool4").conv(3, 3, 512, 1, 1, name='vgg_16/conv5/conv5_1', trainable=self.trainable)
        self.get_dt("vgg_16/conv5/conv5_1").conv(3, 3,512, 1, 1, name='vgg_16/conv5/conv5_2', trainable=self.trainable)
        self.get_dt("vgg_16/conv5/conv5_2").conv(3, 3,512, 1, 1, name='vgg_16/conv5/conv5_3', trainable=self.trainable)
        #经过上面错操作后，图片变成原大小的十六分之一　高五百一十二

        self.get_dt("vgg_16/conv5/conv5_3").conv(3, 3,512, 1, 1, name='rpn_conv/3x3')

########################################################################################################################
        # 生成每个点对应9个小图片的分类，也就是图片的高度发生变更
        # anchor_scales为3个尺度，又有3个比例1：1,1：2,2：1,所以*3,又score为2个得分，所以*2
        # 经过上面错操作后，图片变成原大小的十六分之一　高3*3*2
        # 每一个点对应3类型图，每类新3个，所以是9个图片，再加上每一个图片对应连个分类。所以是3*3*2
        self.get_dt("rpn_conv/3x3").conv(1,1,len(anchor_scales)*3*2,1,1,padding='VALID',active ="",name='rpn_cls_score')

        # 'rpn-data'层给出了rpn中的所有信息，包括图片anchor（标签-1,0,1），回归值（dx，dy，dw，dh），两个权重
        # data：为图片像素信息，info：im_info[0]存的是图片像素行数即高，im_info[1]存的是图片像素列数即宽
        # gt_boxes：GT信息，为5维度，前四个为（xmin，ymin）（xmax，ymax），最后一个为标签
        # rpn_cls_score就把他看成一个简单的feature-map，深度为3*3*2,对应3比例*3尺度*2分类值
        # [anchor_target_layer](https://blog.csdn.net/u014256231/article/details/79698070)
        # 返回每一点对应的框，并一定的派出，和对应的LABEL　前景背景
        self.get_dt('rpn_cls_score', 'gt_boxes', 'im_info', 'data').anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data')
########################################################################################################################
        # 生成每一点对应就个图片的坐标，所以是３*３* 4
        # 回归bbox,存的是（dx,dy,dw,dh）
        #3个尺度，又有3个比例1：1, 1：2, 2：1, 所以 * 3, 又（dx,dy,dw,dh）为4个，所以 * 4
        # 经过上面错操作后，图片变成原大小的十六分之一　高3*3*4
        self.get_dt("rpn_conv/3x3").conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID',  active = "", name='rpn_bbox_pred')
########################################################################################################################
        # ========= RoI Proposal ============
        # 先reshape后softmax激活
        # 1,h,w,18 ----> 1,9w,h,2   18：深度为3*3*2
        # 分类前景背景
        (self.get_dt('rpn_cls_score').reshape_layer(2, name='rpn_cls_score_reshape').get_dt('rpn_cls_score_reshape')# 形状shape(1,9n,n,2)
        .softmax(name='rpn_cls_prob')) # 形状shape(1,9n,n,2)

        # 再reshape
        # 1,9w,h,2 ---> 1,h,w,18
        (self.get_dt('rpn_cls_prob')# 形状shape(1,9n,n,2)
        .reshape_layer(len(anchor_scales) * 3 * 2,name='rpn_cls_prob_reshape'))

        # 形状shape(1,n,n,18),信息还原成'rpn_cls_score'，刚才两步reshape_layer操作：
        # 1、修改为softmax格式。2、还原rpn_cls_score信息位置格式，只不过内容变为sotfmax得分
        (self.get_dt('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         # 初始得到blob，内容为[proposal引索(全0)，proposal]，shape（proposals.shape[0](即暂存的proposals个数),5）
         # [proposal_layer](https://blog.csdn.net/u014256231/article/details/79698562)
         # 根据上面的softmax结果的值排序，取得前多少个图片框，切根据图片的偏移信息'rpn_bbox_pred'，对图片位置大小稍微调整
         .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois'))

        # 产生筛选后的roi，对应labels，三个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），
        # 另两个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
        # [proposal_target_layer](https://blog.csdn.net/u014256231/article/details/79698825)
        #　根据上面的操作得到的图片位置信息，和实际的对象图片信息。对其类别划分，并进一步减少预选图片信息
        (self.get_dt('rpn_rois','gt_boxes').proposal_target_layer(n_classes,name = 'roi-data'))
        # ========= RCNN ============
        (self.get_dt('vgg_16/conv5/conv5_3', 'roi-data')
            .roi_pool(7, 7, 1.0 / 16, name='pool_5').get_dt('pool_5')#上面取得的图片位置信息包含分类，在conv5_3数据上取得
            .fc(4096/2, name='vgg_16/fc6').get_dt('vgg_16/fc6')#每个图片数据全连结处理２５６
            .dropout(0.5, name='drop6').get_dt('drop6')
            .fc(4096/2, name='vgg_16/fc7').get_dt('vgg_16/fc7')#每个图片数据全连结处理２５６
            .dropout(0.5, name='drop7').get_dt('drop7')
            .fc(n_classes, relu=False, name='cls_score').get_dt('cls_score')#对每个图片信息产生分类信息
            .softmax(name='cls_prob'))
        # 对每个图片信息产生分位置信息
        (self.get_dt('drop7').fc(n_classes * 4, relu=False, name='bbox_pred'))
    def compute_loss(self):
        # RPN
        # classification loss
        # 将'rpn_cls_score_reshape'层的输出（1,n，n，18）reshape为（-1,2）,其中2为前景与背景的多分类得分（）
        # 每一点对应９个图片信息的前景背景得分
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])

        # 'rpn-data'层输出的[0]为rpn_label,shape为(1, 1, A * height, width)，中存的是所有anchor的label（-1,0,1）
        # 问题1：目前感觉有异议，数据读取方向labels有问题################################
        # 每一个点对应９个图片的所有前景背景分类
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])

        # 删除不等于-1的值
        # 把rpn_label不等于-1对应引索的rpn_cls_score取出，重新组合成rpn_cls_score
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
        # 把rpn_label不等于-1对应引索的rpn_label取出，重新组合成rpn_label
        rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])

        # score损失：tf.nn.sparse_softmax_cross_entropy_with_logits函数的两个参数logits，labels数目相同（shape[0]相同），分别为最后一层的输出与标签
        # NOTE：这个函数返回的是一个向量，要求交叉熵就tf.reduce_sum，要求损失就tf.reduce_mean
        # 问题2：logits，labels应该shape相同的，但这里不同，有异议
        ##################训练每个点对应９个图片是前景还是背景
        # 前景背景损失函数
        self.rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
#####################################################################

        # bounding box regression L1 loss
        # 'rpn_bbox_pred'层为了回归bbox,存的是（dx,dy,dw,dh）
        # 每个点对应９个图片的位置信息，其实是位置偏移信息，参看
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')

        # 'rpn-data'[1]返回一个用于anchor回归成target的包含每个anchor回归值(dx、dy、dw、dh)的array,形状（(len(inds_inside), 4），即（anchors.shape[0],4）
        # 重新reshape成(1, height, width, A * 4)
        # 取得每个点对应９个图片的位置信息
        rpn_bbox_targets = tf.transpose(self.get_output('rpn-data')[1], [0, 2, 3, 1])

        # rpn_bbox_inside_weights：标签为1的anchor,对应(1.0, 1.0, 1.0, 1.0)
        # 重新reshape成(1, height, width, A * 4)
        # 取得每个点对应９个图片的位置信息的内部信息，不是很理解
        rpn_bbox_inside_weights = tf.transpose(self.get_output('rpn-data')[2], [0, 2, 3, 1])

        # rpn_bbox_outside_weights:标签为0或者1的，权重初始化都为（1/num_examples，1/num_examples，1/num_examples，1/num_examples），num_examples为标签为0或者1的anchor总数
        # 重新reshape成(1, height, width, A * 4)
        # 取得每个点对应９个图片的位置信息的外部信息，不是很理解
        rpn_bbox_outside_weights = tf.transpose(self.get_output('rpn-data')[3], [0, 2, 3, 1])

        # 计算smooth_l1损失
        # 图片位置损失函数，不包含分类
        rpn_smooth_l1 = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,rpn_bbox_outside_weights)

        # rpn_smooth_l1计算出的为一个向量，现在要合成loss形式
        # 位置损失函数
        self.rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
########################################################################

        # R-CNN
        # classification loss
        # 得到最后一个score分支fc层的输出
        # 选定好的图片的预测分类信息
        self.cls_score = self.get_output('cls_score')

        # label：筛选出的proposal与GT结合形成all_roi,从all_roi中筛选出符合的roi，得到这些roi的label
        # 选定好的图片的分类结果
        self.label = tf.reshape(self.get_output('roi-data')[1], [-1])

        # 用这些roi的label与最后一个score分支fc层的输出相比较，得到loss
        # 图片类型损失函数
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_score, labels=self.label))

########################################################################

        # bounding box regression L1 loss
        # 得到最后一个bbox分支fc层的输出
        # 选定图片的预测位置信息
        bbox_pred = self.get_output('bbox_pred')
        # 选定图片的位置结果信息
        bbox_targets = self.get_output('roi-data')[2]
        # 选定图片的内部信息
        bbox_inside_weights = self.get_output('roi-data')[3]
        # 选定图片的外部信息
        bbox_outside_weights = self.get_output('roi-data')[4]

        # 计算smooth_l1损失
        smooth_l1 = _modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        # smooth_l1计算出的为一个向量，现在要合成loss形式
        # 选定图片的位置损失函数
        self.loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        # final loss
        # 总loss　　几乎说有图片的背景前景损失函数位置损失函数　选定图片的分类损失函数位置损失函数
        self.loss = self.cross_entropy + self.loss_box + self.rpn_cross_entropy + self.rpn_loss_box

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)

        # cfg.TRAIN.LEARNING_RATE为0.001,  cfg.TRAIN.STEPSIZE为50000
        # tf.train.exponential_decay（初始lr，初始步数，多少步进入下一平台值，总步数，下一次平台值是多少（基于上次的比率），staircase）
        # staircase为True则遵循刚才规则，如为False则每一次迭代更新一次
        self.lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,cfg.TRAIN.STEPSIZE, 0.9, staircase=True)

        # cfg.TRAIN.MOMENTUM 为 0.9
        momentum = cfg.TRAIN.MOMENTUM

        # 动态系数为0.9的梯度下降法
        self.train_op = tf.train.MomentumOptimizer(self.lr, momentum).minimize(self.loss, global_step=global_step)


