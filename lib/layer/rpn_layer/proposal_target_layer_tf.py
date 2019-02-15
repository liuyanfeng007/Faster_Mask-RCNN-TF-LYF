import numpy as np
import numpy.random as npr

from lib.utils.config import cfg
from lib.utils.bbox_transform import bbox_transform
from lib.utils.cython_bbox import bbox_overlaps

DEBUG = False

#传入的数据为
# rpn_rois：blob，内容为[proposal引索(全零)，proposal]
#gt_boxes：gtound-truth
#_num_classes：类别总数，21
#函数作用：
# 产生筛选后的roi，对应labels，三个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），
# 另两个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
def proposal_target_layer(rpn_rois, gt_boxes,_num_classes):
    all_rois = rpn_rois

    # 建立一个（gt_boxes.shape[0]+proposals.shape[0],5）的array，proposals信息在上GT信息在下面，存入all_rois，GT引索全初始化为0
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)

    # 由于gt_boxes是有5列信息的（x1,y1,x2,y2,标签），此时只取前4个（gt_boxes[:, :-1]）即位置信息，存入all_rois
    all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))

    #all_rois的第一列全为0
    assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

    num_images = 1
    #cfg.TRAIN.BATCH_SIZE为128
    #设定每张图片上roi个数128
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images

    #cfg.TRAIN.FG_FRACTION=0.25
    #设定每张图片上前景roi个数128/4=32
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
    #产生这几个参数
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(all_rois, gt_boxes, fg_rois_per_image,rois_per_image, _num_classes)

    # 除了labels从一维变成二维，其余的本身就是这个shape
    rois = rois.reshape(-1, 5)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    #就是在对应位置>0的置1,其实跟bbox_inside_weights是一样的
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    #print(bbox_targets[0])
    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

#函数作用，产生两个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），
# 另一个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    #取标签
    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    #生成一个全零矩阵，大小（len(rois),4*21）
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)

    #生成一个全零矩阵，大小同样为（len(rois),4*21）
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)

    #取出fg-roi的index，np.where返回的是一个tuple，tuple里存的是array，所以用[0]来去掉tuple外套
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        #对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh）
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        #对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

#函数作用，返回[标签，dx,dy,dw,dh]
def _compute_targets(ex_rois, gt_rois, labels):
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    #返回anchor相对于GT的（dx,dy,dw,dh）四个回归值，shape（len（rois），4）
    targets = bbox_transform(ex_rois, gt_rois)

    # cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED为False
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))/ np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    # 注意，labels传进来是（len（rois），）大小的，labels[:, np.newaxis]将转换成（len（rois），1）大小，之后与targets合并成（len（rois），5）大小
    # 内容信息为：[标签，dx,dy,dw,dh]
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

#函数作用：对rois进一步筛选，产生对应labels，生成bbox_targets, bbox_inside_weights两个（len(rois),4*21）矩阵
#内容：bbox_targets（对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh））
#bbox_inside_weights（对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1））
def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    # 生成包含前景和背景的RoI的随机样本
    # overlaps: (rois x gt_boxes)
    # bbox_overlaps返回一个N*K的array，N为roi的个数，K为GT个数
    # 对应元素（n，k）存的是第n个roi与第k个GT的：重叠面积/（roi面积+GT面积-重叠面积）
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

    # 以横向相比较，取最大值引索，对比结果为每一个ROI找到与其重叠最好的GT，shape（len（all_rois）,）
    gt_assignment = overlaps.argmax(axis=1)

    #以横向相比较，取最大值，对比结果为每一个ROI找到与其重叠最好的GT的IOU：重叠面积/（roi面积+GT面积-重叠面积）,shape（len（all_rois）,）
    max_overlaps = overlaps.max(axis=1)

    #得到的标签为GT的第五维，即GT的标签,此时相当于取的是all_rois的标签
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    #cfg.TRAIN.FG_THRESH为0.5
    #找到IOU大于等于0.5的ROI，获得其引索，np.where返回的是一个tuple，存的是一个ndarray，array里是符合条件的ROI引索，所以用[0]，取出ndarray，即脱掉tuple外套
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    #设定的每张图片fg最多为32,此时防止通过max_overlaps >= cfg.TRAIN.FG_THRESH的ROI还过于32，即取两者的最小值
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    #fg_rois_per_this_image = 100
    # Sample foreground regions without replacement
    #此时就是在筛选：如果fg_rois_per_image<fg_inds.size则不会被筛选掉，如果fg_rois_per_image>fg_inds.size
    #则随机筛选出来fg_rois_per_this_image个fg-roi，筛选的结果是index，存入fg_inds
    if fg_inds.size > 0:
        if fg_inds.size >= fg_rois_per_image:
        #fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=True)

    #同fg一样的方式，筛选出bg-roi
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    #cfg.TRAIN.BG_THRESH_HI=0.5,cfg.TRAIN.BG_THRESH_LO=0.1
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &(max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

    # Sample background regions without replacement
    if bg_inds.size > 0:
        #bg_inds = npr.choice(bg_inds, size=np.longlong(bg_rois_per_this_image), replace=False)
        bg_inds = npr.choice(bg_inds, size=np.longlong(20), replace=True)

    # The indices that we're selecting (both fg and bg)
    #bg_inds, bg_inds顺次存入keep_inds
    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    #取出经过筛选后的roi的标签
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    #前面的fg_rois_per_this_image个roi为fg-roi，之后的为bg-roi，所以把bg-roi标签设置为0
    labels[fg_rois_per_this_image:] = 0

    #取出经过筛选后的roi的信息，存入rois
    rois = all_rois[keep_inds]

    #传入值为rois的（x1,y1,x2,y2）,对应最佳匹配GT的（x1,y1,x2,y2），对应的labels
    #返回[标签，dx,dy,dw,dh]，shape：（len（rois），5）
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # 内容：bbox_targets（对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh））
    # bbox_inside_weights（对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1））
    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)
    #print("--------------------------------------------------------------------------")
    #print(rois[0])
    #print(labels[0])
    #print("--------------------------------------------------------------------------")
    return labels, rois, bbox_targets, bbox_inside_weights