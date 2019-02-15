#import _init_paths
#from model.train import get_training_roidb, train_faster_model,train_mask_model
from model.new_train import get_training_roidb, train_faster_model,train_mask_model
from mydataset.my_batch import my_batch

from lib.utils.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from lib.datasets.factory import get_imdb
from model.model_factory import get_model
import argparse
import pprint
import numpy as np
import sys

#设置参数，dest为目标，可通过args.XXX来访问
#通过命令行调用/experiments/scripts/faster_rcnn_end2end.sh，就是在设置参数
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',default=0, type=int)
    parser.add_argument('--solver', dest='solver',help='solver prototxt',default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',help='number of iterations to train',default=700000, type=int)
    #parser.add_argument('--weights', dest='pretrained_model',help='initialize with pretrained model weights',default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file',default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',help='dataset to train on',default='voc_2007_train', type=str)
    parser.add_argument('--rand', dest='randomize',help='randomize (do not use a fixed seed)',action='store_true')


    #faster rcnn
    parser.add_argument('--network', dest='network_name',help='name of the network',default='network_train', type=str)
    #mask rcnn
    #parser.add_argument('--network', dest='network_name', help='name of the network', default='network_mask', type=str)
    parser.add_argument('--pretrained_model', dest='pretrained_model',help='pretrained model', default='C:\\liuyf\\mysite\\Faster_Mask-RCNN-TF-LYF\\data\\imagenet_weights\\vgg_16.ckpt',nargs=argparse.REMAINDER)
    parser.add_argument('--set', dest='set_cfgs',help='set config keys', default=None,nargs=argparse.REMAINDER)
    parser.add_argument('--data_dir', dest='data_dir',help='data path', default='C:\\liuyf\\mysite\\Faster_Mask-RCNN-TF-LYF\\data\\',nargs=argparse.REMAINDER)

    #如果sys.argv长度为1,则说明没有参数传入，系统会退出
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    #如果还有其他配置文件，就加载
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    #已知类型的前提下，可以使用pprint来标准打印
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    #imdb为存在一个字典(easydict)里的pascal_voc类的一个对象，e.g.{voc_2007_train:内容，voc_2007_val:内容，voc_2007_test:内容,voc_2007_test:内容,voc_2012_train:内容...}
    #内容里有该类里的各种self名称与操作，包括roi信息等等
    #get_imdb函数在/lib/datasets/factory.py中:
    #[factor.py](https://blog.csdn.net/u014256231/article/details/79696391)
    #数据指针，包含图片索引
    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))

    #get_training_roidb函数其实就是将所有的bbox水平翻转一次，然后返回训练需要用的roidb
    #这是一个列表，列表中存的是各个图片的字典，字典中存roi信息，字典引索为图片引索
    #get_training_roidb函数在/lib/fast_rcnn/train.py中
    #[train.py](https://blog.csdn.net/u014256231/article/details/79696680)
    ##roidb = get_training_roidb(imdb)

    #输出路径
    if cfg.FLAGS.mask == 1:
        output_dir = get_output_dir(imdb, "mask")
    else:
        output_dir = get_output_dir(imdb, "faster")

    print('Output will be saved to `{:s}`'.format(output_dir))

    #/(args.device)(args.device_id)
    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print(device_name)

    #get_network在函数/lib/networks/factory.py中，
    #[factory.py](https://blog.csdn.net/u014256231/article/details/79696984)
    network,model_name = get_model(args.network_name)
    print('Use network `{:s}` in training'.format(args.network_name))

    #train_net在函数/lib/fast_rcnn/train.py中，
    #[train.py](https://blog.csdn.net/u014256231/article/details/79696680)
    #if model_name == 'mask':
    if cfg.FLAGS.mask == 1:
        roidb = my_batch("mask")
        #train_mask_model(network, imdb, roidb, output_dir, pretrained_model=args.pretrained_model,max_iters=args.max_iters)
        train_mask_model(network,roidb, output_dir, pretrained_model=args.pretrained_model,max_iters=args.max_iters)
    else:
        roidb = my_batch("faster")
        #train_faster_model(network, imdb, roidb, output_dir,pretrained_model=args.pretrained_model,max_iters=args.max_iters)
        train_faster_model(network, roidb, output_dir, pretrained_model=args.pretrained_model,max_iters=args.max_iters)