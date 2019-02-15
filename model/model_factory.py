__sets = {}

from .Faster_train_model import VOC_train_model
from .Faster_test_model import Faster_test_model
from .Mask_train_model import MASK_train_model
#import networks.VGGnet_test
import pdb
import tensorflow as tf
def get_model(name):

    #根据给定的network_name来拆分，根据test/train位置取net的性质信息
    if name.split('_')[1] == 'test':
        return Faster_test_model(),name.split('_')[1]
        pass
    elif name.split('_')[1] == 'train':
       #此时为训练
       return VOC_train_model(),name.split('_')[1]
    elif name.split('_')[1] == 'mask':
       #此时为训练
       return MASK_train_model(),name.split('_')[1]
    else:
       raise KeyError('Unknown dataset: {}'.format(name))


def list_networks():
    """List all registered imdbs."""
    return __sets.keys()

