import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers.merge import Concatenate
from keras.layers import Input
from model import img_input_shape
from config import stages, np_branch1, thin_model_pretrained
from keras.models import Model
from layers import MobileNevV2_block, separable_conv_block, stage_T_block
from config import weights_best_file
from utils import make_pred
import argparse

parser = argparse.ArgumentParser(description='test model on image')
parser.add_argument('--image', type=str, default='test_img2.jpg')
args = parser.parse_args([])
print('begin construction')
test_inputs = Input(shape=img_input_shape)
stage0_out = MobileNevV2_block(test_inputs)
stage1_branch1_out = separable_conv_block(stage0_out, np_branch1)
x = Concatenate()([stage1_branch1_out, stage0_out])
stageT_branch1_out = None
for sn in range(2, stages + 1):
    stageT_branch1_out = stage_T_block(x, np_branch1, sn)
    if (sn < stages):
        x = Concatenate()([stageT_branch1_out, stage0_out])
test_outputs = [stageT_branch1_out]
thin_test_model = Model(test_inputs, test_outputs)
print('construction has finished')
thin_test_model.load_weights(weights_best_file)
#print('*****************************************************')
path_to_img = 'example_imgs/' + args.image
make_pred(thin_test_model, 1, 1, test_im=path_to_img)