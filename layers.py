from keras.layers import SeparableConv2D
from keras.layers.merge import Multiply
from keras.initializers import random_normal
import keras.backend as K
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, Add, DepthwiseConv2D
from config import kernel_init, kernel_reg


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = -1
    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand

        # use bias
        x = Conv2D(expansion * in_channels,
                   kernel_size=1,
                   padding='same',
                   use_bias=True,
                   activation=None,
                   name=prefix + 'expand_Q')(x)
        x = BatchNormalization(axis=channel_axis,
                               epsilon=1e-3,
                               momentum=0.999,
                               name=prefix + 'expand_BN_Q')(x)
        x = ReLU(6., name=prefix + 'expand_relu_Q')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        correct_pad = ((0, 1), (0, 1))
        x = ZeroPadding2D(padding=correct_pad,
                          name=prefix + 'pad_Q')(x)
    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        activation=None,
                        use_bias=True,
                        padding='same' if stride == 1 else 'valid',
                        name=prefix + 'depthwise_Q')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'depthwise_BN_Q')(x)

    x = ReLU(6., name=prefix + 'depthwise_relu_Q')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=True,
               activation=None,
               name=prefix + 'project_Q')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'project_BN_Q')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add_Q')([inputs, x])
    return x


def MobileNevV2_block(img_input, alpha=0.5):
    correct_pad = ((0, 1), (0, 1))
    x = ZeroPadding2D(padding=correct_pad,
                      name='Conv1_pad_Q')(img_input)
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters, kernel_size=3,
               strides=(2, 2),
               padding='valid',
               use_bias=True,
               name='Conv1_Q')(x)
    x = BatchNormalization(axis=-1,
                           epsilon=1e-3,
                           momentum=0.999,
                           name='bn_Conv1_Q')(x)
    x = ReLU(6., name='Conv1_relu_Q')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    # new_blocks added
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=6)
    #     x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
    #                             expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    #     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
    #                             expansion=6, block_id=11)
    return x


def restore_weights_from_MNet(model, vanilla_model):
    layers_names = set([layer.name for layer in vanilla_model.layers])
    for layer in model.layers:
        layer_name = layer.name[:-2]
        if layer_name in layers_names:
            if layer.weights:
                K.set_value(layer.weights[0], vanilla_model.get_layer(layer_name).get_weights()[0])
            print("Loaded MobileNetV2 layer: " + layer_name)
    print("DONE!")


def separable_conv_block(inp, num_features):
    prefix = 'MConv_stage1_'
    x = SeparableConv2D(filters=128, activation='relu', kernel_size=(3,3), kernel_regularizer=kernel_reg,\
                        kernel_initializer=kernel_init, padding='same', name=prefix + 'L_1')(inp)
    x = SeparableConv2D(filters=128, activation='relu', kernel_size=(3,3),kernel_regularizer=kernel_reg,\
                        kernel_initializer=kernel_init, padding='same', name=prefix + 'L_2')(x)
    x = SeparableConv2D(filters=128, activation='relu', kernel_size=(3,3), kernel_regularizer=kernel_reg,\
                        kernel_initializer=kernel_init, padding='same', name=prefix + 'L_3')(x)
    x = Conv2D(filters=512, activation='relu', kernel_size=(1,1),kernel_regularizer=kernel_reg,\
               kernel_initializer=kernel_init, padding='same', name=prefix + 'L_4')(x)
    x = SeparableConv2D(filters=num_features, activation=None, kernel_size=(1,1),\
                        kernel_regularizer=kernel_reg, kernel_initializer=kernel_init, padding='same', \
                        name=prefix + 'L_5')(x)
    return x


def stage_T_block(inp, num_features, stage_num):
    kern_size = (3,3)
    prefix = 'MConv_stage{}_'.format(stage_num)
    x = SeparableConv2D(filters=128, activation='relu', kernel_size=kern_size, kernel_regularizer=kernel_reg,\
                        kernel_initializer=kernel_init,padding='same', name=prefix + 'L_1')(inp)
    x = SeparableConv2D(filters=128, activation='relu', kernel_size=kern_size,kernel_regularizer=kernel_reg,\
                        kernel_initializer=kernel_init, padding='same', name=prefix + 'L_2')(x)
    x = SeparableConv2D(filters=128, activation='relu', kernel_size=kern_size,kernel_regularizer=kernel_reg, \
                        kernel_initializer=kernel_init, padding='same', name=prefix + 'L_3')(x)
#     x = SeparableConv2D(filters=128, activation='relu', kernel_size=kern_size,kernel_regularizer=kernel_reg, \
#                         kernel_initializer=kernel_init, padding='same', name=prefix + 'L_4')(x)
#     x = SeparableConv2D(filters=128, activation='relu', kernel_size=kern_size,kernel_regularizer=kernel_reg, \
#                         kernel_initializer=kernel_init, padding='same', name=prefix + 'L_5')(x)
    x = Conv2D(filters=128, activation='relu', kernel_size=(1,1), kernel_regularizer=kernel_reg,\
               kernel_initializer=kernel_init,padding='same', name=prefix + 'L_4')(x)
    x = SeparableConv2D(filters=num_features, activation=None, kernel_size=(1,1), \
                        kernel_regularizer=kernel_reg, kernel_initializer=kernel_init,padding='same',\
                        name=prefix + 'L_5')(x)
    return x


def apply_mask(x, mask1, num_p, stage):
    w_name = "weight_stage%d" % (stage)
    w = Multiply(name=w_name)([x, mask1])  # vec_heat
    return w


def get_loss_funcs(batch_size=8):
    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["weight_stage1"] = _eucl_loss
    losses["weight_stage2"] = _eucl_loss
    losses["weight_stage3"] = _eucl_loss
    losses["weight_stage4"] = _eucl_loss
    #losses["weight_stage5"] = _eucl_loss
    #losses["weight_stage6"] = _eucl_loss
    return losses


def show_gpus():
    gpus = K.tensorflow_backend._get_available_gpus()
    print(f"GPUs available: {gpus}")