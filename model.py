from layers import MobileNevV2_block, separable_conv_block, stage_T_block, apply_mask
from keras.layers.merge import Concatenate
from keras.layers import Input
from keras.models import Model
from config import stages, np_branch1, heat_input_shape, img_input_shape

heat_weight_input = Input(shape=heat_input_shape)
img_weight_input = Input(shape=img_input_shape)

inputs = [img_weight_input, heat_weight_input]
outputs = []

#model_vanilla_LIGHT = load_weights('./mobilenetv2_weights/mobilenetv2_0.5aplha.h5')
#Model construction
stage0_out = MobileNevV2_block(img_weight_input)
# stage 1 - branch 1 (confidence maps)
stage1_branch1_out = separable_conv_block(stage0_out, np_branch1)
w1 = apply_mask(stage1_branch1_out, heat_weight_input, np_branch1, 1)
x = Concatenate()([stage1_branch1_out, stage0_out])
outputs.append(w1)
#stage sn >= 2
for sn in range(2, stages + 1):
    # stage SN - branch 1 (confidence maps)
    stageT_branch1_out = stage_T_block(x, np_branch1, sn)
    w1 = apply_mask(stageT_branch1_out, heat_weight_input, np_branch1, sn)
    outputs.append(w1)
    if (sn < stages):
        x = Concatenate()([stageT_branch1_out, stage0_out])


thin_model = Model(inputs, outputs)
#print(thin_model.summary())
thin_model.load_weights('./mobilenetv2_weights/mobilenetv2_0.5aplha.h5', by_name=True)
#restore_weights_from_MNet(thin_model, model_vanilla_LIGHT)