from keras.initializers import random_normal

# data path configs
train_imgs_dir = '/data/train_img/train2017/'
val_imgs_dir = '/data/val_img/val2017/'
train_anns_dir = '/data/train_annot/annotations/person_keypoints_train2017.json'
val_anns_dir =  '/data/val_annot/annotations/person_keypoints_val2017.json'

# model configs
stages = 4
np_branch1 = 4
kernel_reg = None
kernel_init=random_normal(stddev=0.01)
# image configs
imWidth, imHeight = 640, 640
heatmap_size = int(imWidth/8)
batch_size = 8
heat_input_shape = (heatmap_size, heatmap_size, np_branch1)
img_input_shape = (imWidth, imHeight, 3)
# training configs
base_lr = 4e-4
max_iter = 50
logs_dir = "./logs"
weights_best_file = "models/weights/thin_model_weights0810.best.h5"
training_log = "thin_model_weights.csv"

# pretrained models
thin_model_pretrained = 'models/light_mobilenet_model.h5'




