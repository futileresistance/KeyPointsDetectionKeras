from datagenerator import DataGenerator
from cocotools import COCODataPaths
from utils import visualize_some_imgs
from config import train_imgs_dir, train_anns_dir, val_imgs_dir, val_anns_dir, imWidth, imHeight, batch_size


train = COCODataPaths(train_anns_dir, train_imgs_dir)
val = COCODataPaths(val_anns_dir, val_imgs_dir)

data_gen_train = DataGenerator(target_size=(imWidth,imHeight), coco_data=train, batch_size=batch_size,\
                               shuffle=False, aug=True, mode_thin=True)
data_gen_val = DataGenerator(target_size=(imWidth,imHeight), coco_data=val, batch_size=batch_size, shuffle=True,\
                            aug=False, mode_thin=True)

#visualize_some_imgs(data_gen_train)