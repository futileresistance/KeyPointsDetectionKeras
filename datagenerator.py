from keras.utils import Sequence
from albumentations import Rotate
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

class DataGenerator(Sequence):
    def __init__(self, target_size, coco_data, batch_size=4, shuffle=True, aug=False, mode_thin=False):
        # Init attrs
        self.coco = coco_data
        self.target_size = target_size
        self.img_path = self.coco.img_path
        self.images = self.coco.imgs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode_thin = mode_thin
        self.aug = aug
        self.on_epoch_end()


    def get_num_of_imgs(self):
        print(len(self.images))


    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.images) / self.batch_size))


    def __data_generation(self, list_img_IDs):
        # Func to deal with non-colored pics
        def to_rgb(img):
            w, h = img.shape
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, 0] = img / 255
            ret[:, :, 1] = img / 255
            ret[:, :, 2] = img / 255
            return ret

        # augmentations
        augmentation_1 = Rotate(limit=180, p=0.5)
        # Define heatmap size
        self.heatmap_size = int(self.target_size[0] / 8)
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size, self.heatmap_size, self.heatmap_size, self.num_of_features))
        # Normalization func
        normalize = lambda x: x / 255 - 0.5
        # Generate data
        for i, ID in enumerate(list_img_IDs):
            # Store ann & image
            ann_id = self.coco.coco.getAnnIds(imgIds=ID)
            ann = self.coco.coco.loadAnns(ann_id)
            img = self.coco.coco.loadImgs(ID)[0]
            I = plt.imread(self.img_path + img['file_name'])
            # Convert to rgb if 1-channel pic
            if len(I.shape) == 2:
                normalized = to_rgb(I)
            else:
                normalized = normalize(I)
            resized_image = self.prepare(normalized)
            # Store heatmaps
            anns_list = self.get_xydots(ann)
            heatmaps = self.create_heatmap(anns_list, I)
            # Eyes & wrists # 1,2,9,10
            spec_hmaps = heatmaps[:, :, [1, 2, 9, 10]]
            if self.aug:
                data = {"image": spec_hmaps, "mask": resized_image}
                # flips
                data_augmented = augmentation_1(**data)
                image_augmented = data_augmented["mask"]
                hmap_augmented = data_augmented["image"]
                X[i,] = image_augmented
                y[i] = hmap_augmented
            else:
                X[i,] = resized_image
                y[i] = spec_hmaps
        return X, y


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Number of features
        self.num_of_features = 4
        # Find list of IDs
        list_img_IDs = [self.images[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_img_IDs)
        # Mask
        ones_mask = np.ones((self.batch_size, self.heatmap_size, self.heatmap_size, self.num_of_features))
        if self.mode_thin:
            return [X, ones_mask], [y, y, y, y]
        return [X, ones_mask], [y, y, y, y, y, y]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def prepare(self, image):
        size = self.target_size[0]
        x_w = image.shape[1]
        x_h = image.shape[0]
        x_scale = size / x_w
        y_scale = size / x_h
        resized_img = cv2.resize(image, (0, 0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
        return resized_img


    def get_xydots(self, anns):
        annots = []
        for ann in anns:
            ann = ann['keypoints']
            xs = ann[0::3]
            ys = ann[1::3]
            vs = ann[2::3]
            # filter and loads keypoints to the list
            keypoints_list = []
            for idx, (x, y, v) in enumerate(zip(xs, ys, vs)):
                # only visible and occluded keypoints are used
                if v >= 1 and x >= 0 and y >= 0:
                    keypoints_list.append((x, y))
                else:
                    keypoints_list.append(None)
            annots.append(keypoints_list)
        return annots


    def _draw_heatmap(self, image, heatmap, plane_idx, joint, sigma, height, width, stride):
        start = stride / 2.0 - 0.5
        size = int(self.target_size[0] / 8)
        x_w = image.shape[1]
        x_h = image.shape[0]
        x_scale = size / x_w
        y_scale = size / x_h
        center_x, center_y = joint
        center_x *= x_scale
        center_y *= y_scale
        for g_y in range(height):
            for g_x in range(width):
                x = start + g_x * stride
                y = start + g_y * stride
                d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
                exponent = d2 / 2.0 / sigma / sigma
                if exponent > 4.6052:
                    continue
                heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
                if heatmap[g_y, g_x, plane_idx] > 1.0:
                    heatmap[g_y, g_x, plane_idx] = 1.0


    def create_heatmap(self, keypoints, image):
        height = width = int(self.target_size[0] / 8)
        num_of_maps = 17
        heatmap = np.zeros((height, width, num_of_maps), dtype=np.float64)
        sigma = 1
        stride = 1
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                if kp:
                    self._draw_heatmap(image, heatmap, idx, kp, sigma, height, width, stride)
        return heatmap


    def size(self):
        return len(self.images)
