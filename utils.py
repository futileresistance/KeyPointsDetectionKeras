import coremltools
import matplotlib.pyplot as plt
import cv2
import numpy as np
from config import imWidth

def visualize_some_imgs(data):
    img, hmaps = data[3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    for i in range(0, 1):
        axes[i].imshow(img[0][i])
        aug_hmap = cv2.resize(hmaps[0][i, :, :, 0], (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        axes[i].imshow(aug_hmap, alpha=.5)
        axes[i + 1].imshow(img[0][i + 1])
        aug_hmap1 = cv2.resize(hmaps[0][i + 1, :, :, 0], (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        axes[i + 1].imshow(aug_hmap1, alpha=.5)
    plt.show()


def make_pred(model, batch_number, img_idx, loader=False, test_im='example_imgs/test_img3.jpg'):
    global data_gen_val
    if loader:
        from dataloader import data_gen_val
    heatmap_names = ['Left eye', 'Right eye', 'Left wrist', 'Right wrist']
    if test_im:
        img_ = plt.imread(test_im)
        normalize = lambda x: x / 255 - 0.5
        img_ = normalize(img_)
        x_w = img_.shape[1]
        x_h = img_.shape[0]
        x_scale = imWidth / x_w
        y_scale = imWidth / x_h
        resized_img = cv2.resize(img_, (0, 0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
        test_img = np.expand_dims(resized_img, axis=0)
        pred = model.predict(test_img)
        pred_heatmaps = [pred[0][:, :, i] for i in range(4)]
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))
        for i in range(len(pred_heatmaps)):
            resized_pred = cv2.resize(pred_heatmaps[i], (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            axes[i].imshow(resized_img)
            axes[i].imshow(resized_pred, alpha=.5)
            axes[i].set_title(heatmap_names[i])
        plt.show()
    else:
        x, y = data_gen_val[batch_number]
        img_ = x[0][img_idx]
        test_img = np.expand_dims(img_, axis=0)
    # Predict
        pred = model.predict(test_img)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        true_heatmaps = [y[1][img_idx, :, :, i] for i in range(4)]
        pred_heatmaps = [pred[0][:, :, i] for i in range(4)]
        heatmap_names = ['Left eye', 'Right eye', 'Left wrist', 'Right wrist']
        for i in range(len(true_heatmaps)):
            resized_pred = cv2.resize(pred_heatmaps[i], (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            resized_true = cv2.resize(true_heatmaps[i], (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            axes[0][i].imshow(img_)
            axes[0][i].imshow(resized_true, alpha=.5)
            axes[0][i].set_title(heatmap_names[i] + '(Ground Truth)')
            axes[1][i].imshow(img_)
            axes[1][i].imshow(resized_pred, alpha=.5)
            axes[1][i].set_title(heatmap_names[i] + '(Prediction)')
    plt.show()


def save_to_coreml(model):
    ios_model_mobilenet = coremltools.converters.keras.convert(model, \
                                                               input_names='image', image_input_names="image",
                                                               output_names='heatmaps', image_scale=1 / 255.0)
    ios_model_mobilenet.save('LIGHT_thin_model_IOS.mlmodel')

