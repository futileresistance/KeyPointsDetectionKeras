# Real-time human anatomical points localizer
Based on [this](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation) repo and [this](https://arxiv.org/abs/1611.08050) paper.
## Network architecture
<p align="center">
<img src="https://github.com/futileresistance/KeyPointsDetectionKeras/blob/master/readme/pose_estim_arch.png", width="720">
</p>

## Results
<p align="center">
<img src="https://github.com/futileresistance/KeyPointsDetectionKeras/blob/master/readme/result.png", width="720">
</p>

## Requirements
- keras
- pycocotools
- opencv-python
- albumentations
- numpy
- matplotlib

## Installation
```bash
git clone repo
cd repo
pip install -r requirements.txt
```
## Training
Download [COCO](http://cocodataset.org/#download) dataset. Put train and validation images&annotations to the folders in __/data__ respectively. Run:
```python
python train.py
```
## Testing
Put test image in the __example_imgs__ folder. Then run:
```python
python test.py your_test_image_name.jpg
```
## Extra
:eyes:
