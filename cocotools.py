from pycocotools.coco import COCO

class COCODataPaths:
    def __init__(self, annot_path, img_path):
        self.coco = COCO(annot_path)
        catIds = self.coco.getCatIds(catNms=['person'])
        imgIds = self.coco.getImgIds(catIds=catIds)
        annIds = self.coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
        self.img_path = img_path
        self.ann_path = annot_path
        self.annot = annIds
        self.imgs = imgIds
