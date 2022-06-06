import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

# !wget http://images.cocodataset.org/val2017/000000005477.jpg -q -O input.jpg
# We first download an image from the COCO dataset:
im = cv2.imread("./input.jpg")
plt.imshow(im)
plt.show()

# Then, we create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image. First, we make a prediction using a standard Mask R-CNN model.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
mask_rcnn_predictor = DefaultPredictor(cfg)
mask_rcnn_outputs = mask_rcnn_predictor(im)

# Now, we load a PointRend model and show its prediction.
cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Show and compare two predictions:
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
mask_rcnn_result = v.draw_instance_predictions(mask_rcnn_outputs["instances"].to("cpu")).get_image()
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
print("Mask R-CNN with PointRend (top)     vs.     Default Mask R-CNN (bottom)")
plt.imshow(np.concatenate((point_rend_result, mask_rcnn_result), axis=0)[:, :, ::-1])
plt.tight_layout()
plt.savefig('./output.jpg', bbox_inches='tight')
# plt.show()
