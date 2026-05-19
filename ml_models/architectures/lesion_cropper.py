import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.nn as nn

def get_detection_model(num_classes: int = 2) -> nn.Module:
    """
    Initializes the Faster R-CNN model with a custom anchor generator
    optimized for mammogram lesions (masses and calcifications).
    Class 0: Background, Class 1: Lesion.
    """
    # Restrict max anchor sizes so RPN doesn't generate massive boxes
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    # We do not need pretrained weights from ImageNet here because
    # the ModelLoader will inject our custom medical .pth weights.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        rpn_anchor_generator=anchor_generator
    )
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the default head with our specific number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
