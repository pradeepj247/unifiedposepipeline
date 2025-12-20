import abc
import os
from typing import Optional
import typing

import cv2
import numpy as np
import torch

from .configs.ViTPose_common import data_cfg
from .vit_models.model import ViTPose
from .vit_utils.inference import pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import, infer_dataset_by_path
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict

try:
    import torch_tensorrt
except ModuleNotFoundError:
    pass

try:
    import onnxruntime
except ModuleNotFoundError:
    pass

__all__ = ['VitPoseOnly']
np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class VitPoseOnly:
    """
    Simplified class for performing pose estimation using ViTPose models ONLY.
    This version does NOT include YOLO detection - use pre-computed bounding boxes.

    Args:
        model (str): Path to the ViT model file (.pth, .onnx, .engine).
        model_name (str, optional): Name of the ViT model architecture to use.
                                    Valid values are 's', 'b', 'l', 'h'.
                                    Defaults to None, is necessary when using .pth checkpoints.
        dataset (str, optional): Name of the dataset. If None it's extracted from the file name.
                                 Valid values are 'coco', 'coco_25', 'wholebody', 'mpii',
                                                  'ap10k', 'apt36k', 'aic'
        device (str, optional): Device to use for inference. Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(self, model: str,
                 model_name: Optional[str] = None,
                 dataset: Optional[str] = None,
                 device: Optional[str] = None):
        assert os.path.isfile(model), f'The model file {model} does not exist'

        # Device priority is cuda / mps / cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        # Extract dataset name
        if dataset is None:
            dataset = infer_dataset_by_path(model)

        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k', 'custom'], \
            'The specified dataset is not valid'

        # Dataset can now be set for visualization
        self.dataset = dataset

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # Use extension to decide which kind of model has been loaded
        use_onnx = model.endswith('.onnx')
        use_trt = model.endswith('.engine')

        # onnx / trt models do not require model_cfg specification
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify the model_name if not using onnx / trt'
        else:
            # Dynamically import the model class
            model_cfg = dyn_model_import(self.dataset, model_name)

        self.target_size = data_cfg['image_size']
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(model,
                                                             providers=['CUDAExecutionProvider',
                                                                        'CPUExecutionProvider'])
            inf_fn = self._inference_onnx
        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            if use_trt:
                self._vit_pose = torch.jit.load(model)
            else:
                ckpt = torch.load(model, map_location='cpu', weights_only=True)
                if 'state_dict' in ckpt:
                    self._vit_pose.load_state_dict(ckpt['state_dict'])
                else:
                    self._vit_pose.load_state_dict(ckpt)
                self._vit_pose.to(torch.device(device))

            inf_fn = self._inference_torch

        # Override _inference abstract with selected engine
        self._inference = inf_fn  # type: ignore

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        """
        Postprocess the heatmaps to obtain keypoints and their probabilities.

        Args:
            heatmaps (ndarray): Heatmap predictions from the model.
            org_w (int): Original width of the image.
            org_h (int): Original height of the image.

        Returns:
            ndarray: Processed keypoints with probabilities.
        """
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array([[org_w // 2,
                                                                 org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    @abc.abstractmethod
    def _inference(self, img: np.ndarray) -> np.ndarray:
        """
        Abstract method for performing inference on an image.
        It is overloaded by each inference engine.

        Args:
            img (ndarray): Input image for inference.

        Returns:
            ndarray: Inference results.
        """
        raise NotImplementedError

    def inference_bbox(self, img: np.ndarray, bbox: list) -> np.ndarray:
        """
        Perform pose estimation on a specific bounding box region.

        Args:
            img (ndarray): Input image for inference in RGB format.
            bbox (list): Bounding box in format [x1, y1, x2, y2]

        Returns:
            ndarray: Keypoints for the person in the bounding box.
        """
        x1, y1, x2, y2 = bbox
        pad_bbox = 10

        # Slightly bigger bbox
        bbox_expanded = [
            max(0, x1 - pad_bbox),
            max(0, y1 - pad_bbox),
            min(img.shape[1], x2 + pad_bbox),
            min(img.shape[0], y2 + pad_bbox)
        ]

        # Crop image and pad to 3/4 aspect ratio
        img_crop = img[bbox_expanded[1]:bbox_expanded[3], bbox_expanded[0]:bbox_expanded[2]]
        if img_crop.size == 0:
            return np.array([])
            
        img_inf, (left_pad, top_pad) = pad_image(img_crop, 3 / 4)

        keypoints = self._inference(img_inf)[0]
        # Transform keypoints to original image coordinates
        keypoints[:, 0] += bbox_expanded[1] - top_pad  # y coordinate
        keypoints[:, 1] += bbox_expanded[0] - left_pad  # x coordinate

        return keypoints

    def pre_img(self, img):
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)
        img_input = torch.from_numpy(img_input).to(torch.device(self.device))

        # Feed to model
        heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)

        # Feed to model
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, org_w, org_h)
