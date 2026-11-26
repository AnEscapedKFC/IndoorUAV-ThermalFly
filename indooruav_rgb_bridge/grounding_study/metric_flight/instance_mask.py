"""RGB instance segmentation for the grounded object box using frozen SlimSAM."""

from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "history_sam"


class InstanceMask:
    def __init__(self):
        device = os.environ.get("INDOORUAV_MASK_DEVICE", "cpu")
        self.device = torch.device(device)
        self.processor = SamProcessor.from_pretrained(WEIGHTS, local_files_only=True)
        self.model = (
            SamModel.from_pretrained(
                WEIGHTS, local_files_only=True, attn_implementation="sdpa"
            )
            .eval()
            .to(self.device)
        )

    @torch.inference_mode()
    def __call__(self, rgb, box):
        h, w = rgb.shape[:2]
        xyxy = (np.array(box) * [w, h, w, h]).tolist()
        inputs = self.processor(
            images=Image.fromarray(rgb), input_boxes=[[xyxy]], return_tensors="pt"
        )
        masks = self.model(
            pixel_values=inputs.pixel_values.to(self.device),
            input_boxes=inputs.input_boxes.to(self.device),
            multimask_output=False,
        ).pred_masks.cpu()
        mask = self.processor.post_process_masks(
            masks, inputs.original_sizes, inputs.reshaped_input_sizes
        )[0]
        return mask[0, 0].numpy()


@lru_cache(maxsize=1)
def instance_mask():
    return InstanceMask()
