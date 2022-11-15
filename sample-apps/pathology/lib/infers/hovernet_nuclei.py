# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Any, Callable, Dict, Sequence

import numpy as np
import torch
from monai.inferers import Inferer, SlidingWindowInferer, SimpleInferer
from torchvision.transforms import Compose

from lib.transforms import LoadImagePatchd, PostFilterLabeld, ToHoverNetPatchesd, FromHoverNetPatchesd
from monai.apps.pathology.transforms import GenerateWatershedMaskd, GenerateInstanceBorderd, GenerateDistanceMapd, \
    GenerateWatershedMarkersd, Watershedd, GenerateInstanceContour, GenerateInstanceCentroid, GenerateInstanceType
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    BoundingRect,
    EnsureTyped,
    FillHoles,
    GaussianSmooth,
    SqueezeDimd,
    ToNumpyd, Transform, CastToTyped, Activationsd, AsDiscreted, CenterSpatialCropd, Padd, SpatialPadd,
    ScaleIntensityRanged, RemoveSmallObjectsd, SobelGradientsd, BorderPadd,
)
from monai.utils import HoVerNetBranch, convert_to_tensor
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class HovernetNuclei(BasicInferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            roi_size=(256, 256),
            type=InferType.SEGMENTATION,
            labels=None,
            dimension=2,
            description="A pre-trained hovernet model for segmentation + classification of Nuclei",
            **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        return d

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            AsChannelFirstd(keys="image"),
            ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            CastToTyped(keys="image", dtype=torch.float32),
            ToHoverNetPatchesd(image="image"),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        inputs = data[self.input_key]
        max_batch_size = data.get("max_batch_size", 16)
        np = []
        nc = []
        hv = []
        for i in range(math.ceil(inputs.shape[0] / max_batch_size)):
            x1 = i * max_batch_size
            x2 = min(inputs.shape[0], x1 + max_batch_size)
            batched_in = inputs[x1:x2]
            logger.info(f"Running Infer for sub-batch: {x1} to {x2} of {inputs.shape[0]}")

            data[self.input_key] = batched_in
            data = super().run_inferer(data, False, device)

            np.append(data[HoVerNetBranch.NP])
            nc.append(data[HoVerNetBranch.NC])
            hv.append(data[HoVerNetBranch.HV])

        data[HoVerNetBranch.NP] = torch.cat(np)
        data[HoVerNetBranch.NC] = torch.cat(nc)
        data[HoVerNetBranch.HV] = torch.cat(hv)
        return data

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            FromHoverNetPatchesd(keys=(HoVerNetBranch.NP, HoVerNetBranch.NC, HoVerNetBranch.HV)),
            Activationsd(keys=(HoVerNetBranch.NP, HoVerNetBranch.NC), softmax=True),
            AsDiscreted(keys=(HoVerNetBranch.NP, HoVerNetBranch.NC), argmax=True),
            # SobelGradientsd(keys=HoVerNetBranch.NP, kernel_size=21),
            RemoveSmallObjectsd(keys=HoVerNetBranch.NP),

            #GenerateInstanceBorderd(keys="mask", hover_map_key=HoVerNetBranch.HV, kernel_size=21),
            # GenerateDistanceMapd(keys="mask", border_key="border", smooth_fn=GaussianSmooth()),
            # GenerateWatershedMarkersd(
            #     keys="mask", border_key="border", threshold=0.99, radius=3, postprocess_fn=FillHoles(connectivity=2)
            # ),
            # Watershedd(keys="dist", mask_key="mask", markers_key="markers"),
            PostProcessWS(labels=self.labels),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred", dtype=np.uint8),
            PostFilterLabeld(keys="pred"),
            FindContoursd(keys="pred", labels=self.labels, max_poly_area=128 * 128),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)


class PostProcessWS(Transform):
    def __init__(self, labels=None) -> None:
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        device = d.get("device")

        return_binary = True
        return_centroids = False
        output_classes = None

        type_pred = None
        pred_inst = d[HoVerNetBranch.NP]

        inst_info_dict = {}
        if return_centroids:
            inst_id_list = np.unique(pred_inst)[1:]  # exclude background
            for inst_id in inst_id_list:
                inst_map = pred_inst == inst_id
                inst_bbox = BoundingRect()(inst_map)
                inst_map = inst_map[:, inst_bbox[0][0]: inst_bbox[0][1], inst_bbox[0][2]: inst_bbox[0][3]]
                offset = [inst_bbox[0][2], inst_bbox[0][0]]
                try:
                    inst_contour = GenerateInstanceContour()(inst_map, offset)
                except:
                    inst_contour = GenerateInstanceContour()(FillHoles(connectivity=2)(inst_map), offset)

                inst_centroid = GenerateInstanceCentroid()(inst_map, offset)
                if inst_contour is not None:
                    inst_info_dict[inst_id] = {  # inst_id should start at 1
                        "bounding_box": inst_bbox,
                        "centroid": inst_centroid,
                        "contour": inst_contour,
                        "type_probability": None,
                        "type": None,
                    }

        if output_classes is not None:
            for inst_id in list(inst_info_dict.keys()):
                inst_type, type_prob = GenerateInstanceType()(
                    bbox=inst_info_dict[inst_id]["bounding_box"],
                    type_pred=type_pred,
                    seg_pred=pred_inst,
                    instance_id=inst_id,
                )
                inst_info_dict[inst_id]["type"] = inst_type
                inst_info_dict[inst_id]["type_probability"] = type_prob

        logger.info(f"Pred Values: {np.unique(pred_inst, return_counts=True)}")

        # pred_inst = convert_to_tensor(pred_inst, device=device)
        # pred_type_map = torch.zeros_like(pred_inst)
        # for key, value in inst_info_dict.items():
        #     pred_type_map[pred_inst == key] = value["type"]
        # pred_type_map = AsDiscrete(to_onehot=len(self.labels) + 1)(pred_type_map)
        #
        # if return_binary:
        #     pred_inst[pred_inst > 0] = 1

        d["pred"] = pred_inst
        d["pred_class"] = None
        d["pred_instances"] = inst_info_dict
        return d
