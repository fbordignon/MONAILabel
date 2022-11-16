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
from lib.transforms import FromHoverNetPatchesd, LoadImagePatchd, PostFilterLabeld, ToHoverNetPatchesd
from monai.apps.pathology.transforms import (
    GenerateDistanceMapd,
    GenerateInstanceBorderd,
    GenerateInstanceType,
    GenerateWatershedMarkersd,
    GenerateWatershedMaskd,
    Watershedd,
)
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    BoundingRect,
    CastToTyped,
    EnsureTyped,
    FillHoles,
    GaussianSmooth,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
    Transform,
)
from monai.utils import HoVerNetBranch
from tqdm import tqdm

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

        loglevel = data.get("logging", "INFO")
        super().set_loglevel("WARNING")
        for i in tqdm(range(math.ceil(inputs.shape[0] / max_batch_size))):
            x1 = i * max_batch_size
            x2 = min(inputs.shape[0], x1 + max_batch_size)
            batched_in = inputs[x1:x2]
            logger.debug(f"Running Infer for sub-batch: {x1} to {x2} of {inputs.shape[0]}")

            data[self.input_key] = batched_in
            data = super().run_inferer(data, False, device)

            np.append(data[HoVerNetBranch.NP])
            nc.append(data[HoVerNetBranch.NC])
            hv.append(data[HoVerNetBranch.HV])
        super().set_loglevel(loglevel.upper())

        data[HoVerNetBranch.NP] = torch.cat(np)
        data[HoVerNetBranch.NC] = torch.cat(nc)
        data[HoVerNetBranch.HV] = torch.cat(hv)
        return data

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            FromHoverNetPatchesd(keys=(HoVerNetBranch.NP, HoVerNetBranch.NC, HoVerNetBranch.HV)),
            Activationsd(keys=HoVerNetBranch.NC, softmax=True),
            AsDiscreted(keys=HoVerNetBranch.NC, argmax=True),
            GenerateWatershedMaskd(keys=HoVerNetBranch.NP, softmax=True),
            GenerateInstanceBorderd(keys="mask", hover_map_key=HoVerNetBranch.HV, kernel_size=21),
            GenerateDistanceMapd(keys="mask", border_key="border", smooth_fn=GaussianSmooth()),
            GenerateWatershedMarkersd(
                keys="mask", border_key="border", threshold=0.99, radius=3, postprocess_fn=FillHoles(connectivity=2)
            ),
            Watershedd(keys="dist", mask_key="mask", markers_key="markers"),
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
        self.labels = {v: k for k, v in labels.items()}

    def __call__(self, data):
        d = dict(data)

        pred_inst = d["dist"]
        type_pred = d[HoVerNetBranch.NC]
        elements = []

        result = np.zeros_like(pred_inst)
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            inst_map = inst_map.astype(np.uint8)
            inst_bbox = BoundingRect()(inst_map)

            inst_type, type_prob = GenerateInstanceType()(
                bbox=inst_bbox,
                type_pred=type_pred,
                seg_pred=pred_inst,
                instance_id=inst_id,
            )
            result = np.where(pred_inst == inst_id, inst_type, result)

        logger.info(f"Total Instances: {len(elements)}; Labels Found: {np.unique(result)}")

        d["pred"] = result
        return d
