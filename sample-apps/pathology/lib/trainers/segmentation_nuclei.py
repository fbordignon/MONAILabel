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
import os

import numpy as np
import torch
from ignite.metrics import Accuracy
from lib.handlers import TensorBoardImageHandler
from lib.transforms import Agumentd
from lib.utils import split_dataset
from monai.handlers import from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangeD,
    ScaleIntensityRanged,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class SegmentationNuclei(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        roi_size=(512, 512),
        description="Pathology Semantic Segmentation for Nuclei (PanNuke Dataset)",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)

    def x_pre_process(self, request, datastore: Datastore):
        self.cleanup(request)

        cache_dir = os.path.join(self.get_cache_dir(request), "train_ds")
        source = request.get("dataset_source")
        max_region = request.get("dataset_max_region", (10240, 10240))
        max_region = (max_region, max_region) if isinstance(max_region, int) else max_region[:2]

        return split_dataset(
            datastore=datastore,
            cache_dir=cache_dir,
            source=source,
            groups=self._labels,
            tile_size=self.roi_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
        )

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            Agumentd(keys="image", prob=0.7),
            EnsureTyped(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            RandFlipd(keys=("image", "label"), prob=0.5),
            RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(-2, -1)),
            ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                image_key="image",
                num_samples=16,
                spatial_size=self.roi_size,
            ),
            # RandSpatialCropSamplesd(
            #     keys=("image", "label"),
            #     num_samples=32,
            #     roi_size=self.roi_size,
            #     random_size=False,
            # ),
        ]

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys="pred", device=context.device),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=len(self._labels) + 1),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            EnsureTyped(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]

    def train_additional_metrics(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_additional_metrics(self, context: Context):
        return {"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(1024, 1024))

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(
                TensorBoardImageHandler(log_dir=context.events_dir, interval=10, batch_limit=4, tag_name="train")
            )
        return handlers

    def val_handlers(self, context: Context):
        handlers = super().val_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir, interval=10, batch_limit=16))
        return handlers
