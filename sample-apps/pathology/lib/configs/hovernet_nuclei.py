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

import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import HoVerNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class HovernetNuclei(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.labels = {
            "Other": 1,
            "Inflammatory": 2,
            "Epithelial": 3,
            "Spindle-Shaped": 4,
        }
        self.label_colors = {
            "Other": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Epithelial": (0, 0, 255),
            "Spindle-Shaped": (0, 255, 0),
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/pathology_segmentation_hovernet_nuclei.pt"
            download_file(url, self.path[0])

        # Network
        self.network = HoVerNet(
            mode="original",
            in_channels=3,
            out_classes=len(self.labels) + 1,
            act=("relu", {"inplace": True}),
            norm="batch",
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        preload = strtobool(self.conf.get("preload", "false"))
        roi_size = json.loads(self.conf.get("roi_size", "[1024, 1024]"))
        logger.info(f"Using Preload: {preload}; ROI Size: {roi_size}")

        task: InferTask = lib.infers.HovernetNuclei(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=preload,
            roi_size=roi_size,
            config={
                "label_colors": self.label_colors,
                "max_workers": max(1, multiprocessing.cpu_count() // 2),
            },
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        return None
