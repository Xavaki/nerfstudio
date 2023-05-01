"""Data parser for synthetically occluded custom dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import numpy as np
import random
import math

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio


@dataclass
class OcclusionsDataparserConfig(NerfstudioDataParserConfig):
    """Occluded dataset parser config"""

    _target: Type = field(default_factory=lambda: Occlusions)
    """target class to instantiate"""
    use_synthetic_occlusions: bool = True
    """Whether to use occluded data or not"""
    occlusion_type: str = "synthetic_occlusions"
    """Type of occlusions (corresponding dir must exist)"""
    occlusion_level: int = 1
    """Degree of occlusion of images (corresponding dir must exist)"""
    occlusion_prop: float = 0.1
    """Proportion of input train dataset to occlude"""
    shuffle_occlusions: bool = True
    """Whether to occlude same batch of images or not"""


@dataclass
class Occlusions(Nerfstudio):
    """Occlusions Dataset
    """

    config: OcclusionsDataparserConfig

    def __init__(self, config: OcclusionsDataparserConfig):
        super().__init__(config=config)
        self.occluded_imgs_idxs = []
        ...
    def _generate_dataparser_outputs(self, split="train"):
        dataparser_outputs = super()._generate_dataparser_outputs(split)
        if split != "train" or not self.config.use_synthetic_occlusions:
            return dataparser_outputs

        original_filenames = dataparser_outputs.image_filenames
        num_total_images = len(original_filenames)
        num_occluded_images = math.ceil(num_total_images * self.config.occlusion_prop)
        i_all = list(np.arange(num_total_images))
        if self.config.shuffle_occlusions:
            random.shuffle(i_all)
        i_occluded = i_all[:num_occluded_images]

        
        self.occluded_imgs_idxs = i_occluded
        for idx in i_occluded:
            image_filename = original_filenames[idx]
            occluded_image_filename = image_filename.parents[0] / self.config.occlusion_type / f"occlusions_{self.config.occlusion_level}" / image_filename.name
            assert occluded_image_filename.exists(), f"Occlusion info for level {self.config.occlusion_level} not in specified data dir"
            dataparser_outputs.image_filenames[idx] = occluded_image_filename

        return dataparser_outputs
