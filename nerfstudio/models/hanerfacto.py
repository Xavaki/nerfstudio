# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
+ Ha-NeRF occlusion handling by xavaki
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import debugpy
import numpy as np
import torch
from PIL import Image

# xx Ha-NeRF occlusion mask dependencies
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.hanerfacto_field import TCNNHaNerfactoField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.model_components.losses import hanerf_occlusion_mask_loss


@dataclass
class HaNerfactoModelConfig(ModelConfig):
    """HaNerfacto Model Config"""


    # nerfacto parameters
    _target: Type = field(default_factory=lambda: HaNerfacto)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""

    # hanerf parameters
    hanerf_loss_color_mult: float = 0.5
    """Hanerf occlusion mask loss color term multiplier"""
    hanerf_loss_mask_size_delta: float = 0.006
    """Hanerf occlusion mask loss mask size delta"""
    hanerf_loss_mask_digit_delta: float = 0.001
    "Hanerf occlusion mask loss mask digit delta"
    enable_hanerf_loss: bool = True
    "Whether to predict hanerf occlusion mask or not. For debug purposes"
    save_debug_predicted_images: bool = True
    "Whether to save predicted rgbs for a sample of train images in hanerf_debug folder"
    save_debug_hanerf_occlusion_mask: bool = True
    "Whether to save occlusion masks for a sample of train images in hanerf_debug folder"
    save_sample_count: bool = True
    "Whether to save image and pixel count info"
    hanerf_debug_frequency: int = 5000 # xx should depend on total number of iterations
    """How often to save debug information"""



class HaNerfacto(Model):
    """HaNerfacto model

    Args:
        config: HaNerfacto configuration to instantiate model
    """

    config: HaNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNHaNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # xx OCCLUSION MASK
        # xx as per hanerf implementation (see HaNerf paper section 5.1)
        self.debug_img_idxs = (0, 1, 2, 3, 4)
        # encoding of uv/pixel coordinates (not explained in paper but present in implementation)
        self.uv_position_encoding_num_freqs = 10
        self.uv_position_encoding = tcnn.Encoding(n_input_dims=2, encoding_config={"otype": "Frequency", "n_frequencies": self.uv_position_encoding_num_freqs})
        
        self.occlusion_embedding_dim = 128
        self.occlusion_embedding = Embedding(self.num_train_data, self.occlusion_embedding_dim)
        # tinycudann only supports up to 128 neurons, so we use torch implementation instead
        W = 256 
        n_input_dims = self.occlusion_embedding_dim + self.uv_position_encoding.n_output_dims
        self.occlusion_mask_mlp = nn.Sequential(
                                                nn.Linear(n_input_dims, W), nn.ReLU(True),
                                                nn.Linear(W, W), nn.ReLU(True),
                                                nn.Linear(W, W), nn.ReLU(True),
                                                nn.Linear(W, W), nn.ReLU(True),
                                                nn.Linear(W, 1), nn.Sigmoid())

        # xx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.enable_hanerf_loss:
            param_groups["losses"] = list(self.occlusion_mask_mlp.parameters()) + list(self.occlusion_embedding.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            
        @torch.no_grad()
        def setup_debug_hanerf_occlusion_mask_dir(step):
            out_dir = training_callback_attributes.base_dir
            debug_dir = out_dir / "hanerf_debug"
            debug_dir.mkdir()
            train_dataset_ref = training_callback_attributes.pipeline.datamanager.train_dataset
            occluded_imgs_idxs = training_callback_attributes.pipeline.datamanager.dataparser.occluded_imgs_idxs
            if occluded_imgs_idxs:
                self.debug_img_idxs = occluded_imgs_idxs
                
            for i in self.debug_img_idxs:
                imagei_path = train_dataset_ref.image_filenames[i]
                imagei_stem = imagei_path.stem
                imagei_name = imagei_path.name
                debug_imagei_dir = debug_dir / imagei_stem
                debug_imagei_dir.mkdir()

                shutil.copyfile(imagei_path, debug_imagei_dir / "train_image.jpg")
                if "occlusion" in str(imagei_path):
                    non_occluded_imagei_path = imagei_path.parents[2] / imagei_name
                    shutil.copyfile(non_occluded_imagei_path, debug_imagei_dir / "train_image_non_occluded.jpg")



        set_debug_dir = self.config.save_debug_predicted_images or self.config.save_debug_hanerf_occlusion_mask or self.config.save_sample_count
        if set_debug_dir:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    iters=(0,),
                    func=setup_debug_hanerf_occlusion_mask_dir,
                )
            )

        def setup_debug_sample_count(step):
            sample_count_dict = {}
            train_dataset_ref = training_callback_attributes.pipeline.datamanager.train_dataset
            for i in self.debug_img_idxs:
                image = train_dataset_ref[i]['image']
                height, width, _ = image.shape
                sample_count_dict[i] = {"info_dict" : { "img_sample_count" : 0, "prop_pixels_sampled" : 0 }, "pixel_count_img" : torch.zeros((height, width))}
            self.debug_sample_count_dict = sample_count_dict

        if self.config.save_sample_count:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    iters=(0,),
                    func=setup_debug_sample_count,
                )
            )

        
        @torch.no_grad()
        def save_sample_count(step):
            if step == 0: return
            out_dir = training_callback_attributes.base_dir
            debug_dir = out_dir / "hanerf_debug"
            train_dataset_ref = training_callback_attributes.pipeline.datamanager.train_dataset
            for i in self.debug_img_idxs:
                image_stem = train_dataset_ref.image_filenames[i].stem
                debug_image_dir = debug_dir / image_stem

                pixel_count_img = self.debug_sample_count_dict[i]["pixel_count_img"]
                pixel_count_img = pixel_count_img / pixel_count_img.max() 
                save_pixel_count_img = (pixel_count_img.cpu().numpy()*255).astype(np.uint8)
                save_pixel_count_img = Image.fromarray(save_pixel_count_img)
                save_path = debug_image_dir / f"{step}_pixel_count.jpg"
                save_pixel_count_img.save(save_path, 'JPEG')

                sample_count_dict = self.debug_sample_count_dict[i]["info_dict"]
                image_occlusion_embedding = self.occlusion_embedding(torch.tensor([i]).to(self.device))
                sample_count_dict["occlusion_embedding"] = json.dumps(image_occlusion_embedding.tolist())
                save_path = debug_image_dir / f"{step}_sample_count.json"
                with open(save_path, "w") as f:
                    json.dump(sample_count_dict, f)

                
        if self.config.save_sample_count:  
            callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=self.config.hanerf_debug_frequency,
                        func=save_sample_count,
                    )
                )

        @torch.no_grad()
        def get_reshaped_image_cords(image):
            height, width, _ = image.shape
            # create tensor of row indices and column indices
            row_indices = torch.arange(height)
            col_indices = torch.arange(width)

            # create meshgrid of row and column indices
            y_coords, x_coords = torch.meshgrid(row_indices, col_indices)

            # stack x and y coordinates into a single tensor
            image_coords = torch.stack((y_coords, x_coords), dim=-1)

            # reshape pixel coordinates tensor -> (h*w, 2)
            reshaped_image_coords = image_coords.reshape((height*width, 2))
            return reshaped_image_coords
        
        @torch.no_grad()
        def debug_image_predicted(step):
            if step == 0: return
            out_dir = training_callback_attributes.base_dir
            debug_dir = out_dir / "hanerf_debug"
            ray_generator_ref = training_callback_attributes.pipeline.datamanager.train_ray_generator
            train_dataset_ref = training_callback_attributes.pipeline.datamanager.train_dataset
            for i in self.debug_img_idxs:
                image_stem = train_dataset_ref.image_filenames[i].stem
                debug_image_dir = debug_dir / image_stem

                image = train_dataset_ref[i]['image']
                height, width, _ = image.shape
                reshaped_image_coords = get_reshaped_image_cords(image)
                total_num_coords = reshaped_image_coords.shape[0]
                camera_indices = torch.ones((total_num_coords,1)) * i

                ray_indices = torch.cat((camera_indices, reshaped_image_coords), dim=-1).long()
                ray_bundle = ray_generator_ref(ray_indices)
                total_num_rays = total_num_coords
                rays_per_chunk = min(4096, total_num_rays)
                unraveled_all_rgbs = []
                for ray_chunk_n in range(math.ceil(total_num_rays / rays_per_chunk)):
                    low_idx = ray_chunk_n * rays_per_chunk
                    up_idx = min(ray_chunk_n*rays_per_chunk + rays_per_chunk, total_num_rays)
                    ray_chunk = ray_bundle[low_idx : up_idx]
                    chunk_size = ray_chunk.shape[0]
                    chunk_outputs = self.forward(ray_chunk)
                    chunk_rgbs = chunk_outputs['rgb']
                    unraveled_all_rgbs.append(chunk_rgbs)

                unraveled_all_rgbs = torch.cat(unraveled_all_rgbs, dim=0)
                rgbs = unraveled_all_rgbs.reshape((height, width, 3)).cpu().numpy() * 255
                rgbs = rgbs.astype(np.uint8)
                pil_img = Image.fromarray(rgbs)
                save_path = debug_image_dir / f"{step}_predicted.jpg"
                pil_img.save(save_path, 'JPEG')
                
        if self.config.save_debug_predicted_images:  
            callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=self.config.hanerf_debug_frequency, 
                        func=debug_image_predicted,
                    )
                )

        @torch.no_grad()
        def debug_hanerf_occlusion_mask(step):
            if step == 0: return
            out_dir = training_callback_attributes.base_dir
            debug_dir = out_dir / "hanerf_debug"
            train_dataset_ref = training_callback_attributes.pipeline.datamanager.train_dataset
            for i in self.debug_img_idxs:
                image_stem = train_dataset_ref.image_filenames[i].stem
                debug_image_dir = debug_dir / image_stem

                image_occlusion_embedding = self.occlusion_embedding(torch.tensor([i]).to(self.device))
                image = train_dataset_ref[i]['image']
                height, width, _ = image.shape

                # fabricate pixel coordinates tensor
                reshaped_image_coords = get_reshaped_image_cords(image)
                total_num_coords = reshaped_image_coords.shape[0]
                cords_per_chunk = min(4096, total_num_coords)
                unraveled_mask = []
                for cord_chunk_n in range(math.ceil(total_num_coords / cords_per_chunk)):
                    low_idx = cord_chunk_n * cords_per_chunk
                    up_idx = min(cord_chunk_n*cords_per_chunk + cords_per_chunk, total_num_coords)
                    cord_chunk = reshaped_image_coords[low_idx : up_idx]
                    chunk_size = cord_chunk.shape[0]
                    # uv_encoded pixel coordinates tensor
                    encoded_cord_chunk = self.uv_position_encoding(cord_chunk)
                    # repated image occlusion embedding
                    rep_image_occlusion_embedding = image_occlusion_embedding.repeat(chunk_size, 1)

                    occlusion_mlp_input = torch.cat((encoded_cord_chunk, rep_image_occlusion_embedding), dim=-1)
                    unraveled_chunk_mask = self.occlusion_mask_mlp(occlusion_mlp_input)
                    unraveled_mask.append(unraveled_chunk_mask)

                unraveled_mask = torch.cat(unraveled_mask, dim=0)
                mask = unraveled_mask.reshape((height, width)).cpu().numpy() * 255
                mask = mask.astype(np.uint8)
                pil_mask = Image.fromarray(mask)
                mask_save_path = debug_image_dir / f"{step}_mask.jpg"
                pil_mask.save(mask_save_path, 'JPEG')
        
        if self.config.enable_hanerf_loss and self.config.save_debug_hanerf_occlusion_mask:
            callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=self.config.hanerf_debug_frequency,
                        func=debug_hanerf_occlusion_mask,
                    )
                )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)


        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    # xx OCCLUSION MASK
    def get_hanerf_occlusion_loss(self, image, indices, rgb):
        camera_indices = indices[:, 0]

        uv_sample = indices[:,1:3] # pixel coordinates
        uv_embedded = self.uv_position_encoding(uv_sample)
        occlusion_embedding = self.occlusion_embedding(camera_indices)
        mlp_input = torch.cat(
            [
                occlusion_embedding.view(-1, self.occlusion_embedding_dim),
                uv_embedded
            ],
            dim=-1
        )
        occlusion_uncertainty_mask = self.occlusion_mask_mlp(mlp_input) # (B, 1)
        color, mask_size, mask_digit = hanerf_occlusion_mask_loss(image, rgb, occlusion_uncertainty_mask)
        masked_rgb_loss = self.config.hanerf_loss_color_mult * color + self.config.hanerf_loss_mask_size_delta * mask_size + self.config.hanerf_loss_mask_digit_delta * mask_digit
        # normal_rgb_loss = self.rgb_loss(image, rgb) # for comparison

        return masked_rgb_loss
    
    def update_sample_count(self, indices):
        camera_indices = indices[:, 0]
        pixel_coords = indices[:,1:3] # pixel coordinates
        for i in self.debug_img_idxs:
            if i in camera_indices:
                img_pixel_coords = pixel_coords[torch.where(camera_indices == i)]
                pixel_count_img = self.debug_sample_count_dict[i]["pixel_count_img"]
                pixel_count_img[img_pixel_coords[:,0], img_pixel_coords[:,1]] += 1
                self.debug_sample_count_dict[i]["pixel_count_img"] = pixel_count_img

                sample_count_dict = self.debug_sample_count_dict[i]["info_dict"]
                sample_count_dict["img_sample_count"] += 1
                sample_count_dict["prop_pixels_sampled"] = pixel_count_img.count_nonzero().item() / (pixel_count_img.shape[0]*pixel_count_img.shape[1])
                self.debug_sample_count_dict[i]["info_dict"] = sample_count_dict
                

    # xx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        indices = batch["indices"].to(self.device)

        if self.config.save_sample_count:
            self.update_sample_count(indices)

        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            if self.config.enable_hanerf_loss:
                loss_dict["rgb_loss"] = self.get_hanerf_occlusion_loss(image, indices, outputs["rgb"])
            else:
                loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
