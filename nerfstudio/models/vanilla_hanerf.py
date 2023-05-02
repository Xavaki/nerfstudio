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
Implementation of vanilla ha-nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.model_components.losses import hanerf_occlusion_mask_loss



@dataclass
class VanillaHanerfModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: VanillaHanerfModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""

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


class VanillaHanerfModel(Model):
    """Vanilla HaNeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    def __init__(
        self,
        config: VanillaHanerfModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

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

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())

        if self.config.enable_hanerf_loss:
            param_groups["losses"] = list(self.occlusion_mask_mlp.parameters()) + list(self.occlusion_embedding.parameters())

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times)
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

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

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        indices = batch["indices"].to(self.device)

        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])

        if self.config.save_sample_count:
            self.update_sample_count(indices)
        
        if self.training and self.config.enable_hanerf_loss:
            rgb_loss_fine = self.get_hanerf_occlusion_loss(image, indices, outputs["rgb_fine"])
        else:
            rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
    

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        
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
