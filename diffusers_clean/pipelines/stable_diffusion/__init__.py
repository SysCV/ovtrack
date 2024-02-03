import numpy as np
import PIL
from dataclasses import dataclass
from PIL import Image
from typing import List, Union

from ...utils import BaseOutput, is_transformers_available


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: List[bool]


if is_transformers_available():
    from .pipeline_ovtrack_image_generation import \
        StableDiffusionOVTrackPipeline


