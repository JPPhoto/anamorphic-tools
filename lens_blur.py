# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import cv2
import numpy as np
from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


@invocation("lens_blur", title="Lens Blur", tags=["image", "lens", "blur"], version="1.0.0")
class LensBlurInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Adds lens blur to the input image, first converting the input image to RGB"""

    image: ImageField = InputField(description="The image to streak")
    depth_map: ImageField = InputField(description="The depth map to use")
    focal_distance: float = InputField(
        description="The distance at which focus is sharp: 0.0 (near) - 1.0 (far)",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    distance_range: float = InputField(
        description="The depth of field range around the focal distance",
        default=0.2,
        ge=0.0,
        le=1.0,
    )
    max_blur_radius: float = InputField(description="Maximum blur radius", default=5.0, gt=0.0)
    max_blur_steps: int = InputField(description="Number of blur maps to use internally", default=32, gt=1)
    anamorphic_factor: float = InputField(description="Anamorphic squeeze factor", default=1.35, gt=1.0)
    highlight_threshold: float = InputField(
        description="The luminance threshold at which highlight enhancement begins",
        default=0.75,
        ge=0.0,
        le=1.0,
    )

    highlight_factor: float = InputField(
        description="Minimum multiplier for highlights at the threshold",
        default=1.0,
        ge=1.0,
    )

    highlight_factor_high: float = InputField(
        description="Maximum multiplier for highlights at full brightness",
        default=2.0,
        ge=1.0,
    )

    def apply_anamorphic_gaussian_blur(self, image: Image, sigma_y: float, anamorphic_squeeze: float) -> np.ndarray:
        sigma_x = sigma_y / anamorphic_squeeze

        image_numpy = image.astype(np.float32)

        # Let OpenCV auto-calculate optimal kernel size from sigma
        ksize = (0, 0)

        # Apply the blur
        blurred = cv2.GaussianBlur(image_numpy, ksize, sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)

        return blurred

    def compute_blur_indices(self, depth_normalized: np.ndarray, num_blurred_images: int) -> np.ndarray:
        # Use an asymmetric depth of field, using a 1/3 front, 2/3 back rule around the focal plane.
        dof_front = 1 * self.distance_range / 3.0
        dof_back = 2 * self.distance_range / 3.0

        below = depth_normalized < self.focal_distance
        above = ~below

        distance_from_focus = np.zeros_like(depth_normalized)

        # Front of focal plane
        distance_from_focus[below] = np.clip((self.focal_distance - depth_normalized[below] - dof_front), 0.0, 1.0)

        # Behind focal plane
        distance_from_focus[above] = np.clip((depth_normalized[above] - self.focal_distance - dof_back), 0.0, 1.0)

        # Scale to index range
        indices = distance_from_focus * (num_blurred_images - 1)
        return indices

    def apply_blur_based_on_depth(self, blurred_images: list[np.ndarray], depth_map: np.ndarray) -> np.ndarray:
        num_blurred_images = len(blurred_images)
        # Normalize and invert depth map
        depth_normalized = 1 - cv2.normalize(depth_map.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

        indices = self.compute_blur_indices(depth_normalized, num_blurred_images)

        lower_indices = np.floor(indices).astype(np.int32)
        upper_indices = np.clip(lower_indices + 1, 0, num_blurred_images - 1)
        alpha = (indices - lower_indices)[..., np.newaxis]  # Shape: (H, W, 1)

        output = np.zeros_like(blurred_images[0], dtype=np.float32)

        for i in range(num_blurred_images):
            mask_lower = lower_indices == i
            mask_upper = upper_indices == i

            if blurred_images[0].ndim == 3:
                mask_lower = mask_lower[..., np.newaxis]
                mask_upper = mask_upper[..., np.newaxis]

            output += blurred_images[i] * mask_lower * (1 - alpha) + blurred_images[i] * mask_upper * alpha

        return output

    def enhance_highlights(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0

        # Compute per-pixel luminance (ITU-R BT.709)
        luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]

        # Calculate per-pixel enhancement factor
        mask = luminance >= self.highlight_threshold
        t = np.clip((luminance - self.highlight_threshold) / (1.0 - self.highlight_threshold), 0.0, 1.0)
        factor = self.highlight_factor * (1.0 - t) + self.highlight_factor_high * t

        # Expand factor to 3 channels
        factor_expanded = np.where(mask[..., None], factor[..., None], 1.0)

        # Apply enhancement
        enhanced = image * factor_expanded
        enhanced = np.clip(enhanced, 0.0, 1.0) * 255.0

        return enhanced

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        depth_map = context.images.get_pil(self.depth_map.image_name)

        image_mode = image.mode
        image = image.convert("RGB")
        image_numpy = np.array(image.convert("RGB"))
        image_numpy = self.enhance_highlights(image_numpy)

        depth_map = depth_map.convert("L")
        depth_map_numpy = np.array(depth_map)

        img_blurred = [None] * self.max_blur_steps
        img_blurred[0] = image_numpy
        for blur in range(1, self.max_blur_steps):
            img_blurred[blur] = self.apply_anamorphic_gaussian_blur(
                image_numpy, blur * self.max_blur_radius / self.max_blur_steps, self.anamorphic_factor
            )

        img_blurred = self.apply_blur_based_on_depth(img_blurred, depth_map_numpy)
        img_blurred = np.clip(img_blurred, 0, 255).astype(np.uint8)
        lens_blurred_image = Image.fromarray(img_blurred).convert(image_mode)

        image_dto = context.images.save(image=lens_blurred_image)

        return ImageOutput.build(image_dto)
