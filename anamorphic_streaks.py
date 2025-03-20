# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from invokeai.invocation_api import (
    BaseInvocation,
    ColorField,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


@invocation("anamorphic_streaks", title="Anamorphic Streaks", tags=["image", "anamorphic"], version="1.0.0")
class AnamorphicStreaksInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Adds anamorphic streaks to the input image"""

    image: ImageField = InputField(description="The image to streak")
    streak_color: ColorField = InputField(
        default=ColorField(r=0, g=100, b=255, a=255),
        description="The color to use for streaks",
    )
    gamma: float = InputField(description="Gamma to use for finding highlights", default=30.0)
    blur_radius: float = InputField(description="Radius to blur highlights", default=5.0)
    erosion_kernel_size: int = InputField(description="Amount to erode blurred highlights", default=5)
    erosion_iterations: int = InputField(description="Number of erosion iterations", default=2)
    streak_intensity: float = InputField(description="Streak Intensity", default=10)
    internal_reflection_strength: float = InputField(
        ge=0.0, le=1.0, description="Internal reflection strength", default=0.3
    )

    # Create a fading line kernel
    def create_fading_line_kernel(self, length):
        # Create a 1D kernel that fades from center to edges
        x = np.linspace(-1, 1, length)
        kernel_1d = 1 - x**2  # Parabolic fade (you can use other functions too)

        # Reshape to 2D horizontal line
        kernel_2d = kernel_1d.reshape(1, -1)

        # Normalize so sum is 1.0
        return kernel_2d / kernel_2d.sum()

    # Apply the convolution to the binary mask
    def apply_faded_line_convolution(self, mask, line_length):
        # Create the fading line kernel
        kernel = self.create_fading_line_kernel(line_length)

        # Convert binary mask to float for convolution
        mask_float = mask.astype(float)

        # Apply convolution
        result = cv2.filter2D(mask_float, -1, kernel)

        # Normalize result to 0-1 range
        return result

    def add_horizontal_streaks(self, image):
        width, height = image.size

        # Convert to numpy array for processing
        img_array = np.array(image)

        # Create i2 as the L channel with gamma correction
        if len(img_array.shape) == 3:  # RGB or RGBA
            # Convert to grayscale (L channel)
            i2 = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)
        else:  # Already grayscale
            i2 = img_array.copy()

        # Apply gamma correction
        i2 = np.power(i2 / 255.0, self.gamma) * 255.0
        i2 = i2.astype(np.uint8)

        # Apply Gaussian blur
        blurred = gaussian_filter(i2, sigma=self.blur_radius)

        # Convert to OpenCV format for erosion
        blurred_cv = blurred.astype(np.uint8)

        # Create horizontal kernel for erosion
        kernel = np.ones((1, self.erosion_kernel_size), np.uint8)

        # Apply erosion to create streak mask
        streak_mask = cv2.erode(blurred_cv, kernel, iterations=self.erosion_iterations)

        # Normalize mask to 0..1 range
        streak_mask = streak_mask / 255.0

        streak_mask = self.apply_faded_line_convolution(streak_mask, line_length=width)

        # Create diagonally flipped version (both horizontal and vertical)
        hv_flipped = np.flipud(np.fliplr(streak_mask))

        # Combine original with flipped versions
        streak_mask = streak_mask + (hv_flipped * self.internal_reflection_strength)

        # Create output image
        result = img_array.copy()

        # Apply streaks
        streak_color = self.streak_color.tuple()

        if len(img_array.shape) == 3:  # RGB or RGBA
            channels = min(3, img_array.shape[2])  # Handle both RGB and RGBA
            for c in range(channels):
                result[:, :, c] = np.clip(
                    img_array[:, :, c] + streak_color[c] * streak_mask * self.streak_intensity, 0, 255
                ).astype(np.uint8)
        else:  # Grayscale
            streak_gray = 0.299 * streak_color[0] + 0.587 * streak_color[1] + 0.114 * streak_color[2]
            result = np.clip(img_array + streak_gray * streak_mask * self.streak_intensity, 0, 255).astype(np.uint8)

        # Convert back to PIL image
        return Image.fromarray(result)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        streaked_image = self.add_horizontal_streaks(image)

        image_dto = context.images.save(image=streaked_image)

        return ImageOutput.build(image_dto)
