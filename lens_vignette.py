# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

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


@invocation(
    "lens_vignette",
    title="Lens Vignette",
    tags=["image", "postprocessing", "vignette"],
    category="image",
    version="1.1.3",
)
class LensVignetteInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Apply realistic lens vignetting to an image with configurable parameters."""

    image: ImageField = InputField(description="The image to apply vignetting to")
    intensity: float = InputField(default=0.5, ge=0.0, le=4.0, description="Intensity of the vignette effect (0-4)")
    aperture_factor: float = InputField(
        default=0.5, ge=0.2, le=1.5, description="Aperture falloff factor (>1.2 = strong, <0.5 = weak)"
    )
    center_x: float = InputField(default=0.5, ge=0.0, le=1.0, description="X coordinate of vignette center (0-1)")
    center_y: float = InputField(default=0.5, ge=0.0, le=1.0, description="Y coordinate of vignette center (0-1)")
    preserve_highlights: bool = InputField(default=True, description="Preserve highlights")

    def srgb_to_oklch(self, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB in [0,1] to OKLCH color space."""

        # Step 1: Linearize sRGB
        def linearize(c):
            a = 0.055
            return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_linear = linearize(rgb)

        # Step 2: Convert linear RGB to LMS
        lms = np.tensordot(
            rgb_linear,
            np.array(
                [
                    [0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005],
                ],
                dtype=np.float64,
            ),
            axes=([rgb.ndim - 1], [1]),
        )

        lms = np.cbrt(lms)  # cube root

        # Step 3: Convert LMS to OKLab
        lab = np.tensordot(
            lms,
            np.array(
                [
                    [0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660],
                ],
                dtype=np.float64,
            ),
            axes=([rgb.ndim - 1], [1]),
        )

        # Step 4: Convert to OKLCH
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.sqrt(a**2 + b**2)
        eps = 1e-8
        H = np.where(C > eps, np.mod(np.arctan2(b, a), 2 * np.pi), 0.0)
        H *= 360.0 / (2 * np.pi)

        return np.stack([L, C, H], axis=-1)

    def oklch_to_srgb(self, oklch: np.ndarray) -> np.ndarray:
        """Convert OKLCH to sRGB (gamma-encoded, clipped)."""
        L, C, H = oklch[..., 0], oklch[..., 1], oklch[..., 2]
        H /= 360.0 / (2 * np.pi)
        a = C * np.cos(H)
        b = C * np.sin(H)

        # Step 1: OKLab to LMS cube roots
        lab = np.stack([L, a, b], axis=-1)

        lms = np.tensordot(
            lab,
            np.array(
                [
                    [1.0000000000, 0.3963377774, 0.2158037573],
                    [1.0000000000, -0.1055613458, -0.0638541728],
                    [1.0000000000, -0.0894841775, -1.2914855480],
                ],
                dtype=np.float64,
            ),
            axes=([lab.ndim - 1], [1]),
        )

        lms = lms**3  # cube

        # Step 2: LMS to linear sRGB
        rgb_linear = np.tensordot(
            lms,
            np.array(
                [
                    [4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010],
                ],
                dtype=np.float64,
            ),
            axes=([lms.ndim - 1], [1]),
        )

        # Step 3: Gamma encode
        def encode(c):
            a = 0.055
            c_safe = np.maximum(c, 0.0)
            return np.where(c_safe <= 0.0031308, 12.92 * c_safe, (1 + a) * (c_safe ** (1 / 2.4)) - a)

        return np.clip(encode(rgb_linear), 0.0, 1.0)

    def do_highlight_preservation(
        self, img_oklch: np.ndarray, img_vignetted: np.ndarray, highlights_vignetted: np.ndarray
    ) -> np.ndarray:
        L = img_oklch[..., 0]
        highlight = np.clip((L - 0.7) / 0.4, 0.0, 1.0) ** 1.5

        result_L = highlight * highlights_vignetted[..., 0] + (1.0 - highlight) * img_vignetted[..., 0]

        # Replace only the L channel
        result = img_vignetted.copy()
        result[..., 0] = np.clip(result_L, 0.0, 1.0)
        return result

    def apply_vignette(self, image: Image.Image) -> Image.Image:
        """Apply a cosine-fourth power vignette effect with optional off-center falloff."""

        # Load and normalize image
        img = np.array(image).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = img[..., None]

        ch = img.shape[2]
        if ch == 1:
            rgb, alpha = img, None
        elif ch == 2:
            rgb, alpha = img[..., :1], img[..., 1:2]
        elif ch == 3:
            rgb, alpha = img, None
        elif ch == 4:
            rgb, alpha = img[..., :3], img[..., 3:4]
        else:
            rgb, alpha = img[..., :3], img[..., 3:] if ch > 3 else None

        img_oklch = self.srgb_to_oklch(rgb)

        height, width = img.shape[:2]
        y, x = np.mgrid[0:height, 0:width]
        cx = self.center_x * width
        cy = self.center_y * height

        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(
            max(
                cx**2 + cy**2,
                (width - cx) ** 2 + cy**2,
                cx**2 + (height - cy) ** 2,
                (width - cx) ** 2 + (height - cy) ** 2,
            )
        )
        r = distance / max_dist

        # Cosine-fourth law
        vignette = np.cos(np.clip(r * self.aperture_factor, 0.0, 1.0) * (np.pi / 2)) ** 4
        vignette_mask = 1.0 - self.intensity * (1.0 - vignette)
        vignette_mask = vignette_mask[..., None]  # expand for broadcast
        vignette_mask = np.clip(vignette_mask, 0.0, 1.0)

        img_vignetted = img_oklch.copy()
        img_vignetted[..., 0] *= vignette_mask[..., 0]  # apply only to L channel

        if self.preserve_highlights:
            highlight_mask = 1.0 - (self.intensity * 0.5) * (1.0 - vignette)
            highlight_mask = highlight_mask[..., None]
            highlight_mask = np.clip(highlight_mask, 0.0, 1.0)

            highlights_vignetted = img_oklch.copy()
            highlights_vignetted[..., 0] *= highlight_mask[..., 0]

            img_oklch = self.do_highlight_preservation(img_oklch, img_vignetted, highlights_vignetted)
        else:
            img_oklch = img_vignetted

        rgb_srgb = self.oklch_to_srgb(img_oklch)

        if alpha is not None:
            out = np.concatenate([rgb_srgb, alpha], axis=2)
        else:
            out = rgb_srgb

        result = (out * 255.0).clip(0, 255).astype(np.uint8)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        return Image.fromarray(result)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_image = context.images.get_pil(self.image.image_name)
        result_image = self.apply_vignette(pil_image)

        image_dto = context.images.save(image=result_image)

        return ImageOutput.build(image_dto)
