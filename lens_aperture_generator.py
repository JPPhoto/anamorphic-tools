# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


@invocation(
    "lens_aperture_generator", title="Lens Aperture Generator", tags=["image", "lens", "aperture"], version="1.0.1"
)
class LensApertureGeneratorInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Generates a grayscale aperture shape with adjustable number of blades, curvature, rotation, and softness."""

    number_of_blades: int = InputField(
        default=6, ge=3, description="Number of aperture blades (e.g., 5 for pentagon, 6 for hexagon)"
    )
    rounded_blades: bool = InputField(default=True, description="Use curved edges between aperture blades")
    curvature: float = InputField(default=0.5, ge=0.0, le=1.0, description="Aperture blade curvature (1.0 is circular)")
    rotation: float = InputField(
        default=0.0, ge=0.0, le=360.0, description="Clockwise rotation of the aperture shape in degrees"
    )
    softness: float = InputField(
        default=0.1, ge=0.0, le=1.0, description="Softness of the aperture edge (0.0 = hard, 1.0 = feathered)"
    )

    def generate_regular_polygon(
        self, cx: float, cy: float, radius: float, sides: int, rotation_deg: float
    ) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        angles += np.deg2rad(rotation_deg)
        return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1).astype(np.float32)

    def get_center_and_radius(
        self, p1: Tuple[float], p2: Tuple[float], p3: Tuple[float]
    ) -> Tuple[Tuple[float, float], float]:
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

        # Determinant
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if D == 0:
            raise ValueError("Points are collinear; no unique circle exists.")

        # Coordinates of the circumcenter
        x1_sq = x1**2 + y1**2
        x2_sq = x2**2 + y2**2
        x3_sq = x3**2 + y3**2

        cx = (x1_sq * (y2 - y3) + x2_sq * (y3 - y1) + x3_sq * (y1 - y2)) / D
        cy = (x1_sq * (x3 - x2) + x2_sq * (x1 - x3) + x3_sq * (x2 - x1)) / D

        # Radius
        r = math.hypot(cx - x1, cy - y1)

        return (cx, cy), r

    def generate_rounded_polygon(
        self, cx: float, cy: float, radius: float, sides: int, rotation_deg: float, curvature: float
    ) -> np.ndarray:
        rotation_rad = np.deg2rad(rotation_deg) % (2 * np.pi)
        angle_step = 2 * np.pi / sides
        points = []

        # Compute the apothem and curve radius given our curvature
        apothem = radius * np.cos(np.pi / sides)
        curve_radius = apothem * (1 - curvature) + radius * curvature

        for i in range(sides):
            theta1 = i * angle_step + rotation_rad
            theta2 = (i + 1) * angle_step + rotation_rad
            theta_between = (i + 0.5) * angle_step + rotation_rad

            x1, y1 = cx + radius * np.cos(theta1), cy + radius * np.sin(theta1)
            x_between, y_between = cx + curve_radius * np.cos(theta_between), cy + curve_radius * np.sin(theta_between)
            x2, y2 = cx + radius * np.cos(theta2), cy + radius * np.sin(theta2)

            if curvature > 0.0:
                (cx_arc, cy_arc), radius_arc = self.get_center_and_radius((x1, y1), (x_between, y_between), (x2, y2))

                # Figure out the radians between the n-gon points in terms of the arc center, which will always
                # be in the same direction as the center of the n-gon
                t1 = math.atan2(y1 - cy_arc, x1 - cx_arc)
                t2 = math.atan2(y2 - cy_arc, x2 - cx_arc)

                # Make sure our rotation is always CCW and increasing since atan2 can give us <0
                if t1 < 0:
                    t1 = t1 + 2 * np.pi
                while t2 < 0 or t2 < t1:
                    t2 = t2 + 2 * np.pi

                arc_resolution = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) // 10 + 1
            else:
                (cx_arc, cy_arc), radius_arc = (cx, cy), radius
                t1, t2 = theta1, theta2
                arc_resolution = 1

            for t in np.linspace(t1, t2, arc_resolution):
                xt, yt = cx_arc + radius_arc * np.cos(t), cy_arc + radius_arc * np.sin(t)
                points.append((xt, yt))

        return np.array(points, dtype=np.float32)

    def apply_softness(self, mask: np.ndarray, softness: float) -> np.ndarray:
        if softness <= 0.0:
            return mask

        size = mask.shape[0]
        if mask.shape[0] != mask.shape[1]:
            raise ValueError(f"apply_softness only supports square masks, but got shape {mask.shape}")

        # Blur relative to original image size
        blur_radius = int(size * softness / 2)
        blur_diameter = 2 * blur_radius + 1

        # Limit expansion to the blur radius
        pad = blur_radius

        # Pad to slightly larger canvas
        expanded_mask = cv2.copyMakeBorder(
            mask, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_CONSTANT, value=0.0
        )

        blurred = cv2.GaussianBlur(expanded_mask, (blur_diameter, blur_diameter), sigmaX=0)

        # Resize back to original size
        softened = cv2.resize(blurred, (size, size), interpolation=cv2.INTER_AREA)

        return softened

    def generate_aperture_image(self, size: int) -> np.ndarray:
        mask_size = size * 16
        cx, cy = mask_size / 2, mask_size / 2
        radius = mask_size * 0.5

        if self.rounded_blades:
            points = self.generate_rounded_polygon(cx, cy, radius, self.number_of_blades, self.rotation, self.curvature)
        else:
            points = self.generate_regular_polygon(cx, cy, radius, self.number_of_blades, self.rotation)

        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], color=255, lineType=cv2.LINE_AA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
        mask = mask.astype(np.float32) / 255.0

        mask = self.apply_softness(mask, self.softness)

        return mask

    def invoke(self, context: InvocationContext) -> ImageOutput:
        size = 512  # Output image is always 512x512
        aperture_image = self.generate_aperture_image(size)

        image = np.clip(aperture_image * 255.0, 0.0, 255.0)
        image = Image.fromarray(image.astype(np.uint8), mode="L")
        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
