import os
from typing import Literal

from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    ImageOutput,
    InputField,
    InvocationContext,
    invocation,
)


def get_aperture_images() -> list[str]:
    curdir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(curdir, "aperture_images")
    image_files = []

    if os.path.exists(images_dir):
        for fname in os.listdir(images_dir):
            fpath = os.path.join(images_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with Image.open(fpath):
                    image_files.append(fname)
            except Exception:
                continue

    if not image_files:
        image_files.append("--NO IMAGES FOUND--")

    return image_files


@invocation("load_aperture_image", title="Load Aperture Image", tags=["aperture", "image", "load"], version="1.0.0")
class LoadApertureImageInvocation(BaseInvocation):
    """Loads a grayscale aperture image from the presets directory."""

    aperture_image: Literal[tuple(get_aperture_images())] = InputField(
        default=get_aperture_images()[0],
        description="The aperture image to load from the 'aperture_images' folder",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        curdir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(curdir, "aperture_images", self.aperture_image)

        image = Image.open(path).convert("L")
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
