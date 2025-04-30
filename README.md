# anamorphic-tools
Provides InvokeAI nodes for applying anamorphic lens blur, anamorphic flares, spherical distortion, and lens
vignette to images.

Also provided are an aperture generator and loader. The generator's output can be fed into the lens blur node for
different aperture effects. The same goes for the Load Aperture Image node; you can provide your own aperture images
in the `aperture_images` folder. (Note that all images loaded will become square as that is the expectation of the
Lens Blur node.)

The ideal pipeline to use when postprocessing images - one that reflects the real optical characteristics of an
anamorphic lens connected to a camera - is:

Image &rarr; Lens Vignette &rarr; Lens Blur &rarr; Spherical Distortion &rarr; Anamorphic Streaks

## Examples:

![image](https://github.com/user-attachments/assets/86b596f3-5f8a-4fc6-a9e1-fee613002729)

![image](https://github.com/user-attachments/assets/0ec688c5-5903-44d0-9e13-7c0d6bccb20d)

![image](https://github.com/user-attachments/assets/969a5668-e3d6-4727-96f4-352500dea1cc)

![image](https://github.com/user-attachments/assets/c34a0d03-cf65-4fc2-9eb6-e4feeebb4c53)

![image](https://github.com/user-attachments/assets/e56375f7-862a-4bfc-8b43-d284f8477cac)
