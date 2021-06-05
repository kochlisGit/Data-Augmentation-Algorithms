# random-data-augmentations
Implementation of fast Data Augmentation for Image Classification / Detection tasks.

# Supported data augmentations:
1. Random Horizontal Flip
2. Random Crop
3. Random Padding
4. Random Brightness
5. Random Saturation
6. Random Contrast
7. Random Rotation

# Example
Applying each augmentation with a probability p.

`image = sequential(image, augmentation_prob=0.8)`
