import os
import random
from datetime import datetime

import numpy as np
import cv2
from matplotlib import pyplot as plt

import albumentations as A


def overlay_transparent(background, overlay, x, y, overlay_size=None):
    bg = background.copy()

    if overlay_size is not None:
        overlay = cv2.resize(overlay.copy(), overlay_size)

    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)
    h, w = overlay_color.shape[:2]
    roi = bg[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    # Update the original image with our new ROI
    bg[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg


def augment_image_by_image(path_to_image, path_to_destiny='Dataset/augmented_logo',
                           path_to_logo='Dataset/main_logo.png', logo=True, amount=1):
    random.seed()
    if logo:
        input1 = cv2.imread(path_to_logo, cv2.IMREAD_UNCHANGED)
    input2 = cv2.imread(path_to_image)

    for _ in range(amount):
        image = input2
        if logo:
            logo = input1

            blue, green, red = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            logo[np.where((logo == [0, 0, 0, 255]).all(axis=2))] = blue, green, red, 255

            h_bg, w_bg = image.shape[:2]
            final_size_logo = random.randrange(100, min(h_bg, w_bg))
            h_offset, w_offset = random.randrange(0, h_bg - final_size_logo), \
                                 random.randrange(0, w_bg - final_size_logo)

            image = overlay_transparent(image, logo, w_offset, h_offset, (final_size_logo, final_size_logo))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augment = A.Compose([
            A.ShiftScaleRotate(p=1, shift_limit=0.02, rotate_limit=35, scale_limit=0.07),
            A.OneOf([
                A.ChannelShuffle(p=0.45),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50)
            ]),
            A.Blur(p=0.4),
            A.GaussNoise(p=0.6, var_limit=30),
            A.GridDistortion(),
            A.RandomRotate90(),
            A.ElasticTransform(approximate=True, alpha=3, p=1),
        ], p=1)

        res = augment(image=image)
        # plt.imshow(res['image'])
        # plt.show()
        filename = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + '.jpg'
        file = os.path.join(path_to_destiny, filename)
        cv2.imwrite(file, res['image'])


def augment_image_by_directory(path_to_directory, path_to_destiny, path_to_logo, amount_per_image=1):

    for filename in sorted(os.listdir(path_to_directory)):
        file = os.path.join(path_to_directory, filename)
        if os.path.isfile(file):
            augment_image_by_image(file, path_to_destiny, path_to_logo, amount=amount_per_image)


augment_image_by_image('Dataset/without_logo/5HNVJvUkRBM.jpg', 'Dataset/input', logo=False, amount=2)








