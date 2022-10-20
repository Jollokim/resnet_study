from logging import root
import os
import random
import numpy as np
import cv2 as cv
from tqdm import tqdm

# modified version of: https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition/blob/main/aug_images.py


def noise_image(img, variability):
    deviation = variability*random.random()

    noise = np.int32(np.random.normal(0, deviation, img.shape))
    img = np.int32(img)
    img += noise
    img = np.uint8(np.clip(img, 0., 255.))
    return img


def shear_image(img, x_shearing_factor):
    if x_shearing_factor >= 0:
        off_bound_x = (img.shape[1])+(x_shearing_factor*img.shape[0])
    else:
        raise NotImplementedError(
            'shear_image has not been implemented for shearing facotr of -1 yet.')

    img_shear = np.full((int(img.shape[0]), int(
        off_bound_x), 3), 255, dtype=np.float64)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x = col + (x_shearing_factor*row)
            y = row

            # if x <= -1 or y <= -1:
            #     continue
            try:
                img_shear[int(y)][int(x)] = img[row, col]
            except IndexError:
                pass

    return img_shear


def resize_img(img):
    img = cv.resize(img, (64, 64)).copy()

    return img


def pad_crop_img(img, pad):
    img_pad = cv.copyMakeBorder(
        img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=[0, 0, 0])

    x = random.randint(0, pad*2)
    y = random.randint(0, pad*2)

    # print(x, y)
    # x = 16
    # y = 16

    return img_pad[x:64+x, y:64+y]


def aug_img(img):
    img = cv.imread(img)

    img = noise_image(img, 25)

    if random.randint(0, 1) == 0:
        pad = 8
        img = pad_crop_img(img, pad)

    return img


def main():
    root_dataset = 'image_data/tiny/train'
    aug_dataset = 'image_data/tiny/train_aug'
    multiplier = 3

    for cls in tqdm(os.listdir(root_dataset)):
        for img in os.listdir(f'{root_dataset}/{cls}'):
            os.makedirs(f'{aug_dataset}/{cls}', exist_ok=True)

            cv.imwrite(f'{aug_dataset}/{cls}/{img}',
                       cv.imread(f'{root_dataset}/{cls}/{img}'))

            for i in range(multiplier):
                cv.imwrite(f'{aug_dataset}/{cls}/{img}_aug{i}.png',
                           aug_img(f'{root_dataset}/{cls}/{img}'))


if __name__ == "__main__":
    main()
