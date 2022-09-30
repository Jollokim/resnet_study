import os
import random
import numpy as np
import cv2 as cv

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
        raise NotImplementedError('shear_image has not been implemented for shearing facotr of -1 yet.')
    
    img_shear = np.full((int(img.shape[0]), int(off_bound_x), 3), 255, dtype=np.float64)

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
    img_pad = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT,value=[0, 0, 0])

    x = random.randint(0, pad*2)     
    y = random.randint(0, pad*2)

    print(x, y)
    x = 16
    y = 16

    return img_pad[x:64+x, y:64+y] 

def main():
    img_path = 'image_data/tiny/train/American alligator/n01698640_0.JPEG'

    img = cv.imread(img_path)

    cv.imwrite(r'image_data/test.png', img)

    pad = 8

    img_pad = pad_crop_img(img, pad)

    cv.imwrite('image_data/pad_test.png', img_pad)

    img_noise = noise_image(img, 50)

    cv.imwrite(r'image_data/noise_test.png', img_noise)


if __name__ == "__main__":
    main()