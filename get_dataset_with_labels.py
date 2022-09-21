import os
import pandas as pd
import shutil

from tqdm import tqdm

"""
A script for making the tiny_imagenet easier to work with.
Moving files from folders of ids to folders of labels.
"""

def main():
    words_file = r'image_data/words.txt'
    images_folder = r'image_data/train'
    new_images_folder = r'image_data/all_label'

    os.makedirs(new_images_folder, exist_ok=True)

    df = pd.read_csv(words_file, sep='\t', header=0, names=['id', 'label'])

    print(df)

    for folder in tqdm(os.listdir(images_folder)):
        id = df[df['id'] == folder].iloc[0, 0]
        label = df[df['id'] == folder].iloc[0, 1]
        
        try:
            label = label[:label.index(',')]
        except ValueError:
            pass
        
        label_dir = os.path.join(new_images_folder, label)

        os.makedirs(label_dir, exist_ok=True)

        for img in os.listdir(os.path.join(images_folder, id, 'images')):
            shutil.copyfile(os.path.join(images_folder, id, 'images', img), os.path.join(label_dir, img))




if __name__ == '__main__':
    main()