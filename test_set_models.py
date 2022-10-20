import os
from time import sleep


def main():
    folder_path = 'logs_and_weights'

    for model_dir in os.listdir(folder_path):
        for f in os.listdir(f'{folder_path}/{model_dir}'):
            if 'epoch' in f:
                print('Testing:', model_dir)
                os.system(
                    f'python main.py --mode test --name {model_dir} --model {model_dir} --pretrained_weights {folder_path}/{model_dir}/{f} --test_folder image_data/tiny/test')
                break

if __name__ == '__main__':
    main()
