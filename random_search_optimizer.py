from main import main
import argparse

import numpy as np
from numpy.random import SeedSequence, MT19937, RandomState

# function for defining all the commandline parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)

    # Model mode
    parser.add_argument('--seed', type=int, required=True,
                        help='seed for HPO')
    return parser

class Arguments:
    def __init__(self,
        mode,
        model,
        name, 
        lr, 
        beta1, 
        beta2, 
        eps, 
        weight_decay, 
        start_factor, 
        end_factor, 
        seed,
        train_folder,
        valid_folder,
        test_folder,
        pretrained_weights=None,
        no_validate=False,
        log_path='HPO_optimizer/',
        batch_size=32,
        epochs=25,
        num_workers=5,
    ) -> None:

        self.mode = mode
        self.model = model
        self.name = name
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.start_factor = start_factor
        self.end_factor = end_factor

        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.test_folder = test_folder

        self.pretrained_weights = pretrained_weights

        self.seed = seed

        self.no_validate = no_validate
        self.log_path = log_path

        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers

        self.overwrite = None


def start():
    arg_parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    cmd_args = arg_parser.parse_args()

    rng = RandomState(MT19937(SeedSequence(cmd_args.seed)))
    
    weight_decay_pool = [10**-5, 10**-4, 10**-3, 0]

    for i in range(15):

        hp = {
            'eps': rng.uniform(10**-9, 10**-7),
            'lr': rng.uniform(10**-4, 2*(10**-3)),
            'beta1': 1 - rng.uniform(10**-3, 2*(10**-1)),
            'beta2': 1 - rng.uniform(10**-4, 10**-1),
            'weight_decay': weight_decay_pool[rng.randint(0, len(weight_decay_pool))],
            'start_factor': rng.uniform(0.5, 1),
        }

        print(hp)

        args = Arguments(
            mode='train',
            model='ResNet56Projection',
            name=f'seed{cmd_args.seed}_model{i}',
            lr=hp['lr'],
            beta1=hp['beta1'],
            beta2=hp['beta2'],
            eps=hp['eps'],
            weight_decay=hp['weight_decay'],
            start_factor=1,
            end_factor=hp['start_factor'],
            seed=1,
            train_folder='image_data/tiny/train',
            valid_folder='image_data/tiny/valid',
            test_folder='image_data/tiny/test',
        )

        main(args)

        args.mode = 'test'

        main(args)

        with open(f'{args.log_path}/{args.name}/hp_config.txt', 'a') as f:
            for key in hp.keys():
                f.write(f'{key}={hp[key]}\n')

if __name__ == '__main__':
    start()