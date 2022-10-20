import os
import numpy as np
from scipy.stats import wilcoxon

def compare_models(acc1, acc2):
    _, p = wilcoxon([acc1], [acc2], alternative='greater')

    return p

def calculate_95ci(acc, n):
    z = 1.96
    std = np.sqrt((acc*(1-acc) / n))

    return z * std

def main():
    path = 'logs_and_weights'

    mx_acc = 0
    best_exp = ''

    for exp in os.listdir(path):
        try:
            with open(f'{path}/{exp}/test_results.txt', 'r') as f:
                acc = float(f.readline())
                print(acc, f'+-{calculate_95ci(acc, 10000)}')
                print(exp, '\n')

                if mx_acc < acc:
                    mx_acc = acc
                    best_exp = exp
        except FileNotFoundError:
            pass

    print('\nBest:')
    print(mx_acc)
    print(best_exp)

    print(compare_models(0.4625, 0.4484))
            

if __name__ == '__main__':
    main()