import os

def main():
    path = 'HPO_optimizer'

    mx_acc = 0
    best_exp = ''

    for exp in os.listdir(path):
        try:
            with open(f'{path}/{exp}/test_results.txt', 'r') as f:
                acc = float(f.readline())

                if mx_acc < acc:
                    mx_acc = acc
                    best_exp = exp
        except FileNotFoundError:
            pass

    print(mx_acc)
    print(best_exp)
            

if __name__ == '__main__':
    main()