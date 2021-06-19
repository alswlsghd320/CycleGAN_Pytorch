from train import train
from test import test
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train is True:
        train()
    elif args.eval is True:
        test()