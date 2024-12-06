from time import time
import numpy as np
import torch

from model import Model
from datasets import fontImageGenerator, loadDataset, ImageDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = 32
    np.random.seed(seed)
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu

    fontImageGenerator('datasets/fonts/', 'datasets/fontImages/')

    X_partial, Y_partial = loadDataset('datasets/fontImages/'+'partial/')
    X_thousand, Y_thousand = loadDataset('datasets/fontImages/'+'thousand/')

    partial_dataset = ImageDataset(X_partial, Y_partial, device)
    thousand_dataset = ImageDataset(X_thousand, Y_thousand, device)

    m1 = Model(device)
    m2 = Model(device)
    
    m1.train(partial_dataset, batch_size=16)
    m2.train(thousand_dataset, batch_size=16)

    m1.test(thousand_dataset)
    m2.test(partial_dataset)


if __name__ == '__main__':
    main()
