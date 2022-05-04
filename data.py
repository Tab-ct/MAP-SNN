from scipy.io import loadmat 
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import snntorch
from snntorch.spikevision import spikedata


import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms, utils


def load_data(args):

    if args.dataset == "N-MNIST":  
        train_ds = spikedata.NMNIST("data/nmnist", train=True, num_steps=args.steps, dt=int(300000/args.steps))
        test_ds = spikedata.NMNIST("data/nmnist", train=False, num_steps=args.steps, dt=int(300000/args.steps))
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.train_batch_size, pin_memory=True, num_workers=8)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.eval_batch_size, pin_memory=True, num_workers=8)

    if args.dataset == "SHD":
        train_ds = spikedata.SHD("data/shd", train=True, num_steps=args.steps, dt=int(800*1000/args.steps))
        test_ds = spikedata.SHD("data/shd", train=False, num_steps=args.steps, dt=int(800*1000/args.steps))
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.train_batch_size, pin_memory=True)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.eval_batch_size, pin_memory=True)

    return train_dl, test_dl
