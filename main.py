import argparse
from snntorch.spikevision import spikedata
import data
import model

from train import train_snn, print_results, test_snn

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import os

if __name__ == '__main__':
################################
########   Args Init   #########
################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr",type=float,default=1e-3)             # learning rate
    parser.add_argument("--num_epochs",type=int,default=100)        # training epoch
    parser.add_argument("--gpu_id",type=int,default=0)             
    parser.add_argument("--criterion",type=str,default="CrossEntropy",choices=['CrossEntropy','MSE'])  
    parser.add_argument("--optimizer",type=str,default='Adam',choices=['Adam','SGD'])
    parser.add_argument("--dataset",type=str,default="SHD",choices=['SHD','N-MNIST'])
    parser.add_argument("--train_batch_size",type=int,default=128)    # batch size of train samples
    parser.add_argument("--eval_batch_size",type=int,default=128)     # batch size of test samples
    parser.add_argument("--n_classes",type=int,default=20)            # number of classes
    parser.add_argument("--train_acc_print_time",type=int,default=4)  # print times of each training epoch
    parser.add_argument("--steps",type=int,default=50)                # timesteps of datasets
    args = parser.parse_args()

    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu_id))

    print(f"Device Using: {device}")



##################################
########  Model Creating  ########
##################################
    if args.dataset == "SHD":
        snn = model.SNN_model_for_SHD_MAP_SNN(device=device)
    else:
        snn = model.SNN_model_for_N_MNIST_MAP_SNN(device=device)
    snn = snn.to(device)

################################
########  Data Loading  ########
################################
    train_loader, test_loader = data.load_data(args)    

##################################
########  Model Training  ########
##################################
    acc_record, loss_test_record = train_snn(snn, args, train_loader, test_loader, device=device)
    # torch.save(snn, "./snn_model.cp") 


##################################
########  Print Result  ##########
##################################
# print_results(acc_record, loss_test_record)



