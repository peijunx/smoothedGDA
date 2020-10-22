from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict


import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

from model import ConvNet

import numpy as np

import matplotlib.pyplot as plt

from lossfns import *
from adversary import *
from dataprocess import *

import os

import time
from torch.autograd import Variable as V

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args):
    
    loader_train, loader_test = loadData(args)
    dtype = torch.cuda.FloatTensor
    
    model = unrolled(args, loader_train, loader_test, dtype)

    fname = "smoothed_gda_model/MNIST_CWM_retain.pth"
    torch.save(model, fname)

    print("Training done, model save to %s :)" % fname)
    
    pgdAttackTest(model, loader_test, dtype)
    fgsmAttackTest(model, loader_test, dtype)


def unrolled(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()
        
    SCHEDULE_EPOCHS = [50, 50] 
    learning_rate = 5e-4

    p = 0.2 # parameter for the quadratic term
    beta = 0.8 # parameter for the smoothed term
    alpha = 0.5 # parameter for update t
    
    initialized_t = torch.zeros(10, 1)
    initialized_t[0, 0] = 1 # satisfy the simplex condition
    t = Variable(initialized_t, requires_grad=True)
    t = t.cuda()
    
    for num_epochs in SCHEDULE_EPOCHS:
        
        print('\nTraining %d epochs with learning rate %.5f' % (num_epochs, learning_rate))
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        w = list(model.parameters())[0].data
        z = w # z is used to smoothed primal iterate w := model.parameters()
        
        for epoch in range(num_epochs):
            
            print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
            # print(model.training)
            
            for i, (X_, y_) in enumerate(loader_train):

                X = Variable(X_.type(dtype), requires_grad=False)
                y = Variable(y_.type(dtype), requires_grad=False)

                ########################################################################
                # start updating optimizer variables [(w, b), t, z]

                N = X.shape[0]
                X = X.repeat(1, 10, 1, 1).reshape(N * 10, 1, 28, 28)
                X_copy = X.clone()
                X.requires_grad = True

                eps = 0.4

                y = y.view(-1, 1).repeat(1, 10).view(-1, 1).long().cuda()

                index = torch.tensor([jj for jj in range(10)] * N).view(-1, 1).cuda().long()

                MaxIter_max = 11
                step_size_max = 0.1

                for generate_sample_i in range(MaxIter_max):

                    output = model(X)
                    
                    maxLoss = (output.gather(1, index) - output.gather(1, y)).mean()

                    X_grad = torch.autograd.grad(maxLoss, X, retain_graph=True)[0]
                    X = X + X_grad.sign() * step_size_max
                    
                    X.data = X_copy.data + (X.data - X_copy.data).clamp(-eps, eps)
                    X.data = X.data.clamp(0, 1)

                preds = model(X)

                loss = (-F.log_softmax(preds)).gather(1, y).view(-1, 10).mean(dim=0).view(-1, 10)
                
                # then dot product with t and times the regularization term
                w = list(model.parameters())[0].data
                loss = torch.mm(loss, t) + (z - w).cuda().pow(2).sum() * p / 2

                ## START with update w
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## DONE with update w
                ## START with update t

                preds = model(X) # use the updated weights w to get the updated predictions
                current_loss = (-F.log_softmax(preds)).gather(1, y).view(-1, 10).mean(dim=0)

                t = t + alpha * current_loss.view(10, 1)
                t, _ = torch.sort(t, dim=0, descending=True)

                # assume t is sort from large to small t1 >= t2
                u = V(t.cpu(), requires_grad=False)

                num_pos_vales = 0 # number of positive values in the projected solution (will compute)
                for l in range(1, 10+1):
                    check_value = u[l-1, 0] + (1 - sum(u[0:l, 0])) / l
                    if check_value <= 0:
                        num_pos_vales = l-1;
                        break

                if num_pos_vales == 0:
                    num_pos_vales = 10

                lambda_ = (1 - sum(u[0:num_pos_vales, 0])) / num_pos_vales
                t = F.relu(lambda_ + t)
                t = Variable(t, requires_grad=True)
                t = t.cuda()

                ## DONE with udpate t
                ## START with update z
                w = list(model.parameters())[0].data
                z = z + beta * (w - z)

                ## DONE with udpate z
                ## DONE with updating optimizer variables

                ## GET true loss
                loss_wo_t = (-F.log_softmax(preds)).gather(1, y).view(-1, 10).mean(dim=0).view(-1, 10) # l_i before multiplied by t_i
                true_loss = torch.mm(loss_wo_t, t)
                ########################################################################

                if (i + 1) % args.print_every == 0:
                    print('Batch %d done, loss = %.7f\n' % (i + 1, true_loss.item()))

                    non_zero_indices = (t != 0).nonzero().split(1, dim=1)
                    print("loss values corresponding to nonzero ti\n")
                    print(loss_wo_t.view(10, 1)[non_zero_indices])

                    print('\nBatch %d done, number of postive entries in t = %d' % (i+1, num_pos_vales))


                    test(model, loader_test, dtype)

            print('Batch %d done, loss = %.7f' % (i + 1, true_loss.item()))
            
            pgdAttackTest(model, loader_test, dtype)
            fgsmAttackTest(model, loader_test, dtype)
            
        
        learning_rate *= 0.1

    return model


def test(model, loader_test, dtype):
    num_correct = 0
    num_samples = 0
    model.eval()
    for X_, y_ in loader_test:

        X = Variable(X_.type(dtype), requires_grad=False)
        y = Variable(y_.type(dtype), requires_grad=False).long()

        logits = model(X)
        _, preds = logits.max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    accuracy = float(num_correct) / num_samples * 100
    print('\nAccuracy = %.2f%%' % accuracy)
    model.train()

def normal_train(args, loader_train, loader_test, dtype):

    model = ConvNet()
    model = model.type(dtype)
    model.train()
        
    loss_f = nn.CrossEntropyLoss()

    SCHEDULE_EPOCHS = [15] 
    learning_rate = 0.01
    
    for num_epochs in SCHEDULE_EPOCHS:

        start_time = time.time()
        
        print('\nTraining %d epochs with learning rate %.4f' % (num_epochs, learning_rate))
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            
            print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
            # print(model.training)
            
            for i, (X_, y_) in enumerate(loader_train):

                X = Variable(X_.type(dtype), requires_grad=False)
                y = Variable(y_.type(dtype), requires_grad=False).long()

                preds = model(X)

                loss = loss_f(preds, y)
                
                if (i + 1) % args.print_every == 0:
                    print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('\nTraining %d epochs takes time %.4f minutes' % (num_epochs, time.time() - start_time / 60.0))

            print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

            test(model, loader_test, dtype)
        
        learning_rate *= 0.1

    return model

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='size of each batch of cifar-10 training images')
    parser.add_argument('--print-every', default=50, type=int,
                        help='number of iterations to wait before printing')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)


