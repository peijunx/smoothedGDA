# A smoothed GDA algorithm for the nonconvex-concave min-max problem with an $\mathcal{O}\left(1/\epsilon^2\right)$ iteration complexity

This repository is the official implementation of [A smoothed GDA algorithm for the nonconvex-concave min-max problem with an $\mathcal{O}\left(1/\epsilon^2\right)$ iteration complexity]

## Requirement

This code requires standard packages likes PyTorch and numpy as specified in at the beginning of the code.

## Training and Evaluation

We include two files:

* nouiehed19.py is for training and evalution the algorithms in the paper [Solving a class of non- convex min-max games using iterative first order methods]
* smoothed_GDA.py is for or training and evalution the algorithms in our paper  [A smoothed GDA algorithm for the nonconvex-concave min-max problem with an $\mathcal{O}\left(1/\epsilon^2\right)$ iteration complexity]

You can directly run these two python files to train the models. The parameters are specified in each file and you can change them correspondingly in the file.

The average runtime for each training with evaluation (100 epochs) is 8-9 hours on a NVIDIA GeForce GTX1080 GPU. We recommend using the parameters $p, \beta$ in the range of $[0.5, 1]$ and $\alpha = 1$ in running Smoothed-GDA algorithm. We run the experiments with 3 different seeds and the maximal variance of the loss values at each iteration for the algorithm in [Solving a class of non- convex min-max games using iterative first order methods] is 1.22 and is 0.55 in our algorithm. 

