import numpy as np
import os
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import pylab
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from gbmTUCKER import TUCKERGBMBINBIN

########## CONFIGURATION ##########
BATCH_SIZE = 32
INPUT_UNITS = 13 * 13  # 13 x 13 images
OUTPUT_UNITS = 13 * 13  # shiftimages

ALPHA = 1e-3
HIDDEN_UNITS = 11 * 11
RANKS = [120, 120, 120]
CD_K = 1
EPOCHS = 1
MOMENTUM = 0.85
WEIGHT_DECAY = 0.9
ACTFUN = 'sigmoid'
LOG_INTERVAL = 1

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

########## LOADING DATASET ##########
print('Loading dataset...')
#inputimages = torch.from_numpy(np.loadtxt('inputimages.txt')).float()
#outputimages = torch.from_numpy(np.loadtxt('outputimages.txt')).float()
inputimages = np.loadtxt('inputimages.txt')
outputimages = np.loadtxt('outputimages.txt')

inputimages_train, inputimages_test, outputimages_train, outputimages_test = train_test_split(inputimages, outputimages, test_size=0.2, random_state=42)

inputimages_train = torch.from_numpy(inputimages_train).float()
outputimages_train = torch.from_numpy(outputimages_train).float()
inputimages_test = torch.from_numpy(inputimages_test).float()
outputimages_test = torch.from_numpy(outputimages_test).float()

train_tensor = data_utils.TensorDataset(inputimages_train, outputimages_train)
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)

test_tensor = data_utils.TensorDataset(inputimages_test, outputimages_test)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=True)

########## TRAINING GBM ##########
#param_grid = {'hidden': [8 * 8, 10 * 10, 12 * 12], 'lrate': [1e-2, 1e-3, 1e-4]}
param_grid = {'hidden': [11 * 11], 'lrate': [1e-2], 'ranks': [[120, 120, 120]]}
#param_grid = {'hidden': [10 * 10], 'lrate': [1e-2], 'ranks':  [[80, 80, 80], [120, 120, 120], [169, 169, 169]]}
grid = ParameterGrid(param_grid)
list_loss_ = []
labels_loss = ["Train", "Test"]

def train(EPOCHS):
    # Training loop
    sublist_loss_ = []
    for batch_idx, (input, output) in enumerate(train_loader):

        if batch_idx > 120:
            continue

        if len(input) != BATCH_SIZE:
            continue

        datain = Variable(input)
        dataout = Variable(output)
        if CUDA:
            datain, dataout = datain.cuda(), dataout.cuda()

        xprobs, yprobs = gbm(datain, dataout)  # Forward pass - Contrastive divergence
        loss = - (gbm.energy_posgrad(datain, dataout) - gbm.energy_neggrad())
        sublist_loss_.append(loss.item())
        train_op.zero_grad()
        loss.backward()
        train_op.step()
        if batch_idx % LOG_INTERVAL == 0:
            # vis.plot_loss(np.mean(loss_),batch_idx)
            # loss_.clear()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                EPOCHS, batch_idx * len(datain), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

    list_loss_.append(sublist_loss_)

def test(EPOCHS):
    # Testing loop
    sublist_loss_ = []
    for batch_idx, (input, output) in enumerate(test_loader):

        if len(input) != BATCH_SIZE:
            continue

        datain = Variable(input)
        dataout = Variable(output)
        if CUDA:
            datain, dataout = datain.cuda(), dataout.cuda()

        xprobs, yprobs = gbm(datain, dataout)  # Forward pass - Contrastive divergence
        loss = - (gbm.energy_posgrad(datain, dataout) - gbm.energy_neggrad())
        sublist_loss_.append(loss.item())

    list_loss_.append(sublist_loss_)


for params in grid:
    HIDDEN_UNITS = params['hidden']
    L_RATE = params['lrate']
    RANKS = params['ranks']

    print('Training GBM...')

    gbm = TUCKERGBMBINBIN(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RANKS, BATCH_SIZE, cditerations=CD_K, use_cuda=CUDA,
                          alpha=ALPHA, actfun=ACTFUN)

    if CUDA:
        gbm.cuda()

    # train_op = optim.Adamax(gbm.parameters(),lr=L_RATE)
    # train_op = optim.Adagrad(gbm.parameters(),lr=L_RATE)
    # train_op = optim.SGD(gbm.parameters(), lr=L_RATE, momentum=MOMENTUM)
    # train_op = optim.Rprop(gbm.parameters(), lr=L_RATE)
    # train_op = optim.Adam(gbm.parameters(), lr=L_RATE)
    train_op = optim.SGD(gbm.parameters(), lr=L_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
    for epoch in range(1, EPOCHS + 1):
        test(epoch)
pylab.figure()
for index, loss in enumerate(list_loss_):
    pylab.plot(list_loss_[index], label=labels_loss[index] )

pylab.legend()
pylab.xlabel("Batch number")
pylab.ylabel("Loss (Energy)")
pylab.legend(loc='upper right')
#pylab.title("Training vs Test Loss")
#pylab.show()
pylab.savefig("losstrainingtest.png")