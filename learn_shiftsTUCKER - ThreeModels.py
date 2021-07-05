import numpy as np
import pylab
import torch
import time
import torch.optim as optim
import torch.utils.data as data_utils
from gbmTUCKER import TUCKERGBMBINBIN
from sklearn.model_selection import ParameterGrid
from torch.autograd import Variable

########## CONFIGURATION ##########
BATCH_SIZE = 32
INPUT_UNITS = 13 * 13  # 13 x 13 images
OUTPUT_UNITS = 13 * 13  # shiftimages
HIDDEN_UNITS = 100

ALPHA = 1e-3
CD_K = 1
L_RATE = 1e-5
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
inputimages = torch.from_numpy(np.loadtxt('inputimages.txt')).float()
outputimages = torch.from_numpy(np.loadtxt('outputimages.txt')).float()

train_tensor = data_utils.TensorDataset(inputimages, outputimages)
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)

########## TRAINING GBM ##########

param_grid = {'hidden': [10 * 10], 'lrate': [1e-2], 'ranks': [[80, 80, 80], [120, 120, 120], [169, 169, 169]]}
grid = ParameterGrid(param_grid)
list_loss_ = []
list_time_ = []


def train(EPOCHS):
    # Training loop
    sublist_loss_ = []
    sublist_time_ = []
    time_0 = time.time()
    for batch_idx, (input, output) in enumerate(train_loader):
        if batch_idx > 110:
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
        time_1 = time.time()
        total_time = time_1 - time_0
        sublist_time_.append(total_time)
        if batch_idx % LOG_INTERVAL == 0:
            # vis.plot_loss(np.mean(loss_),batch_idx)
            # loss_.clear()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                EPOCHS, batch_idx * len(datain), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

    list_loss_.append(sublist_loss_)
    list_time_.append(sublist_time_)

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

pylab.figure()
for index, loss in enumerate(list_loss_):
    pylab.plot(list_loss_[index], label=str(param_grid['ranks'][index]))

pylab.legend()
pylab.xlabel("Batch Number")
pylab.ylabel("Loss (Energy)")
pylab.legend(title='Core tensor ranks')
#pylab.title("Energy loss comparison for different configurations")
#pylab.show()
pylab.savefig("losscomparison.png")

pylab.figure()
for index, loss in enumerate(list_time_):
    pylab.plot(list_time_[index], label=str(param_grid['ranks'][index]))

pylab.legend()
pylab.xlabel("Batch Number")
pylab.ylabel("Training Time (sec)")
pylab.legend(loc='upper left', title='Core tensor ranks')
#pylab.title("Training Time for different configurations")
#pylab.show()
pylab.savefig("ttimecomparison.png")
