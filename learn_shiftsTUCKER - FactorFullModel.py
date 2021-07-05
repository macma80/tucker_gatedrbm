import numpy as np
import pylab
import torch
import time
import torch.optim as optim
import torch.utils.data as data_utils
# from gbmPYTORCH import FACTOREDGBMBINBIN
from gbmFULL import UNFACTOREDGBMBINBIN
from gbmTUCKER import TUCKERGBMBINBIN
from sklearn.model_selection import ParameterGrid
from torch.autograd import Variable

########## CONFIGURATION ##########
BATCH_SIZE = 16
INPUT_UNITS = 13 * 13  # 13 x 13 images
OUTPUT_UNITS = 13 * 13  # shiftimages
HIDDEN_UNITS = 100

# TUCKERGBM PARAMETERS
ALPHA = 1e-3
HIDDEN_UNITS = 11 * 11
RANKS = [120, 120, 120]
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
param_grid = {'hidden': [11 * 11], 'lrate': [1e-3], 'ranks': [[80, 80, 80]]}
grid = ParameterGrid(param_grid)
list_loss_ = []
list_time_ = []
labels_loss = ["Factored Model", "Unfactored Model"]

def train(EPOCHS):
    # Training loop
    sublist_loss_ = []
    sublist_time_ = []
    time_0 = time.time()
    for batch_idx, (input, output) in enumerate(train_loader):
        if batch_idx > 80:
            continue

        if len(input) != BATCH_SIZE:
            continue

        datain = Variable(input)
        dataout = Variable(output)
        if CUDA:
            datain, dataout = datain.cuda(), dataout.cuda()

        xprobs, yprobs = gbm_factored(datain, dataout)  # Forward pass - Contrastive divergence
        loss = - (gbm_factored.energy_posgrad(datain, dataout) - gbm_factored.energy_neggrad())
        sublist_loss_.append(loss.item())
        factored_train_op.zero_grad()
        loss.backward()
        factored_train_op.step()
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

    sublist_loss_ = []
    sublist_time_ = []
    time_0 = time.time()
    for batch_idx, (input, output) in enumerate(train_loader):
        # if batch_idx > 200:
        #     continue

        if len(input) != BATCH_SIZE:
            continue

        datain = Variable(input)
        dataout = Variable(output)
        if CUDA:
            datain, dataout = datain.cuda(), dataout.cuda()

        xprobs, yprobs = gbm_unfactored(datain, dataout)  # Forward pass - Contrastive divergence
        loss = - (gbm_unfactored.energy_posgrad(datain, dataout) - gbm_unfactored.energy_neggrad())
        sublist_loss_.append(loss.item())
        unfactored_train_op.zero_grad()
        loss.backward()
        unfactored_train_op.step()
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
    print("Stop")

for params in grid:
    HIDDEN_UNITS = params['hidden']
    L_RATE = params['lrate']
    RANKS = params['ranks']

    print('Training GBM...')

    # gbm = FACTOREDGBMBINBIN(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RANKS, BATCH_SIZE, cditerations=CD_K,use_cuda=CUDA, alpha=ALPHA, actfun=ACTFUN)
    gbm_unfactored = UNFACTOREDGBMBINBIN(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, BATCH_SIZE, cditerations=CD_K, use_cuda=CUDA)
    gbm_factored = TUCKERGBMBINBIN(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RANKS, BATCH_SIZE, cditerations=CD_K, use_cuda=CUDA,
                          alpha=ALPHA, actfun=ACTFUN)

    if CUDA:
        gbm_factored.cuda()

    # train_op = optim.Adamax(gbm.parameters(),lr=L_RATE)
    # train_op = optim.Adagrad(gbm.parameters(),lr=L_RATE)
    # train_op = optim.SGD(gbm.parameters(), lr=L_RATE, momentum=MOMENTUM)
    # train_op = optim.Rprop(gbm.parameters(), lr=L_RATE)
    # train_op = optim.Adam(gbm.parameters(), lr=L_RATE)
    unfactored_train_op = optim.SGD(gbm_unfactored.parameters(), lr=L_RATE)
    factored_train_op = optim.SGD(gbm_factored.parameters(), lr=L_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


    for epoch in range(1, EPOCHS + 1):
        train(epoch)

pylab.figure()
for index, loss in enumerate(list_loss_):
    pylab.plot(list_loss_[index], label=labels_loss[index])

pylab.legend()
pylab.xlabel("Batch number")
pylab.ylabel("Loss (Energy)")
pylab.legend()
#pylab.title("Loss in Factored and Unfactored Models")
#pylab.show()
pylab.savefig("lossfactoredfull.png")

pylab.figure()
for index, loss in enumerate(list_time_):
    pylab.plot(list_time_[index], label=labels_loss[index])

pylab.legend()
pylab.xlabel("Batch Number")
pylab.ylabel("Training Time (sec)")
pylab.legend(loc='upper left')
#pylab.title("Training Time in Factored and Unfactored Models")
#pylab.show()
pylab.savefig("ttimefactoredfull.png")