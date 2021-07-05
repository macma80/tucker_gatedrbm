import numpy as np
import pylab
import torch
import torch.optim as optim
import torch.utils.data as data_utils
# from gbmPYTORCH import FACTOREDGBMBINBIN
# from gbmFULL import UNFACTOREDGBMBINBIN
from gbmTUCKER import TUCKERGBMBINBIN
from sklearn.model_selection import ParameterGrid
from torch.autograd import Variable

########## CONFIGURATION ##########
BATCH_SIZE = 32
INPUT_UNITS = 13 * 13  # 13 x 13 images
OUTPUT_UNITS = 13 * 13  # shiftimages
# OUTPUT_UNITS = 18*18 #rotateimages Uncomment line 61 to use rotate images
# OUTPUT_UNITS = 20*20 #zoomimages Uncomment line 62 to use zoom images
HIDDEN_UNITS = 100

'''
#FACTOREDGBM PARAMETERS
ALPHA=1e-3
RANKS = [120,120,120]
CD_K = 1
L_RATE = 1e-3
EPOCHS = 1
MOMENTUM = 0.9
WEIGHT_DECAY=0
ACTFUN='sigmoid'
LOG_INTERVAL = 10
'''
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
# outputimages = torch.from_numpy(np.loadtxt('shiftimages.txt')).float()
# outputimages = torch.from_numpy(np.loadtxt('rotateimages.txt')).float()
# outputimages = torch.from_numpy(np.loadtxt('zoomimages.txt')).float()

train_tensor = data_utils.TensorDataset(inputimages, outputimages)
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)

########## TRAINING GBM ##########
param_grid = {'hidden': [8 * 8, 10 * 10, 12 * 12], 'lrate': [1e-2, 1e-3, 1e-4]}
param_grid = {'hidden': [10 * 10], 'lrate': [1e-2]}
param_grid = {'hidden': [10 * 10], 'lrate': [1e-2], 'ranks': [[80, 80, 80], [120, 120, 120], [169, 169, 169]]}
grid = ParameterGrid(param_grid)
list_loss_ = []


def dispims(M, height, width, border=0, bordercolor=0.0, **kwargs):
    """ Display the columns of matrix M in a montage. """
    numimages = M.shape[1]
    n0 = np.int(np.ceil(np.sqrt(numimages)))
    n1 = np.int(np.ceil(np.sqrt(numimages)))
    im = bordercolor * \
         np.ones(((height + border) * n1 + border, (width + border) * n0 + border), dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i * n1 + j < M.shape[1]:
                im[j * (height + border) + border:(j + 1) * (height + border) + border, \
                i * (width + border) + border:(i + 1) * (width + border) + border] = \
                    np.vstack(( \
                        np.hstack((np.reshape(M[:, i * n1 + j], (width, height)).T, \
                                   bordercolor * np.ones((height, border), dtype=float))), \
                        bordercolor * np.ones((border, width + border), dtype=float) \
                        ))
    pylab.imshow(im.T, cmap=pylab.cm.gray, interpolation='None', **kwargs)

def train(EPOCHS):
    # Training loop
    sublist_loss_ = []

    for batch_idx, (input, output) in enumerate(train_loader):
        # if batch_idx > 10:
        #     continue

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
    ######### VISUALIZE THE FILTERS ##########
    ##  visualize the filters FACTOREDGBMBINBIN:
    pylab.clf()
    pylab.subplot(1, 3, 1)
    pylab.title('X Filters')
    pylab.ylabel('Filters After Training')
    dispims(gbm.wxf.data.cpu().numpy(), int(np.sqrt(INPUT_UNITS)), int(np.sqrt(INPUT_UNITS)), 2)
    pylab.axis('off')
    pylab.subplot(1, 3, 2)
    pylab.title('Y Filters')
    dispims(gbm.wyf.data.cpu().numpy(), int(np.sqrt(OUTPUT_UNITS)), int(np.sqrt(OUTPUT_UNITS)), 2)
    pylab.axis('off')
    pylab.subplot(1, 3, 3)
    pylab.title('H Filters')
    dispims(gbm.whf.data.cpu().numpy(), int(np.sqrt(HIDDEN_UNITS)), int(np.sqrt(HIDDEN_UNITS)), 2)
    pylab.axis('off')
    # pylab.subplot(1, 1, 1)
    # pylab.title('WXF mult Filters')
    # dispims(
    #     torch.matmul(gbm.wxf, gbm.core).data.cpu().numpy().reshape(INPUT_UNITS, RANKS[1]*RANKS[2]),
    #     int(np.sqrt(INPUT_UNITS)), int(np.sqrt(INPUT_UNITS)), 2)
    # pylab.axis('off')
    # pylab.subplot(1, 1, 1)
    # pylab.title('Core Filters')
    # dispims(
    #     gbm.core.reshape(RANKS[0]*RANKS[2], RANKS[1]).data.cpu().numpy(),
    #     RANKS[0] , RANKS[2], 2)
    # pylab.axis('off')
    # pylab.show()
    dir_path = "images/" + "HUNITS" + str(HIDDEN_UNITS) + "LRATE" + str(L_RATE).replace(".", "") + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    name = dir_path + "BATCH" + str(batch_idx)
    pylab.savefig("%s_model.png" % name, facecolor='w', edgecolor='w')


for params in grid:
    HIDDEN_UNITS = params['hidden']
    L_RATE = params['lrate']
    RANKS = params['ranks']

    print('Training GBM...')

    # gbm = FACTOREDGBMBINBIN(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RANKS, BATCH_SIZE, cditerations=CD_K,use_cuda=CUDA, alpha=ALPHA, actfun=ACTFUN)
    # gbm = UNFACTOREDGBMBINBIN(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, BATCH_SIZE, cditerations=CD_K,use_cuda=CUDA)
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
pylab.xlabel("Batch number")
pylab.xlabel("Loss (Energy)")
pylab.legend(loc='upper left')
pylab.title("Training vs Test Loss")
pylab.show()
