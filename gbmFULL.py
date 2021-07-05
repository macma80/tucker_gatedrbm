import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class UNFACTOREDGBM(nn.Module):
    def __init__(self,
                 numin,
                 numout,
                 nummap,
                 batchsize,
                 cditerations=1,
                 use_cuda=False):
        super(UNFACTOREDGBM, self).__init__()

        alpha=1e-3
        #alpha=1e-6
        self.weights = nn.Parameter(torch.randn(numin,numout,nummap)*alpha)
        self.ib = nn.Parameter(torch.ones(numin) * 0.5)
        self.ob = nn.Parameter(torch.zeros(numout))
        self.hb = nn.Parameter(torch.zeros(nummap))
        self.batchsize = batchsize
        self.batchsize2 = 16
        self.cditerations = cditerations
        self.use_cuda = use_cuda

        self.weights_inc = torch.zeros(self.weights.shape)
        self.ib_inc = torch.zeros(self.ib.shape)
        self.ob_inc = torch.zeros(self.ob.shape)
        self.hb_inc = torch.zeros(self.hb.shape)

        #GIBBS CACHE
        #self.x_probs = torch.zeros(numin,self.batchsize2, self.batchsize)
        #self.y_probs = torch.zeros(self.batchsize, numout, self.batchsize)
        #self.h_probs = torch.zeros(self.batchsize, self.batchsize2, nummap)
        self.xstates = torch.zeros(self.batchsize,numin)
        self.ystates = torch.zeros(self.batchsize2,numout)
        self.hstates = torch.zeros(self.batchsize,nummap)
        self.actsxy = torch.zeros(self.batchsize,self.batchsize2, nummap)
        self.actsxh = torch.zeros(self.batchsize,numout,self.batchsize)
        self.actsyh = torch.zeros(numin,self.batchsize2,self.batchsize)

        if self.use_cuda:
            #self.wxf_inc = self.wxf_inc.cuda()
            #self.wyf_inc = self.wyf_inc.cuda()
            #self.whf_inc = self.whf_inc.cuda()
            self.weights_inc = self.weights_inc.cuda()
            self.ib_inc = self.ib_inc.cuda()
            self.ob_inc = self.ob_inc.cuda()
            self.hb_inc = self.hb_inc.cuda()

            #self.x_probs = self.x_probs.cuda()
            #self.y_probs = self.y_probs.cuda()
            #self.h_probs = self.h_probs.cuda()
            self.xstates = self.xstates.cuda()
            self.ystates = self.ystates.cuda()
            self.hstates = self.hstates.cuda()
            self.actsxy = self.actsxy.cuda()
            self.actsxh = self.actsxh.cuda()
            self.actsyh = self.actsyh.cuda()

    def forward(self, inputs, outputs):
        #self.actsx = torch.matmul(inputs, self.weights.view(169,-1))
        #self.actsy = torch.matmul(outputs, self.weights.view())
        pre_h1, h1 = self.hidprobs(inputs, outputs)

        h_ = h1
        for _ in range(self.cditerations):
            pre_o_, o_ = self.outprobs(inputs, h_)
            pre_h_, h_ = self.hidprobs(inputs, o_)

        return self.x_probs, self.y_probs

    def energy_posgrad(self, inputs, outputs):

        self.weights_inc = torch.matmul(inputs.t(),torch.matmul(outputs.t(),self.h_probs).transpose(0,1)).transpose(1,0)
        self.ib_inc = torch.sum(inputs,0)
        self.ob_inc = torch.sum(outputs,0)
        self.hb_inc  = torch.sum(self.h_probs,0)

        grads = torch.cat((self.weights_inc.flatten(),self.ib_inc.flatten(),self.ob_inc.flatten(),self.hb_inc.flatten()))

        return (grads).mean()

    def energy_neggrad(self):
        #using probs
        #self.weights_inc = torch.matmul(self.x_probs.mean(2),torch.matmul(self.y_probs.mean(0),self.h_probs).transpose(0,1))
        #using states
        self.weights_inc = torch.matmul(self.xstates.t(), torch.matmul(self.ystates.t(), self.h_probs).transpose(0, 1)).transpose(1, 0)

        self.ib_inc = torch.sum(self.x_probs,0)
        self.ob_inc = torch.sum(self.y_probs,0)
        self.hb_inc  = torch.sum(self.h_probs,0)

        grads = torch.cat((self.weights_inc.flatten(),self.ib_inc.flatten(),self.ob_inc.flatten(),self.hb_inc.flatten()))
        return (grads).mean()


class UNFACTOREDGBMBINBIN(UNFACTOREDGBM):

    def hidprobs(self, inputs, outputs):
        self.actsxy = torch.matmul(inputs,torch.matmul(outputs,self.weights).transpose(0,1)).transpose(1,0)
        self.h_probs = self._sigmoid(self.actsxy+self.hb)
        self.hstates = self.sample_from_p(self.h_probs.mean(1))

        return self.h_probs, self.hstates


    def outprobs(self, inputs, hiddens):
        self.actsxh = torch.matmul(inputs,torch.matmul(hiddens, self.weights.transpose(1,2)).transpose(0,1)).transpose(1,0).transpose(2,1)
        self.y_probs = self._sigmoid((self.actsxh.transpose(1,2) + self.ob).transpose(2,1))
        #sample_y = self.sample_from_p(self.y_probs.mean(2)[:60])
        self.ystates = self.sample_from_p(self.y_probs.mean(2))

        self.actsyh = torch.matmul(self.ystates,torch.matmul(hiddens,self.weights.transpose(1,2)).transpose(2,1))
        self.x_probs = self._sigmoid(self.actsyh.transpose(0,2)+self.ib).transpose(2,0)
        self.xstates = self.sample_from_p(self.x_probs.mean(1)).transpose(0,1)
        return self.y_probs, self.ystates

    def sample_from_p(self, p):
        random_prob = Variable(torch.rand(p.size()))
        if self.use_cuda:
            random_prob = random_prob.cuda()
        return F.relu(torch.sign(p - random_prob))

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _logsumexp(self, inputs, dim=None, keepdim=False):
        """Numerically stable logsumexp.

        Args:
            inputs: A Variable with any shape.
            dim: An integer.
            keepdim: A boolean.

        Returns:
            Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
        """
        # For a 1-D array x (any array along a single dimension),
        # log sum exp(x) = s + log sum exp(x - s)
        # with s = max(x) being a common choice.
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = torch.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def free_energy(self, X, Y):
        """
        Compute free energy for the vectors in (corresponding) columns of X and Y. 
        """
        numin, numcases = X.t().shape
        nummap, numfac = self.whf.shape
        factorsX = torch.matmul(X, self.wxf)
        factorsY = torch.matmul(Y, self.wyf)
        factorsXY = torch.mul(factorsX, factorsY)
        F = torch.sum( self._logsumexp( torch.stack(
                   (Variable(torch.cuda.FloatTensor(nummap,numcases).fill_(1)),
                    (torch.matmul(factorsXY, self.whf.t()) + self.hb).t()),2)
                                , 2), 0)
        F += torch.matmul(X, self.ib)
        F += torch.matmul(Y, self.ob)
        return -F.mean()