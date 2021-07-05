import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FACTOREDGBM(nn.Module):
    def __init__(self,
                 numin,
                 numout,
                 nummap,
                 ranks,
                 batchsize,
                 cditerations=1,
                 use_cuda=False,
                 alpha=1e-3,
                 actfun='sigmoid'):
        super(FACTOREDGBM, self).__init__()

        self.actfun = actfun

        self.wxf = nn.Parameter(torch.randn(numin,ranks[0])*alpha)
        self.wyf = nn.Parameter(torch.randn(numout,ranks[1])*alpha)
        self.whf = nn.Parameter(torch.randn(nummap,ranks[2])*alpha)
        self.ib = nn.Parameter(torch.ones(numin) * 0.5)
        self.ob = nn.Parameter(torch.zeros(numout))
        self.hb = nn.Parameter(torch.zeros(nummap))
        self.batchsize = batchsize
        self.cditerations = cditerations
        self.use_cuda = use_cuda

        self.wxf_inc = torch.zeros(numin, ranks[0])
        self.wyf_inc = torch.zeros(numout, ranks[1])
        self.whf_inc = torch.zeros(nummap, ranks[2])
        self.ib_inc = torch.zeros(numin)
        self.ob_inc = torch.zeros(numout)
        self.hb_inc = torch.zeros(nummap)

        #GIBBS CACHE
        self.x_probs = torch.zeros(self.batchsize, numin)
        self.y_probs = torch.zeros(self.batchsize, numout)
        self.h_probs = torch.zeros(self.batchsize, nummap)
        self.actsx = torch.zeros(self.batchsize, ranks[0])
        self.actsy = torch.zeros(self.batchsize, ranks[1])
        self.actsh = torch.zeros(self.batchsize, ranks[2])
        self.actsxy = torch.zeros(self.batchsize, ranks[0])
        self.actsxh = torch.zeros(self.batchsize, ranks[1])
        self.actsyh = torch.zeros(self.batchsize, ranks[2])

        if self.use_cuda:
            self.wxf_inc = self.wxf_inc.cuda()
            self.wyf_inc = self.wyf_inc.cuda()
            self.whf_inc = self.whf_inc.cuda()
            self.ib_inc = self.ib_inc.cuda()
            self.ob_inc = self.ob_inc.cuda()
            self.hb_inc = self.hb_inc.cuda()

            self.x_probs = self.x_probs.cuda()
            self.y_probs = self.y_probs.cuda()
            self.h_probs = self.h_probs.cuda()
            self.actsx = self.actsx.cuda()
            self.actsy = self.actsy.cuda()
            self.actsh = self.actsh.cuda()
            self.actsxy = self.actsxy.cuda()
            self.actsxh = self.actsxh.cuda()
            self.actsyh = self.actsyh.cuda()

    def forward(self, inputs, outputs):
        self.actsx = torch.matmul(inputs, self.wxf)
        self.actsy = torch.matmul(outputs, self.wyf)
        self.hidprobs()

        for _ in range(self.cditerations):
            self.outprobs()
            self.hidprobs()

        return self.x_probs, self.y_probs

    def energy_posgrad(self, inputs, outputs):

        self.wxf_inc = torch.matmul(inputs.t(), self.actsyh)
        self.wyf_inc = torch.matmul(outputs.t(), self.actsxh)
        self.whf_inc = torch.matmul(self.h_probs.t(), self.actsxy)
        self.ib_inc = torch.sum(inputs,0)
        self.ob_inc = torch.sum(outputs,0)
        self.hb_inc  = torch.sum(self.h_probs,0)

        grads = torch.cat((self.wxf_inc.view(-1), self.wyf_inc.view(-1),
                           self.whf_inc.view(-1), self.ib_inc.view(-1), self.ob_inc.view(-1), self.hb_inc.view(-1)), 0)

        return (grads).mean()

    def energy_neggrad(self):

        self.wxf_inc = torch.matmul(self.x_probs.t(), self.actsyh)
        self.wyf_inc = torch.matmul(self.y_probs.t(), self.actsxh)
        self.whf_inc = torch.matmul(self.h_probs.t(), self.actsxy)
        self.ib_inc = torch.sum(self.x_probs,0)
        self.ob_inc = torch.sum(self.y_probs,0)
        self.hb_inc  = torch.sum(self.h_probs,0)

        grads = torch.cat((self.wxf_inc.view(-1), self.wyf_inc.view(-1),
                           self.whf_inc.view(-1), self.ib_inc.view(-1), self.ob_inc.view(-1), self.hb_inc.view(-1)), 0)

        return (grads).mean()


class FACTOREDGBMBINBIN(FACTOREDGBM):

    def hidprobs(self):
        self.actsxy = torch.mul(self.actsx, self.actsy)
        if self.actfun == 'sigmoid':
            self.h_probs = F.sigmoid(torch.matmul(self.actsxy, self.whf.t()) + self.hb)
        elif self.actfun == 'relu':
            self.h_probs = F.relu(torch.matmul(self.actsxy, self.whf.t()) + self.hb)
        sample_h = self.sample_from_p(self.h_probs)

        self.actsh = torch.matmul(self.h_probs, self.whf)
        self.actsxh = torch.mul(self.actsx, self.actsh)
        self.actsyh = torch.mul(self.actsy, self.actsh)
        return self.h_probs, sample_h


    def outprobs(self):
        self.actsxh = torch.mul(self.actsh, self.actsx)
        if self.actfun == 'sigmoid':
            self.y_probs = F.sigmoid(torch.matmul(self.actsxh, self.wyf.t()) + self.ob)
        elif self.actfun == 'relu':
            self.y_probs = F.relu(torch.matmul(self.actsxh, self.wyf.t()) + self.ob)
        sample_y = self.sample_from_p(self.y_probs)

        self.actsy = torch.matmul(self.y_probs, self.wyf)
        self.actsyh = torch.mul(self.actsh, self.actsy)
        if self.actfun == 'sigmoid':
            self.x_probs = F.sigmoid(torch.matmul(self.actsyh, self.wxf.t()) + self.ib)
        elif self.actfun == 'relu':
            self.x_probs = F.relu(torch.matmul(self.actsyh, self.wxf.t()) + self.ib)
        self.actsx = torch.matmul(self.x_probs,self.wxf)
        return self.y_probs, sample_y

    def sample_from_p(self, p):
        random_prob = Variable(torch.rand(p.size()))
        if self.use_cuda:
            random_prob = random_prob.cuda()
        return F.relu(torch.sign(p - random_prob))

    def get_hidprobs(self, inputs):
        self.actsx = torch.matmul(inputs, self.wxf)
        #self.actsy = torch.matmul(outputs, self.wyf)
        self.actsy = torch.matmul(self.y_probs, self.wyf)
        h_probs, h_sample = self.hidprobs()
        return h_probs

