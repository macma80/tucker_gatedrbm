import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TUCKERGBM(nn.Module):
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
        super(TUCKERGBM, self).__init__()


        #self.core, self.factors = tlrm.tucker_tensor((numin, numout, nummap), ranks)

        self.actfun = actfun
        rng = torch.Generator()
        #rng.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)

        self.core = nn.Parameter(torch.randn(ranks, generator=rng)*alpha)
        self.wxf = nn.Parameter(torch.randn(numin, ranks[0],generator=rng)*alpha)
        self.wyf = nn.Parameter(torch.randn(numout, ranks[1],generator=rng)*alpha)
        self.whf = nn.Parameter(torch.randn(nummap, ranks[2],generator=rng)*alpha)

        self.ib = nn.Parameter(torch.ones(numin) * 0.5)
        self.ob = nn.Parameter(torch.zeros(numout))
        self.hb = nn.Parameter(torch.zeros(nummap))
        self.batchsize = batchsize
        self.batchsize2 = 64 #Hardcoded to 64 because I want to see that the tensor matrix multiplications are done correctly
        self.cditerations = cditerations
        self.use_cuda = use_cuda

        self.core_inc = torch.zeros(self.core.shape)
        self.wxf_inc = torch.zeros(self.wxf.shape)
        self.wyf_inc = torch.zeros(self.wyf.shape)
        self.whf_inc = torch.zeros(self.whf.shape)
        self.ib_inc = torch.zeros(self.ib.shape)
        self.ob_inc = torch.zeros(self.ob.shape)
        self.hb_inc = torch.zeros(self.hb.shape)

        #GIBBS CACHE
        self.x_probs = torch.zeros(numin,self.batchsize2, self.batchsize)
        self.y_probs = torch.zeros(self.batchsize, numout, self.batchsize)
        self.h_probs = torch.zeros(self.batchsize, self.batchsize2, nummap)
        self.xstates = torch.zeros(self.batchsize,numin)
        self.ystates = torch.zeros(self.batchsize2,numout)
        self.hstates = torch.zeros(self.batchsize,nummap)
        self.actsx = torch.zeros(self.batchsize,numin)
        self.actsy = torch.zeros(self.batchsize2,numout)
        self.actsh = torch.zeros(self.batchsize,nummap)
        self.corexy = torch.zeros(self.batchsize,self.batchsize2, ranks[2])
        self.corexh = torch.zeros(self.batchsize,ranks[1],self.batchsize)
        self.coreyh = torch.zeros(ranks[0],self.batchsize2,self.batchsize)

        if self.use_cuda:
            self.core_inc = self.core_inc.cuda()
            self.wxf_inc = self.wxf_inc.cuda()
            self.wyf_inc = self.wyf_inc.cuda()
            self.whf_inc = self.whf_inc.cuda()
            self.ib_inc = self.ib_inc.cuda()
            self.ob_inc = self.ob_inc.cuda()
            self.hb_inc = self.hb_inc.cuda()

            self.x_probs = self.x_probs.cuda()
            self.y_probs = self.y_probs.cuda()
            self.h_probs = self.h_probs.cuda()
            self.xstates = self.xstates.cuda()
            self.ystates = self.ystates.cuda()
            self.hstates = self.hstates.cuda()
            self.actsx = self.actsx.cuda()
            self.actsy = self.actsy.cuda()
            self.actsh = self.actsh.cuda()
            self.corexy = self.corexy.cuda()
            self.corexh = self.corexh.cuda()
            self.coreyh = self.coreyh.cuda()

    def forward(self, inputs, outputs):
        self.actsx = torch.matmul(inputs,self.wxf)
        self.actsy = torch.matmul(outputs,self.wyf)
        pre_h1, h1 = self.hidprobs(inputs, outputs)

        h_ = h1
        for _ in range(self.cditerations):
            pre_o_, o_ = self.outprobs(inputs, h_)
            pre_h_, h_ = self.hidprobs(inputs, o_)

        return self.x_probs, self.y_probs

    def energy_posgrad(self, inputs, outputs):
        self.actsx = torch.matmul(inputs, self.wxf)
        self.actsy = torch.matmul(outputs, self.wyf)
        self.actsh = torch.matmul(self.hstates, self.whf)
        self.corexy = torch.matmul(self.actsx,torch.matmul(self.actsy,self.core).transpose(0,1))
        self.corexh = torch.matmul(self.actsx,torch.matmul(self.actsh,self.core.transpose(1,2)).transpose(0,1)).transpose(1,0).transpose(2,1)
        self.coreyh = torch.matmul(self.actsh,torch.matmul(self.actsy,self.core).transpose(1,2))
        self.core_inc = torch.matmul(self.whf.t(),torch.matmul(self.actsx.t(),torch.matmul(self.actsy.t(),self.h_probs).transpose(0,1)).transpose(1,0).transpose(1,2)).transpose(2,1)
        self.wxf_inc = torch.matmul(self.x_probs.reshape(self.x_probs.shape[0],-1),self.coreyh.reshape(-1,self.coreyh.shape[0]))
        self.wyf_inc = torch.matmul(self.y_probs.reshape(self.y_probs.shape[1],-1),self.corexh.reshape(-1,self.corexh.shape[1]))
        self.whf_inc = torch.matmul(self.h_probs.reshape(self.h_probs.shape[2],-1),self.corexy.reshape(-1,self.corexy.shape[2]))
        self.ib_inc = torch.sum(inputs,0)
        self.ob_inc = torch.sum(outputs,0)
        self.hb_inc  = torch.sum(self.h_probs,0)

        grads = torch.cat((self.core_inc.flatten(),self.wxf_inc.flatten(),self.wyf_inc.flatten(),self.whf_inc.flatten(),
                           self.ib_inc.flatten(),self.ob_inc.flatten(),self.hb_inc.flatten()))

        return (grads).mean()

    def energy_neggrad(self):
        self.actsx = torch.matmul(self.xstates, self.wxf)
        self.actsy = torch.matmul(self.ystates, self.wyf)
        self.actsh = torch.matmul(self.hstates, self.whf)
        self.corexy = torch.matmul(self.actsx,torch.matmul(self.actsy,self.core).transpose(0,1))
        self.corexh = torch.matmul(self.actsx,torch.matmul(self.actsh,self.core.transpose(1,2)).transpose(0,1)).transpose(1,0).transpose(2,1)
        self.coreyh = torch.matmul(self.actsh,torch.matmul(self.actsy,self.core).transpose(1,2))
        self.core_inc = torch.matmul(self.whf.t(),torch.matmul(self.actsx.t(),torch.matmul(self.actsy.t(),self.h_probs).transpose(0,1)).transpose(1,0).transpose(1,2)).transpose(2,1)
        self.wxf_inc = torch.matmul(self.x_probs.reshape(self.x_probs.shape[0],-1),self.coreyh.reshape(-1,self.coreyh.shape[0]))
        self.wyf_inc = torch.matmul(self.y_probs.reshape(self.y_probs.shape[1],-1),self.corexh.reshape(-1,self.corexh.shape[1]))
        self.whf_inc = torch.matmul(self.h_probs.reshape(self.h_probs.shape[2],-1),self.corexy.reshape(-1,self.corexy.shape[2]))
        self.ib_inc = torch.sum(self.x_probs,0)
        self.ob_inc = torch.sum(self.y_probs,0)
        self.hb_inc  = torch.sum(self.h_probs,0)

        grads = torch.cat((self.core_inc.flatten(), self.wxf_inc.flatten(), self.wyf_inc.flatten(), self.whf_inc.flatten(),
                           self.ib_inc.flatten(), self.ob_inc.flatten(), self.hb_inc.flatten()))

        return (grads).mean()


class TUCKERGBMBINBIN(TUCKERGBM):

    def hidprobs(self, inputs, outputs):
        self.corexy = torch.matmul(self.whf,torch.matmul(self.actsx,torch.matmul(self.actsy,self.core).transpose(0,1)).transpose(1,2)).transpose(2,1).transpose(1,0)
        if self.actfun=='sigmoid':
            self.h_probs = F.sigmoid(self.corexy+self.hb)
        elif self.actfun=='relu':
            self.h_probs = F.relu(self.corexy + self.hb)
        self.hstates = self.sample_from_p(self.h_probs.mean(1))

        return self.h_probs, self.hstates


    def outprobs(self, inputs, hiddens):
        self.actsx = torch.matmul(inputs,self.wxf)
        self.actsh = torch.matmul(hiddens,self.whf)
        self.corexh = torch.matmul(self.wyf,torch.matmul(self.actsx,torch.matmul(self.actsh,self.core.transpose(1,2)).transpose(0,1)).transpose(1,0).transpose(2,1))
        if self.actfun=='sigmoid':
            self.y_probs = F.sigmoid((self.corexh.transpose(1,2) + self.ob).transpose(2,1))
        elif self.actfun=='relu':
            self.y_probs = F.relu((self.corexh.transpose(1, 2) + self.ob).transpose(2, 1))
        self.ystates = self.sample_from_p(self.y_probs.mean(2))

        self.actsy = torch.matmul(self.ystates,self.wyf)
        self.coreyh = torch.matmul(self.wxf,torch.matmul(self.actsy,torch.matmul(self.actsh,self.core.transpose(1,2)).transpose(2,1)).transpose(0,1)).transpose(1,0)
        if self.actfun=='sigmoid':
            self.x_probs = F.sigmoid(self.coreyh.transpose(0,2)+self.ib).transpose(2,0)
        elif self.actfun=='relu':
            self.x_probs = F.relu(self.coreyh.transpose(0, 2) + self.ib).transpose(2, 0)
        self.xstates = self.sample_from_p(self.x_probs.mean(1)).transpose(0,1)
        self.actsx = torch.matmul(self.xstates,self.wxf)
        return self.y_probs, self.ystates

    def sample_from_p(self, p):
        random_prob = Variable(torch.rand(p.size()))
        if self.use_cuda:
            random_prob = random_prob.cuda()
        return F.relu(torch.sign(p - random_prob))

    def get_hidprobs(self, inputs, outputs):
        self.actsx = torch.matmul(inputs, self.wxf)
        self.actsy = torch.matmul(outputs, self.wyf)
        #self.actsy = torch.matmul(self.y_probs, self.wyf)
        #self.actsy = torch.matmul(self.y_probs.mean(2), self.wyf)
        h_probs, h_sample = self.hidprobs(inputs, outputs)
        return h_probs

