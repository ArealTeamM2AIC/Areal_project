import torch as th
import torch.nn as nn
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
import sample_code_submission.utils as utils

class ConvModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModel, self).__init__()
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=(5,5), padding=2)
        self.rel_conv1 = nn.ReLU()

        #kernel_pool = (3,3)
        #self.pool = nn.MaxPool2d(kernel_pool, stride=1, padding=1)

    def forward(self, data):
        # data.size() = (1, in_channels, w, h)
        out = self.conv1(data)
        out = self.rel_conv1(out)

        #out = self.pool(out)

        # out.size() = (1,self.out_channels, w, h)
        # squeeze(0) -> (self.out_channels, w, h)
        # permute(1, 2, 0) -> (w, h, self.out_channels)
        out = out.squeeze(0).permute(1, 2, 0)
        return out.contiguous().view(-1, self.out_channels)

class LinModel(nn.Module):
    def __init__(self, out_channels):
        super(LinModel, self).__init__()
        self.out_channels = out_channels

        self.lin1 = nn.Linear(self.out_channels, self.out_channels * 2)
        self.act1 = nn.ReLU()

        self.lin2 = nn.Linear(self.out_channels * 2, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, data):
        out = self.lin1(data)
        out = self.act1(out)
        out = self.lin2(out)
        return self.act2(out)


class model (BaseEstimator):
    def __init__(self, out_channels=5, lr=1e-5, nb_epoch=4, verbose=False, batch_size=1000):
        super(model, self).__init__()
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.in_channel = 3
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose

        self.conv_model = ConvModel(3, self.out_channels)
        self.lin_model = LinModel(self.out_channels)

        self.loss_fn = nn.BCELoss()
        self.optim = th.optim.Adagrad(list(self.conv_model.parameters()) + list(self.lin_model.parameters()), lr=lr)

        self.is_trained=False

    def fit(self, X, y):
        '''
        param X : numpy.ndarray
            numpy.ndarray.shape = (N, 1, C, H, W)
            C : channels
            H : height
            W : width
            N : number of image
        param y : list(numpy.ndarray)
            numpy.ndarray.shape = (W * H)
            H : height
            W : width
        '''
        if len(X.shape) != 5:
            exit("X.shape = (N, 1, C, H, W), len(X.shape) != 5 !")
        if X.shape[1] != 1:
            exit("X.shape = (N, 1, C, H, W), X.shape[1] must be 1 !")
        if X.shape[2] != 3:
            exit("X.shape = (N, 1, C, H, W), X.shape[2] must be 3 !")

        self.conv_model.train()
        self.lin_model.train()

        for i in range(self.nb_epoch):
            sum_loss = 0

            for j, (img, gt) in enumerate(zip(X, y)):

                out = self.conv_model(utils.to_float_tensor(img))

                splitted_out = th.split(out, self.batch_size, dim=0)
                splitted_gt = th.split(utils.to_float_tensor(gt), self.batch_size, dim=0)

                for o, y in zip(list(splitted_out), list(splitted_gt)):
                    self.optim.zero_grad()

                    out_batch = self.lin_model(o)
                    loss = self.loss_fn(out_batch, y.view(-1,1))

                    loss.backward(retain_graph=True)
                    self.optim.step()

                    sum_loss += loss.item()

                if self.verbose:
                    print("Epoch %d, image %d" % (i, j))

            sum_loss /= len(train_np)
            if self.verbose:
                print("[Epoch %d] loss = %f" % (i, sum_loss))

        self.is_trained = True

    def predict(self, X):
        '''
        param X : numpy.ndarray
            numpy.ndarray.shape = (N, 1, C, H, W)
            C : channels
            H : height
            W : width
            N : number of image
        return : numpy.ndarray
            numpy.ndarray.shape = (N, W * H)
        '''
        if len(X.shape) != 5:
            exit("X.shape = (N, 1, C, H, W), len(X.shape) != 5 !")
        if X.shape[1] != 1:
            exit("X.shape = (N, 1, C, H, W), X.shape[1] must be 1 !")
        if X.shape[2] != 3:
            exit("X.shape = (N, 1, C, H, W), X.shape[2] must be 3 !")

        self.conv_model.eval()
        self.lin_model.eval()
        res = np.zeros((X.shape[0], X.shape[3] * X.shape[4]))
        for j, img in enumerate(X):
            out_conv = self.conv_model(utils.to_float_tensor(img))
            pred = self.lin_model(out_conv)
            res[j,:] = pred.detach().numpy().reshape(-1)
        return res

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
