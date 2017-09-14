from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import data_helpers
import numpy as np
import pickle
import time
import os
import sys
from sklearn.model_selection import KFold

use_cuda = torch.cuda.is_available()
print('use_cuda = {}\n'.format(use_cuda))

print('-'*20)
mode = "static"
mode = "nonstatic"
use_pretrained_embeddings = False
use_pretrained_embeddings = True
print('MODE      = {}'.format(mode))
print('EMBEDDING = {}'.format("pretrained" if use_pretrained_embeddings else "random"))
print('-'*20 + '\n')

np.random.seed(0)

X, Y, vocabulary, vocabulary_inv_list = data_helpers.load_data()

vocab_size = len(vocabulary_inv_list)
sentence_len = X.shape[1]
num_classes = max(Y) +1

print('vocab size       = {}'.format(vocab_size))
print('max sentence len = {}'.format(sentence_len))
print('num of classes   = {}'.format(num_classes))
print('-'*20 + '\n')

ConvMethod = "in_channel__is_1"
ConvMethod = "in_channel__is_embedding_dim"

class CNN(nn.Module):
    def __init__(self, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=300, pretrained_embeddings=None):
        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = mode=="nonstatic"

        if use_cuda:
            self.embedding = self.embedding.cuda()

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size +1

            if ConvMethod == "in_channel__is_embedding_dim":
                conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters, kernel_size = kernel_size, stride = 1)
            else:
                conv1d = nn.Conv1d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size*embedding_dim, stride = embedding_dim)

            conv_blocks.append(
                nn.Sequential(
                    conv1d,
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size = maxpool_kernel_size)
                ).cuda()
            )

        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters*len(kernel_sizes), num_classes)

    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        #x = F.dropout(x, p=0.5)

        if ConvMethod == "in_channel__is_embedding_dim":
            #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        else:
            #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)

        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)

        out = F.dropout(out, p=0.5, training=self.training)
        out = out.view(out.size(0), -1)

        return F.softmax(self.fc(out))

def evaluate(model, x_test, y_test):
    model.eval()
    inputs = Variable(x_test)
    preds = model(inputs).cuda()
    preds = torch.max(preds, 1)[1].cuda()
    eval_acc = sum(preds.data == y_test) / len(y_test)
    return eval_acc


embedding_dim = 300
num_filters = 100
kernel_sizes = [3,4,5]
batch_size = 50

def load_pretrained_embeddings():
    pretrained_fpath_saved = os.path.expanduser("models/googlenews_extracted.pl")

    if os.path.exists(pretrained_fpath_saved):
        with open(pretrained_fpath_saved, 'r') as f:
            embedding_weights = pickle.load(f)
    else:
        print('- Error: file not found : {}\n\n'.format(pretrained_fpath_saved))
        sys.exit()
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    out = np.array(embedding_weights.values())
    #np.random.shuffle(out)
    return out

if use_pretrained_embeddings:
    pretrained_embeddings = load_pretrained_embeddings()
else:
    pretrained_embeddings = np.random.uniform(-0.01, -0.01, size=(vocab_size, embedding_dim))


def train_test_one_split(cv, train_index, test_index):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    x_test = torch.from_numpy(x_test).long().cuda()
    y_test = torch.from_numpy(y_test).long().cuda()

    model = CNN(kernel_sizes, num_filters, embedding_dim, pretrained_embeddings)
    if cv==0:
        print("\n{}\n".format(str(model)))

    if use_cuda:
        model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 1+10):
        start = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            model.train()
            preds = model(inputs).cuda()
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 0: # seems not make difference or even worse (it was used in Kim's original paper)
                constrained_norm = 3
                if model.fc.weight.norm().data[0] > constrained_norm:
                    model.fc.weight.data = model.fc.weight.data * constrained_norm / model.fc.weight.data.norm()

        eval_acc = evaluate(model, x_test, y_test)
        print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'.format(epoch, loss.data[0], eval_acc, time.time()-start) )
    return eval_acc

cv_folds = 10
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
acc_list = []
start = time.time()
for cv, (train_index, test_index) in enumerate(kf.split(X)):
    acc = train_test_one_split(cv, train_index, test_index)
    print('cv = {}    train size = {}    test size = {}\n'.format(cv, len(train_index), len(test_index)))
    acc_list.append(acc)
print('\navg acc = {:.3f}   (total time: {:.1f}s)\n'.format(sum(acc_list)/len(acc_list), time.time()-start))
