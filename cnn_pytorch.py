from __future__ import division
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

use_cuda = torch.cuda.is_available()
print('use_cuda', use_cuda)

np.random.seed(10)

def load_data():
    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {rank: word for rank, word in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv

# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data()
print(" ".join([word for id, word in vocabulary_inv.items() if id<10]))
print(" ".join([word for id, word in vocabulary_inv.items() if id>len(vocabulary_inv)-10]))

vocab_size = len(vocabulary_inv)
sentence_len = x_train.shape[1]

print('vocab size', vocab_size)
print('max sentence len', sentence_len)
print('num of train sentences', x_train.shape[0], '  test sentences', x_test.shape[0])
print('train', x_train[0][:5])
print('test ', x_test[0][:5])
print('-----------\n')


class KimCNN(nn.Module):
    def __init__(self, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=300, num_classes=2, pretrained_weight=None):
        super(KimCNN, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.embedding.weight.requires_grad = False
        self.embedding.weight.requires_grad = True
        print('***** update embedding weights = {}'.format(self.embedding.weight.requires_grad))

        if pretrained_weight is None:
            print('***** embeddings: randomly generated')
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            print('***** embeddings: pretrained')

        if use_cuda:
            self.embedding = self.embedding.cuda()

        self.conv_blocks = []
        for kernel_size in kernel_sizes:

            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size +1
            self.maxpool_kernel_size = sentence_len - kernel_size +1

            if 0:
              self.conv_blocks.append(
                nn.Sequential(
                    #nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters, kernel_size = kernel_size, stride = 1),
                    nn.Conv1d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size*embedding_dim, stride = embedding_dim),

                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size = maxpool_kernel_size)
                ).cuda()
              )

            #for i in range(len(self.kernel_sizes)):
                #conv = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size = kernel_size*embedding_dim, stride = embedding_dim),
                #setattr(self, 'conv_{i}', conv)

            self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=ks*embedding_dim, stride=embedding_dim) for ks in kernel_sizes])



        #self.fc = nn.Sequential( nn.Linear(in_size, num_classes), nn.Softmax())
        self.fc = nn.Linear(num_filters*len(kernel_sizes), num_classes)

    def get_conv(self, i):
        return getattr(self, 'conv_{i}')

    def forward(self, x):       # x: (batch, sentence_len)
        #print(x[:2])
        #sys.exit()

        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        #x = F.dropout(x, p=0.5)

        # nn.Conv1d
        #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
        #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        #x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)

        #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
        #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)

        #xs= [conv_block(x) for conv_block in self.conv_blocks]
        #xs= [F.max_pool1d(F.relu(self.get_conv(i)(x)), maxpool_kernel_size) 
        xs= []
        for conv in self.convs:
            tmp = conv(x)
            tmp = F.relu(tmp)
            tmp = F.max_pool1d(tmp, self.maxpool_kernel_size) 
            xs.append(tmp)

        out = torch.cat(xs, 2)

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
num_classes = 2
kernel_sizes = [3,4,5]
batch_size = 50

def load_pretrained_embeddings():
    #pretrained_fpath_saved = os.path.expanduser("models/googlenews_extracted.pl")
    pretrained_fpath_saved = os.path.expanduser("models/googlenews_customized.pl")
    #pretrained_fpath_saved = os.path.expanduser("models/googlenews_customized2.pl")

    if os.path.exists(pretrained_fpath_saved):
        with open(pretrained_fpath_saved, 'r') as f:
            embedding_weights = pickle.load(f)
            #print('loading pretrained word embeddings ... done') # data: {word_index:300-dim weight}
    else:
        print('- error: not found file', pretrained_fpath_saved)
    # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
    out = np.array(embedding_weights.values())

    use_shuffled_pretrained_weights = False
    use_shuffled_pretrained_weights = True
    print('***** shuffle pretrained weights = {}'.format(use_shuffled_pretrained_weights))

    if use_shuffled_pretrained_weights:
        #print ('before shuffle', out[1][:5])
        np.random.shuffle(out)
        #print (out[1][:5])
    words = open(os.path.expanduser("models/words.dat")).read().strip().split()
    #print words[:10]
    return out, words

pretrained_weight = None
pretrained_weight = 1
if pretrained_weight is not None:
    pretrained_weight, word_list = load_pretrained_embeddings()

model = KimCNN(kernel_sizes, num_filters, embedding_dim, num_classes, pretrained_weight)
if use_cuda:
    model = model.cuda()

print("\n")
print(model)
print("\n")

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0002)

loss_fn = nn.CrossEntropyLoss()

x_train = torch.from_numpy(x_train).long()
y_train = torch.from_numpy(y_train).long()
dataset_train = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

x_test = torch.from_numpy(x_test).long().cuda()
y_test = torch.from_numpy(y_test).long().cuda()
#dataset_test = TensorDataset(x_test, y_test)
#test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

for epoch in range(1, 15):
    s = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

        model.train()
        preds = model(inputs).cuda()
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if i>10: break

        #constrained_norm = 3
        #if model.fc.weight.norm().data[0] > constrained_norm:
        #    model.fc.weight.data = model.fc.weight.data * constrained_norm / model.fc.weight.data.norm()


    eval_acc = evaluate(model, x_test, y_test)
    print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)'.format(epoch, loss.data[0], eval_acc, time.time()-s) )
