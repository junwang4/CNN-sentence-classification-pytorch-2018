# CNN-sentence-classification-pytorch-2017

Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence) is a popular sentence-level sentiment classification approach, and it has attracted
over 1000 stars on github.com.

Kim's own implementation was based on Theano version 0.7, which is now outdated. I tried to create a virtual conda env for python 2.7 and theano 0.7, and it works for CPU. However, when switching to GPU, it simply doesn't run through. (Note that the CPU version runs very slow, with each epoch taking ~240 seconds or 4 minutes on a latest 2017 Ubuntu machine. On contrast, the GPU version only takes a couple of seconds per epoch when running my Keras or PyTorch version.)

There are several versions available on Github written in tensorflow, keras, and pytorch. But it seems that there is no 10-fold cross-validation result.

I only focus on the Cornell Movie Review data since it is the data provided in Kim's repo.

To create a customized pretrained word embeddings from GoogleNews-vectors-negative300.bin, modify the code `utils.py`. You just need to change the following line:
```
pretrained_embedding_fpath = os.path.expanduser("~/.keras/models/GoogleNews-vectors-negative300.bin")
```


Train mode | Pretrained embeddings  |  Random (-0.01, 0.01)   | Random (shuffling pretrained embeddings)
--- | --- | --- | ---
nonstatic   | 80.0 | 75.1 | 76.3
static   | 80.1 | |


```
PYTHON 2.7 and 3.5

use_cuda = True

MODE      = static
EMBEDDING = pretrained

vocab size       = 18765
max sentence len = 56
num of classes   = 2

CNN (
  (embedding): Embedding(18765, 300)
  (conv_blocks): ModuleList (
    (0): Sequential (
      (0): Conv1d(300, 100, kernel_size=(3,), stride=(1,))
      (1): ReLU ()
      (2): MaxPool1d (size=54, stride=54, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential (
      (0): Conv1d(300, 100, kernel_size=(4,), stride=(1,))
      (1): ReLU ()
      (2): MaxPool1d (size=53, stride=53, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Sequential (
      (0): Conv1d(300, 100, kernel_size=(5,), stride=(1,))
      (1): ReLU ()
      (2): MaxPool1d (size=52, stride=52, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (fc): Linear (300 -> 2)
)

cv = 0    train size = 9595    test size = 1067
[epoch: 1] train_loss: 0.580   acc: 0.776   (1.3s)
[epoch: 2] train_loss: 0.539   acc: 0.776   (0.8s)
[epoch: 3] train_loss: 0.578   acc: 0.779   (0.9s)
[epoch: 4] train_loss: 0.495   acc: 0.784   (0.8s)
[epoch: 5] train_loss: 0.440   acc: 0.786   (0.8s)
[epoch: 6] train_loss: 0.498   acc: 0.790   (0.8s)
[epoch: 7] train_loss: 0.373   acc: 0.790   (0.8s)
[epoch: 8] train_loss: 0.400   acc: 0.786   (1.0s)
[epoch: 9] train_loss: 0.394   acc: 0.794   (1.0s)
[epoch: 10] train_loss: 0.398   acc: 0.795   (1.0s)
......
cv = 9    train size = 9596    test size = 1066
[epoch: 1] train_loss: 0.587   acc: 0.788   (0.8s)
[epoch: 2] train_loss: 0.533   acc: 0.806   (0.8s)
[epoch: 3] train_loss: 0.502   acc: 0.809   (0.8s)
[epoch: 4] train_loss: 0.459   acc: 0.805   (0.8s)
[epoch: 5] train_loss: 0.427   acc: 0.822   (0.9s)
[epoch: 6] train_loss: 0.425   acc: 0.799   (0.9s)
[epoch: 7] train_loss: 0.410   acc: 0.821   (0.8s)
[epoch: 8] train_loss: 0.439   acc: 0.811   (0.8s)
[epoch: 9] train_loss: 0.379   acc: 0.816   (0.9s)
[epoch: 10] train_loss: 0.399   acc: 0.829   (0.8s)

avg acc = 0.801  (total time: 90.5s)

```
