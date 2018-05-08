## Get Started
STEP 1
```
python utils.py path_to_your_GoogleNews-vectors-negative300.bin
```
It will take about 1 minute to run the above command. Once it is done, you can find a
customized pretrained word embedding file in a newly created folder `models`. In the case of python 3, it is: `models/googlenews_extracted-python3.pl`. Having this customized pretrained embedding file, you won't need to load the very big GoogleNews file next time.

STEP 2
```
python cnn_pytorch.py
```

Here is the output of running the above command (your numbers are likely to be different). BTW, I observed that PyTorch 0.4 can reduce running time to 66 seconds.
```
MODE      = static
EMBEDDING = pretrained

vocab size       = 18765
max sentence len = 56
num of classes   = 2

...

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

### A Glance of the Performance
Train mode | Pretrained embeddings  |  Random (-0.01, 0.01)   | Bag-of-words/tfidf + SVM
|--- | --- | --- | --- |
|nonstatic   | 80.0 | 75.0 |  |
|static      | 80.1 | 48.7 (around 50.0) |  |
|    |     |    | 79.1 |


## Background
Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence) is a popular sentence-level sentiment classification approach, and it has attracted
over 1000 stars on github.com.

Kim's own implementation was based on Theano version 0.7, which is now outdated. I tried to create a virtual conda env for python 2.7 and theano 0.7, and it works for CPU. However, when switching to GPU, it simply doesn't run through. (Note that the CPU version runs very slow, with each epoch taking ~240 seconds or 4 minutes on a latest 2017 Ubuntu machine. On contrast, the GPU version only takes a couple of seconds per epoch when running my Keras or PyTorch version.)

There are several versions available on Github written in tensorflow, keras, and pytorch. But it seems that there is no 10-fold cross-validation result.

I only focus on the Cornell Movie Review data since it is the data provided in Kim's repo.

## More
To compare CNN with the traditional bag of words model, I also provide a python script `bag_of_words.py`.
