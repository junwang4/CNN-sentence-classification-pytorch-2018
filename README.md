# CNN-sentence-classification-pytorch-2017

Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence) is a popular sentence-level sentiment classification approach, and it has attracted
over 1000 stars on github.com.

Kim's own implementation was based on Theano version 0.7, which is now outdated. I tried to create a virtual conda env for python 2.7 and theano 0.7, and it works for CPU. However, when switching to GPU, it simply doesn't run through. (Note that the CPU version runs very slow, with each epoch taking ~240 seconds or 4 minutes on a latest 2017 Ubuntu machine. On contrast, the GPU version only takes a couple of seconds per epoch when running my Keras or PyTorch version.)

There are several versions available on Github written in Tensorflow, Keras, and Pytorch. But it seems that they didn't integrate the pretrained word embeddings (e.g. the GoogleNews-vectors-negative300 embeddings used originally in Kim's paper).

In this repo, I want to put them together...

I only focus on the Cornell Movie Review data since it is the data provided in Kim's repo.
