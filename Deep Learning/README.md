# About
SkolTech course

The course is about Deep Learning, i.e. a new generation of neural network-based methods that have dramatically improved the performance of AI systems in such domains as computer vision, speech recognition, natural language analysis, reinforcement learning, bioinformatics. The course covers the basics of supervised and unsupervised deep learning. It also covers the details of the two most successful classes of models, namely convolutional networks and recurrent networks. In terms of application, the class emphasizes computer vision and natural language analysis tasks. The course involves a significant practical component with a large number of practical assignments.


# Shedule
<p align="center">
  <img src="Shedule DL.png" >
</p>

# Assignments

**assignment 1** (Make Artificial Neural Network from scratch by PyTorch)
- homework_main-basic:	main notebook with research of digit classification
- homework_modules:	with all blocks implemented (Linear transform layer, SoftMax, LogSoftMax, Batch normalization, Dropout, Activation functions, Criterions, Optimizers)
- homework_differentiation:	some theoretical tasks

**assignment 2** 
- hw2_part1: 
- 2.1.1 simple task
- 2.1.2 implement [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway's_Game_of_Life) in PyTorch
- 2.1.3 build a multilayer perceptron (*i.e. a neural network of linear layers*) from scratch using **low-level** PyTorch interface for solving character recognition problem: 10 letters, ~14 000 train samples

- hw2_part2 (build a heavy convolutional neural net (CNN) to solve Tiny ImageNet image classification):
- try CNN
- try VGG-3 like CNN
- try VGG-4

**assignment 3** 
- hw03_part1_both:  train neural network to segment cells edges
- hw03_part2_autoencoders_basic:    train deep autoencoders and deploy them to faces and search for similar images (PCA, autoencoder ,denoising autoencoder)
- hw03_part2_vae_advanced:     train an autoencoder to model images of faces (Variational Autoencoder, make smiling faces)
- hw03_part3_gan_basic:  Vanilla GAN, Wasserstein GAN 
- hw03_part3a_gan_advanced:    Generating human faces with Adversarial Networks
- hw03_part3b_prd_score_advanced:   Compare VAE and GAN via Precision and Recall

**assignment 4** 
- hw04_basic_part1_regression: sequence processing to the task of predicting job salary (use nltk, job title, job description)
- hw04_basic_part2_image_captioning:  teach a network to do image captioning and compare top 10 description by BLEU score



# Project 
The topic of stock prices forecasting has been developing since the first exchange was opened, high volatility of the data and its interdependence on the variety of factors made the forecasting of stock prices an extraordinary tricky task. 
Several econometric models were developed for this sake, and the increasing amount of papers was written on the application of machine learning algorithms to time series analysis. This project is dedicated to the application of deep learning techniques to the forecasting of stock prices for several selected companies. 
We implemented Bytenet, Attention Is All You Need, BiLSTM, Dual-Stage Attention-Based RNN, ARIMA and regression models.

