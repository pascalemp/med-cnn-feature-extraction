# COVID-19 X-Ray Feature Inference using Convolutional Neural Networks

## Aim and Objectives

The main aim of this project is to implement a Convolutional Neural Network to classify X-Ray images in order to determine if a given radiograph is said to have pneumonia or not. Due to the ambiguous nature of Neural Networks, it's not fully understood the reasons behind _why_ they choose to classify images in a certain way. Because of this, I've decided to analyse the model outputs at different depths of the network. Further to this, it would be advantaegous to gain some insight into the features that the model is learning, in order to visualise what may constitute a diagnosis of pneumonia. This will be done by visualising both the input images and the respective 'learned' features by examining the respective Convolutional Layer outputs.

### Specific Objective(s)

* __Objective 1:__ _Pre-process image data and convert to a network friendly format, using Pandas DataFrame objects._
* __Objective 2:__ _Generate a Neural Network to classify the images into their respective categories._
* __Objective 3:__ _Analyse some of the Convolutional Layer outputs._

## The Dataset

The dataset I have chosen to use is a collection of grayscale peadiatric X-Ray images, consisting of two categories: 'pneumonia' and 'normal'. There are a total of 5856 jpeg images, with pre-allocated folder labels and train-test-validation splits.

There is good variation within the respective subsets, whereby both anterior and posterior images are included. The images were selected from retrospective cohorts of pediatric patients, agin between one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. 

In terms of image accuracy and quality, all images were initially screened by physicians and any low quality or unreadable / ambiguous scans were removed. The images were then expertly classified by leading physicians before being assigned their respective labels.

There are a few small issues with the data that may pose issues later during training; there is a significant class imbalance. In total, there are 1583 images classed as 'normal', in contrast to 4273 images classed as 'pneumonia', this equates to there being 2.7 times more images of the 'pneumonia' class than 'normal' class. It is not yet clear as to whether this will impact performance, but this will need to be considered prior to training our network.

Another minor issue is regarding the image sizes, each image is of varying size, dependent on whether the images were taken anteriorly or posteriorly, which leads to slight variance in the aspect ratio. Because of this, we may need to consider an algorithm to determine 'regions of interest' in the images, prior to training. This will also need to be considered when pre-processing the data, but due to the uniform nature of X-Ray images in general, this shouldn't be an image if we are required to crop or downsample.

The dataset itself is derived originally from [this](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) paper, but is downloaded from [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) Kaggle repository.

## Network Architecture

The architecture of the bulk of the project revolves around the implementation of a Convolutional Neural Network. In the _Program Code_ section below, you will see more detail regarding this architecture, but the network structure is based on that of the _MiniVGGNet_ architecture:


![network_image](https://www.researchgate.net/publication/332435757/figure/fig1/AS:748140944703488@1555382109474/Architecture-of-MiniVGGNet.jpg)

This network tends to perform well on a variety of different image classification tasks, and will hopefully provide a good basis for our network to be built upon. The use of convolutional layers here is imperative, as these allow us to convolve image filters known as _kernels_ over the image, in order to extract features and information. The combination of this, along with the _max pooling_ layers, will allow the most common features to be carried through the network, for which the model will be able to infer further patterns / structures.

## Processing Modules and Algorithms

The bulk of the processing comes from cleaning and formatting the dataset itself, this will involve:

* __Extracting Image Paths + Categories from the raw dataset:__

    This will allow the images to be referenced later by the network in order to deduce whether an input image was originally classed as 'normal' or 'pneumonia'.
    
    
* __Combining the Train, Test and Validation sets in order to manually split the images.__

    Since there is a lack of images to begin with, we want to prioritise model training over testing, as this is more important in this case.


* __Converting output samples to One-Hot Encoded Values which are required for classification.__

    This is required for the model to use _softmax_ activation in order to output a probability that the model has chosen the correct class.
    

* __Resizing / Downsampling images to better fit our chosen network architecture.__

    Since the images vary in size, we must downsample them. This is also done in order to ensure the model doesn't take too long to train.
    
    
* __Training the Neural model.__


* __Analysing and visualising the layer outputs for a variety of layers.__

    We will make a smaller, truncated version of our model and analyse the intermediary outputs of certain layers and display them visually to try and gain insight into how the model is _learning_.
