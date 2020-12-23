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
