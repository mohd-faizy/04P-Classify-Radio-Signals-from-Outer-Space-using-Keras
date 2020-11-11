# Classify_Radio_Signals_from_Outer_Space_using_Keras

> In this project, you will learn the basics of using Keras with TensorFlow as its backend and use it to solve an image classification problem. The data consists of 2D spectrograms of deep space radio signals collected by the Allen Telescope Array at the SETI Institute. The spectrograms will be treated as images to train an image classification model to classify the signals into one of four classes. By the end of the project, you will have built and trained a convolutional neural network from scratch using Keras to classify signals from space.

The model could be optimized using hyperparameter tuning. However, the goal of this notebook is not to build a high performing classifier, rather to show the basic steps to build an image classifier using convolutional neural network. The readers can also get the idea of

- Data augmentation using ImageDataGenerator, and
- Way of saving the weights of a model at some interval which can later be used for transfer learning through 'callbacks' during fitting the model

<center><img src='http://blog.yavilevich.com/wp-content/uploads/2016/08/fosphor2-cut.png' width=700 height=200></center>

> ## Objectives:
    - Build and train a convolutional neural network (CNN) using Keras
    - Display results and plot 2D spectrograms with Python in Jupyter Notebook

## __SETI Dataset__

> [__Click here__](https://drive.google.com/file/d/1R2BlsYydirhMmf89_D1imOT5aVvkXHi2/view?usp=sharing)

## __Packages Used__

 - [Tensorflow](https://www.tensorflow.org/)
 - [Sklearn](https://scikit-learn.org)
 - [Matplotlib](https://matplotlib.org/)
 - [Numpy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [seaborn](https://seaborn.pydata.org/)


## __1: Introduction and Import Libaries__

  - Introduction to the data and and overview of the project.
  - Introduction to the Rhyme interface.
  - Import essential modules and helper functions from NumPy, Matplotlib, and Keras.

## __2: Load and Preprocess SETI Data__

  - Display 2D spectrograms using Matplotlib.
  - Reshape the input data with NumPy.

## __3: Create Training and Validation Data Generators__

  - Generate batches of tensor image data with real-time data augmentation.
  - Specify paths to training and validation image directories and generates batches of augmented data.

## __4: Create a Convolutional Neural Network (CNN) Model__

  - Design a convolutional neural network with 2 convolution layers and 1 fully connected layers to predict four signal types.
  - Use Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.

## __5: Learning Rate Scheduling and Compile the Model__

  - When training a model, it is often recommended to lower the learning rate as the training progresses.
  - Apply an [Exponential Decay Function](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay) to the provided initial learning rate.

## __6: Train the Model__

  - Train the CNN by invoking the `model.fit()` method.
  - Use `ModelCheckpoint()` to save the weights associated with the higher validation accuracy after every epoch
  - Display live training loss and accuracy plots in Jupyter Notebook using [livelossplot](https://github.com/stared/livelossplot).
  
## __7: Evaluate the Model__

  - Evaluate the CNN by invoking the `model.fit()` method.
  - Obtain the classification report to note the precision and recall of your classifier.
