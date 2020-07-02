# 04P_Classify_Radio_Signals_from_Outer_Space_using_Keras
The data we are going to use consists of 2D spectrograms of deep space radio signals collected by the Allen Telescope Array at the SETI Institute. We will treat the spectrograms as images to train an image classification model to classify the signals into one of four classes.

<center><img src='http://blog.yavilevich.com/wp-content/uploads/2016/08/fosphor2-cut.png' width=700 height=200></center>


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
    - Apply an exponential decay function to the provided initial learning rate.

## __6: Train the Model__

    - Train the CNN by invoking the model.fit() method.
    - Use ModelCheckpoint() to save the weights associated with the higher validation accuracy after every epoch
    - Display live training loss and accuracy plots in Jupyter Notebook using livelossplot.
## __7: Evaluate the Model__

    - Evaluate the CNN by invoking the model.fit() method.
    - Obtain the classification report to note the precision and recall of your classifier.
