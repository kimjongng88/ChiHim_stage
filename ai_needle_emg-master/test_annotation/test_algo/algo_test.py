# Import the necessary packages
from learning_rate_schedulers import PolynomialDecay
from minigooglenet import MiniGoogLeNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use("Agg")


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
 
    # returned the smoothed labels
    return labels

def input_smoothing_value():
    print('Please input the Smoothing value:')
    SMOOTHING = input()
    print('Smoothing value of {}'.format(SMOOTHING))
    return SMOOTHING

SMOOTHING = input_smoothing_value()
SMOOTHING = float(SMOOTHING)


# define the total number of epochs to train for, initial learning
# rate, and batch size
NUM_EPOCHS = 2#32
INIT_LR = 5e-3
BATCH_SIZE = 64

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
print(trainX)
print(trainX.shape)
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors, converting the data
# type to floats so we can apply label smoothing
lb = LabelBinarizer()
print(trainY,type(trainY),'ervoor')
trainY = lb.fit_transform(trainY)
print(trainY,type(trainY),'ernaaaaaaaaaaa')
testY = lb.transform(testY)
trainY = trainY.astype("float")
testY = testY.astype("float")

# apply label smoothing to the *training labels only*
print("[INFO] smoothing amount: {}".format(SMOOTHING))
print("[INFO] before smoothing: {}".format(trainY[0]))
trainY = smooth_labels(trainY, SMOOTHING)

print("[INFO] after smoothing: {}".format(trainY[0]))
# construct the image generator for data augmentation
aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest")

# construct the learning rate scheduler callback
schedule = PolynomialDecay(maxEpochs=NUM_EPOCHS,
                           initAlpha=INIT_LR,
                           power=1.0)
callbacks = [LearningRateScheduler(schedule)]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.7)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
print(trainX.shape, trainY.shape, 'shaaaaaaaaaaaaaaaappppppppppppppppppeeeeeeeeeeeeeeeee')
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=1)

model.save('minigooglenet_explicit_smooth_labels.h5')

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

# construct a plot that plots and saves the training history
def plot_history_metrics(metric, val_metric, lbl_metric, lbl_val_metric, title, ylabel, plt_file_name):
    N = np.arange(0, NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, metric, label=lbl_metric)
    plt.plot(N, val_metric, label=lbl_val_metric)
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(ylabel)
    plt.legend(loc="lower left")
    plt.savefig(plt_file_name)

plot_history_metrics(H.history["loss"],
                     H.history["val_loss"],
                     "train_loss",
                     "val_loss",
                     "Training Loss vs Validation Loss",
                     "Loss",
                     "loss_value_label_smoothing_explicitly_updating_labels_list")

plot_history_metrics(H.history["accuracy"],
                     H.history["val_accuracy"],
                     "train_accuracy",
                     "val_accuracy",
                     "Training Accuracy vs Validation Accuracy",
                     "Accuracy",
                     "Accuracy_value_label_smoothing_explicitly_updating_labels_list")

# initialize the Optimizer and Loss
print("[INFO] smoothing amount: {}".format(SMOOTHING))
opt = SGD(lr=INIT_LR, momentum=0.9)
loss = CategoricalCrossentropy(label_smoothing=SMOOTHING)

print("[INFO] compiling model...")
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# Train the MiniGoogleNet network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=1)

plot_history_metrics(H.history["accuracy"],
                     H.history["val_accuracy"],
                     "train_accuracy",
                     "val_accuracy",
                     "Training Accuracy vs Validation Accuracy",
                     "Accuracy",
                     "accuracy_value_label_smoothing_by_loss_function")

plot_history_metrics(H.history["loss"],
                     H.history["val_loss"],
                     "train_loss",
                     "val_loss",
                     "Training Loss vs Validation Loss",
                     "Loss",
                     "loss_value_label_smoothing_by_loss_function")