################################################################################
                           #imports#
################################################################################
import math, os
import numpy as np
import scipy.signal
import random
from keras import backend as K
from keras.callbacks import ModelCheckpoint , TensorBoard , CSVLogger
from datetime import datetime
import pickle
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import keras
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,GlobalMaxPooling2D,BatchNormalization,MaxPool2D,Flatten,Input,Conv2D
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from keras.metrics import MeanSquaredError, Accuracy, Precision, Recall 
from sklearn.metrics import confusion_matrix
from keras import metrics
from tensorflow.keras.models import load_model, save_model
from keras.regularizers import l1

from classification_models.keras import Classifiers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
keras.callbacks.History()

# Ensure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


################################################################################
                           #load np files#
################################################################################

patch = np.load('/.../AUG_patch_64_conca.npy')
label = np.load('/.../AUG_patch_64_all_labels.npy')

y = np.array(label)
X = np.array(patch)
########################################################
num_0 = list(y).count(0)
num_1 = list(y).count(1)
total = num_0 + num_1 
########################################################
#y_one_hot = to_categorical(label_left_cc)
y_one_hot = np.array(label)
match = list(zip(X, y_one_hot))
random.shuffle(match)
X, y_one_hot = zip(*match)
X = np.array(X)
y_one_hot = np.array(y_one_hot)
########################################################

X_tr, X_test, y_tr, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42,stratify=y_one_hot)
y_test = to_categorical(y_test)

########################################################
# prints:
print('Examples:\n    Total: {}\n    num_0: {} ({:.2f}% of total)\n'.format(
    total, num_0, 100 * num_0 / total))
print('Examples:\n    Total: {}\n    num_1: {} ({:.2f}% of total)\n'.format(
    total, num_1, 100 * num_1 / total))


print("########################")
print("number of patches is:", len(X))
print("########################")
print("shape of patches is:", X[0].shape)
print("########################")
print("lenght of label is:", len(y_one_hot))
print("########################")
print("shape of label is:", y_one_hot[0].shape)
print("########################")
#print("weight of classes is:", d_class_weights)


################################################################################
                          
################################################################################
def datagenerator(images, labels, batchsize, mode="train"):
    while True:

        start = 0
        end = batchsize

        while start  < len(images):
            x = images[start:end]
            y = labels[start:end]
            start += batchsize
            end += batchsize
            yield x, y

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def balanced_accuracy(y_true, y_pred):
    """
    Computes the balanced accuracy.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)), axis=0)
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)), axis=0)
    false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)), axis=0)

    sensitivity = true_positives / (true_positives + false_negatives + K.epsilon())
    specificity = true_negatives / (true_negatives + false_positives + K.epsilon())

    balanced_accuracy = K.mean((sensitivity + specificity) / 2)

    return balanced_accuracy

def f1(y_true, y_pred):
    """
    Calculate F1 score using Keras backend operations
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score


input_data=[]

num_folds = 5
batch_size = 128

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

accuracies = []
balanced_accuracies = []
f1_scores = []
auc_pr_scores = []
auc_roc_scores = []


i = 0
for fold, (train_index, val_index) in enumerate(skf.split(X_tr, y_tr)):
    print(f"Fold {fold + 1}/{num_folds}")
    
    # Split data into train and validation sets
    X_train, X_val = X_tr[train_index], X_tr[val_index]
    y_train, y_val = y_tr[train_index], y_tr[val_index]

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=(64,64,3), weights='imagenet', include_top=False)
    base_model = Model(inputs=base_model.input, outputs=base_model.output)
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x = Dense(512,activation='relu',kernel_regularizer=l1(0.02))(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    preds=Dense(2,activation='softmax')(x)

    adam=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model=Model(inputs=base_model.input,outputs=preds)
    for layer in base_model.layers:
        layer.trainable = True
    print(model.summary())


    training_generator=datagenerator(X_train,y_train,batch_size)

    model.compile(optimizer= adam,loss ='binary_crossentropy', metrics=['accuracy',f1,metrics.AUC(name='AUC_PR', curve='PR'),metrics.AUC(name='AUC_ROC', curve='ROC')])
    i+=1
    checkpoint = ModelCheckpoint("/home/workspace/saved_models/Dental.keras"+format(i), monitor='loss', verbose=1, save_model= True ,save_best_only = True, mode='auto', period=1)
    

    callbacks_list=[checkpoint]

    with tf.device('/GPU:0'):
        history = model.fit(training_generator,
                    steps_per_epoch = len(X) // batch_size,
                    validation_data=(X_val, y_val),
                    epochs=150,
                    callbacks=callbacks_list
                    )

    with open('trainHistoryDict'+format(i), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    evaluation = model.evaluate(X_test, y_test)
    accuracies.append(evaluation[1])
    f1_scores.append(evaluation[2])
    auc_pr_scores.append(evaluation[3])
    auc_roc_scores.append(evaluation[4])




print("all accuracies",accuracies)
print("all f1", f1_scores)
print("all auc-pr",auc_pr_scores)
print("all auc-roc",auc_roc_scores)



print("averages:")
print("Average Accuracy:", sum(accuracies) / num_folds)
print("Average F1 Score:", sum(f1_scores) / num_folds)
print("Average AUC-PR:", sum(auc_pr_scores) / num_folds)
print("Average AUC-ROC:", sum(auc_roc_scores) / num_folds)


############################################################################################################
          ##################################################################################
                       ########################################################
                                         ####################
                                                 ####
