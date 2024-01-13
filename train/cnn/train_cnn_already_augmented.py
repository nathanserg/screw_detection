import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras.applications import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import itertools

os.getcwd()

TRAIN_DATASET_PATH = 'datasets/Dataset_PJE_Grayscale_Augmented'
VALID_DATASET_PATH = 'datasets/Dataset_PJE_Grayscale_Augmented'

model = DenseNet201
model_name = 'DenseNet201'

experimentation_number = 100
experimentation_description = 'Grayscale'


tensorboard_directory   = 'log_files'
weight_directory        = 'weights'
final_weight_directory = weight_directory + '/DenseNet201_final.h5'

COLOR_SPACE = "rgb" #"grayscale"


IMAGE_SIZE    = (71, 71)
NUM_CLASSES   = 6
BATCH_SIZE    = 64  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 100


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    output_filename = f'analyse/{experimentation_number}_{experimentation_description}'
    create_folder(output_filename)
    plt.savefig(output_filename+'/Confusion_Matrix.svg')
    plt.show()

def create_folder(folder_path):
    # Check if the folder does not exist
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def plot_history_function(history):
    plt.figure(1)
    plt.subplot(211)
    plt.ylim(0.5, 1)
    plt.plot(history.history['accuracy'], marker='o', label='training accuracy')
    plt.plot(history.history['val_accuracy'], marker='o', label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.subplot(212)
    plt.plot(history.history['loss'], marker='o', label='training loss')
    plt.plot(history.history['val_loss'], marker='o', label='validation loss')
    plt.xlabel('Number of epoch')
    plt.ylabel('loss')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.legend()
    output_filename = f'analyse/{experimentation_number}_{experimentation_description}'
    create_folder(output_filename)
    plt.savefig(output_filename+'/Accuracy_Loss_Graphs.svg')
    plt.show()


datagen = ImageDataGenerator(rescale=1./255,
                            validation_split=0.25)

train_batches = datagen.flow_from_directory(TRAIN_DATASET_PATH,
                                            target_size=IMAGE_SIZE,
                                            color_mode=COLOR_SPACE,
                                            class_mode='categorical',
                                            batch_size=BATCH_SIZE,
                                            subset='training')


valid_batches = datagen.flow_from_directory(VALID_DATASET_PATH,
                                            target_size=IMAGE_SIZE,
                                            color_mode=COLOR_SPACE,
                                            class_mode='categorical',
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            subset='validation')


# Afficher la data augmentation
"""
for j in range(5):
    x,y = train_batches.next()
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        image = 255*x[i-1]
        print(image.shape())
        RGB_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        fig.add_subplot(rows, columns, i)
        plt.imshow(RGB_img.astype('uint8'))
    plt.show()
"""



print('****************')
classes_name = [0 for i in range(NUM_CLASSES)]
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
    classes_name[idx] = cls
print(classes_name)
print('****************')

l2 = keras.regularizers.l2(0.01)
base_model = model(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

base_model.trainable = False

# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2)(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2)(x)
x = Dropout(0.1)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2)(x)
x = Dropout(0.1)(x)

# and a logistic layer -- let's say we have 2 classes
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, factor=0.25,
                                  patience=5, min_lr=1e-12)
mc = ModelCheckpoint('checkpoint.h5', monitor='val_accuracy', verbose=1,
                      save_best_only=True,
                      save_weights_only=True, mode='auto', patience=1)
callbacks_list = [reduce_lr, mc]

# train the model
history1 = model.fit(train_batches,
                     steps_per_epoch = train_batches.samples // BATCH_SIZE,
                     validation_data = valid_batches, 
                     validation_steps = valid_batches.samples // BATCH_SIZE,
                     epochs = NUM_EPOCHS,
                     callbacks=callbacks_list)


#Fine Tuning the model with a slow learning rate
# model.save_weights(final_weight_directory)
# model.load_weights(final_weight_directory)

# base_model.trainable = True

# model.compile(
#     optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
#     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])

# NUM_EPOCHS = 100
# BATCH_SIZE = 32

# history2 = model.fit(train_batches,
#                      steps_per_epoch = train_batches.samples // BATCH_SIZE,
#                      validation_data = valid_batches, 
#                      validation_steps = valid_batches.samples // BATCH_SIZE,
#                      epochs = NUM_EPOCHS)

results = model.evaluate(valid_batches, steps = valid_batches.samples // BATCH_SIZE)
print(results[1])

# # save trained weight
model.save_weights(final_weight_directory)


plot_history_function(history1)
# plot_history_function(history2)


Y_pred = model.predict_generator(valid_batches, valid_batches.samples // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(valid_batches.classes, y_pred)
cm_plot_labels = ['allen', 'hexagonal', 'nas', 'philipsAndPZ', 'slotted', 'torx']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels)


