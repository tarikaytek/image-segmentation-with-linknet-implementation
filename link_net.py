
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import random

from tensorflow.keras import backend as K
import numpy as np

import matplotlib.pyplot as plt

##downloading dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

#checking what it looks like
print('how dataset looks like:', dataset['train'])

#info about dataset
print('info about dataset:', info)

####### analysing dataset by counting its labels 
train_labels = []
train_species = []

for e in dataset['train']:
  train_species.append(int(e['species']))
  train_labels.append(int(e['label']))

test_labels = []
test_species = []

for e in dataset['test']:
  test_species.append(int(e['species']))
  test_labels.append(int(e['label']))
#######

print('num of samples in train set: ', len(dataset['train']), 'num of samples in test set: ', len(dataset['test']))

from collections import Counter

train_labels_counter = Counter(sorted(train_labels))
train_species_counter = Counter(sorted(train_species))
test_labels_counter = Counter(sorted(test_labels))
test_species_counter = Counter(sorted(test_species))


########## how many labels do we have? how many of each of them do we have?
########## output of this part is very long so i commented it out
########## outputs can be seen in the video
'''
for e in train_labels_counter:
  print(e, train_labels_counter[e])
print('------------------------------------------')

for e in train_species_counter:
  print(e, train_species_counter[e])
print('------------------------------------------')

for e in test_labels_counter:
  print(e, test_labels_counter[e])
print('------------------------------------------')

for e in test_species_counter:
  print(e, test_species_counter[e])
'''
### Some hiperparameters 
IMG_SIZE = (256,256)
SPLIT = 0.7
BATCH_SIZE = 64
#BUFFER_SIZE = 1000
N_CLASSES = 3


####### loading data
####### resizes and normalizes it on the fly
####### mask is like [1,2,3] but it should be [0,1,2]. so we substract 1 
def load(data, img_size=IMG_SIZE):
    img = tf.image.resize(data['image'], size=img_size) / 255.
    mask = tf.image.resize(data['segmentation_mask'], size=img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) -1
    return img, mask

##### train val split with given ratio
def split(dataset, split_ratio=SPLIT, buffer_size=tf.data.AUTOTUNE):
    train_size = int(len(dataset) * split_ratio)
    dataset.shuffle(len(dataset))
    train = dataset.take(train_size)
    val = dataset.skip(train_size)

    return train, val

##### load train and test sets
train = dataset['train'].map(load)
test = dataset['test'].map(load)

##### split training data into train and val sets
train, val = split(dataset=train, split_ratio=SPLIT)

##### batching all the data
train_batches = train.cache().batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)
val_batches = val.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


print('size of train set: ', len(train), 'size of validation set: ', len(val), 'size of test set: ', len(test))

##### function that returns proper model according to Link Net paper
def create_model():

    #function that returns output of conv layer consists of 3 layers: 
    # 2D convolution with given parameters, batch normalization, relu activation function
    def conv_layer(input, filters, kernel_size, strides=1, padding='same'):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    #function that returns output of full-conv(transposed conv) layer consists of 3 layers:
    #transposed 2d convolution with given parameters, batch normalization, relu activation function
    def full_conv_layer(input, filters, kernel_size, strides=2, padding='same'):

        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        return x

    #function that returns output of an encoder block as its described in paper
    def encoder_block(input, filters, strides_of_first_conv=2):
        

        res = conv_layer(input, filters, 1, strides=strides_of_first_conv)

        x = conv_layer(input, filters, 3, strides=strides_of_first_conv)
        x = conv_layer(x, filters, 3)
        
        x = layers.Add()([x, res])

        res = x

        x = conv_layer(x, filters, 3)
        x = conv_layer(x, filters, 3)

        x = layers.Add()([x, res])

        return x
    
    #function that returns  output of a decoder block as its described in paper
    def decoder_block(input, filters, strides_of_full_conv=2):

        x = conv_layer(input, filters/2, 1)

        x = full_conv_layer(x, filters/2, 3, strides=strides_of_full_conv)

        x = conv_layer(x, filters, 1)

        return x
    
    ####### needed methods has implemented, creating model using these functions below.

    #initializing input tensor in proper shape
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    ###initial layer 
    x = conv_layer(input=inputs, filters=64, kernel_size=7, strides=2)
    x = layers.MaxPooling2D(pool_size=2)(x)

    ###encoder blocks
    #saving their outputs to use it in residual connections
    encoder_1_output = encoder_block(input=x, filters=64, strides_of_first_conv=1)
    encoder_2_output = encoder_block(input=encoder_1_output, filters=128)
    encoder_3_output = encoder_block(input=encoder_2_output, filters=256)
    encoder_4_output = encoder_block(input=encoder_3_output, filters=512)

    ###decoder blocks
    #residual connections is made here by tf.layers.Add function which adds given tensors 
    
    #decoder block 4
    x = decoder_block(input=encoder_4_output, filters=256)
    #decoder block 3
    x = layers.Add()([x, encoder_3_output])
    x = decoder_block(input=x, filters=128)
    #decoder block 2
    x = layers.Add()([x, encoder_2_output])
    x = decoder_block(input=x, filters=64)
    #decoder block 1
    x = layers.Add()([x, encoder_1_output])
    x = decoder_block(input=x, filters=64, strides_of_full_conv=1)

    ###final block
    x = full_conv_layer(input=x, filters=32, kernel_size=3)
    x = conv_layer(input=x, filters=32, kernel_size=3)
    #output layer
    outputs = layers.Conv2DTranspose(filters=N_CLASSES, kernel_size=2, strides=2, activation='softmax', padding='same')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
    
##### instantiating model
model = create_model()
model.summary()


#### dice coefficient score
def dice_coef(y_true, y_pred):
    pred_mask = create_mask(y_pred)
    pred_mask = tf.cast(pred_mask, tf.float32)
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = K.flatten(pred_mask)

    intersect = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice = (2. * intersect + K.epsilon()) / (union + K.epsilon())
    dice = K.mean(dice)
    
    return dice
  
### dice coefficient loss function
def dice_loss(y_true, y_pred):
  return 1. - dice_coef(y_true, y_pred)

### creates mask of given softmax output
def create_mask(y_pred):
  pred_mask = tf.math.argmax(y_pred, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask


### function for displaying 
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


#### Some hiperparameters for model
EPOCHS=30
STEPS_PER_EPOCH = len(train) / BATCH_SIZE
VAL_SUBSPLITS = 5
VALIDATION_STEPS= len(val) / BATCH_SIZE / VAL_SUBSPLITS

#callback function to save best model 
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)
]

### compiling by adam optimizer and sparse categorical crossentropy loss function
model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['accuracy', dice_coef])
hist = model.fit(train_batches, validation_data=val_batches, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, callbacks=callbacks)

### plots historical score of given metric
def plot(metric):
  train = hist.history[metric]
  val = hist.history['val_' + metric]

  plt.figure()
  plt.plot(hist.epoch, train, 'r', label='training ' + metric)
  plt.plot(hist.epoch, val, 'bo', label='validation ' + metric)
  plt.title('training and validation ' + metric)
  plt.xlabel('epoch')
  plt.ylabel(metric + ' value')
  #plt.ylim([0, 1])
  plt.legend()
  plt.show()

### plots for below metrics 
plot('loss')
plot('accuracy')
plot('dice_coef')

####evaluating
result = model.evaluate(test_batches)

####overall scores in training and validation data
for e in hist.history:
  l = hist.history[e]
  print('max ' + e, max(l))
  print('avg ' + e, sum(l) / len(l))
  print('final ' + e, l[-1], '\n')

###scores in test data
print('loss in test data:', result[0])
print('accuracy in test data:', result[1])
print('dice coefficient in test data:', result[2])


####printing same predictions 
num_of_samples = 5

for img, mask in test_batches.take(num_of_samples):
  pred = model.predict(img)
  pred_mask = create_mask(pred)
  i = random.randint(0, BATCH_SIZE)
  display([img[i], mask[i], pred_mask[i]])

print('YAY')