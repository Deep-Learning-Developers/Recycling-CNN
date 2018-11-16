import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
np.random.seed(3)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='1' # Disables info log 

work_dir = ''
image_height, image_width = 28, 28
train_dir = os.path.abspath('./BottleNotBottleALL/Train')
test_dir = os.path.abspath('./BottleNotBottleALL/Valid')
no_classes = 2
num_epochs = int(input('Input Number of Epochs: '))
batch_size = 10
no_train = 932+389
no_validation = 323+175
no_test = 131+321
input_shape = (image_height, image_width, 3)
epoch_steps = no_train // batch_size
test_steps = no_test // batch_size
classes=['Bottle','Not']

input_shape = (image_width, image_height, 3)
epoch_steps = no_train // batch_size
test_steps = no_test // batch_size

print(train_dir)

def simple_cnn(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))
    model.add(keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1024, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    return model

simple_cnn_model = simple_cnn(input_shape)
generator_train = ImageDataGenerator(rescale=1. / 255)
generator_test = ImageDataGenerator(rescale=1. / 255)
train_images = generator_train.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height),
    classes=classes
    )
test_images = generator_test.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height),
    classes=classes)
history = simple_cnn_model.fit_generator(
    train_images,
    steps_per_epoch=epoch_steps,
    epochs=num_epochs,
    validation_data=test_images,
    validation_steps=test_steps,
    verbose=1)

# Save the CNN Model
model_yaml = simple_cnn_model.to_yaml()
with open('CDModel-BNB-SGD-Graphing'+str(num_epochs)+'.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
    
# Save model weights
simple_cnn_model.save_weights('CDModel-BNB-SGD-Graphing-'+str(num_epochs)+'.h5')
print('')
print('Saved Model and Weights.')

score = simple_cnn_model.evaluate_generator(test_images)
print('Test loss:', score[0])
print('Test accuracy: {:.5f}%'.format(score[1]*100))


#Print all data in the history dictionary
#print(history.history.keys()) # For diagnostic purposes
#print(history.history['acc'])
#print(history.history['val_acc'])
#print(history.history['loss'])
#print(history.history['val_loss'])

#Summarize history of accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch #')
plt.legend(['Training', 'Validation'], loc='best')
plt.savefig('BottleNotBottle_Figures\CDModel-BNB-SGD-Graphing-Accuracy-Graph-'+str(num_epochs)+'.png')
# plt.show() # Shows the graph but prevents the program from continuing until figure is closed
plt.close() # Closes Accuracy Figure to graph Loss Figure

# Summarize history of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.legend(['Training', 'Validation'], loc='best')
plt.savefig('BottleNotBottle_Figures\CDModel-BNB-SGD-Graphing-Loss-Graph-'+str(num_epochs)+'.png')
# plt.show() # Shows the graph but prevents the program from continuing until figure is closed
