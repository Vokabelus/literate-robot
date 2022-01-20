import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import models

plt.style.use('dark_background')

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import normalize, to_categorical

# normalize inputs

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = normalize(X_train,axis=1)
X_test  = normalize(X_test,axis=1)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Data augmentation

train_data = ImageDataGenerator(rotation_range=45,
            width_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

train_data.fit(X_train)

train_generator = train_data.flow(
    X_train,
    Y_train,
    batch_size=32)

# DL part model similar to VGG
activation = 'sigmoid'
model = Sequential()
model.add(Conv2D(32,3,activation=activation,padding = 'same',input_shape = (32,32,3)))
model.add(BatchNormalization())

model.add(Conv2D(32,3,activation=activation,padding = 'same',kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())

model.add(MaxPooling2D())
model.add(Conv2D(64,3,activation=activation,padding = 'same',kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())


model.add(Conv2D(64,3,activation=activation,padding = 'same',kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary()) 

# fit generator

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 500,
        epochs = 10,
        validation_data = (X_test, Y_test)
)

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
