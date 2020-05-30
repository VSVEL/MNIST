import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as k


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test , num_classes = 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape)
print(x_train.shape[0])
print(x_test.shape[0])


num_classes = 10
epochs = 15
model = Sequential()
model.add(Conv2D(32,(3,3,), activation='relu', input_shape = input_shape))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.SGD(lr=0.01, momentum = 0.9), metrics = ['accuracy'])


hist = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose = 1, validation_data=(x_test,y_test))

model.save('mnist.h5')
score = model.evaluate(x_test,y_test,verbose=0)
print(score[0])
print(score[1])
