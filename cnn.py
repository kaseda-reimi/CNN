from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

classes = 10
lr = 0.001
batch_size = 128
epochs = 10

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train  = x_train.reshape(60000, 28, 28, 1)
    x_test   = x_test.reshape(10000, 28, 28, 1)
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255
    y_train  = to_categorical(y_train, classes)
    y_test   = to_categorical(y_test, classes)

    # モデルの定義
    model = Sequential()

    model.add(Conv2D(32,(3, 3),input_shape=(28,28,1)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3, 3)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    #model.add(Dropout(1.0))
    model.add(Dense(classes, activation=("softmax")))

    adam = Adam(learning_rate=lr)

    model.compile(
        optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1
    )

    score = model.evaluate(x_test, y_test, verbose=1)
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss,     marker='.', label='loss')
    plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("cnn_learning.png")

if __name__ == '__main__':
    main()