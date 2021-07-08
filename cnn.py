from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import os

classes = 10
lr = 0.001
batch_size = 128
epochs = 10

def main():
    (X_train, y_train),(X_test, y_test) = cifar10.load_data()
    # floatに型変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # 各画素値を正規化
    X_train /= 255.0
    X_test /= 255.0

    Y_train = to_categorical(y_train, classes)
    Y_test = to_categorical(y_test, classes)

    # モデルの定義
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,input_shape=(32,32,3)))
    model.add(Activation("relu"))

    model.add(Conv2D(32,kernel_size=3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,kernel_size=3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    #model.add(Dropout(1.0))

    model.add(Dense(classes, activation=("softmax")))

    adam = Adam(learning_rate=lr)

    model.compile(
        optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"]
    )

    history = model.fit(
        X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1
    )

if __name__ == '__main__':
    main()