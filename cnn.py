from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.datasets import cifar10

nb_classes = 10
lr = 0.001
batch_size = 32
nb_epoch = 100

def main():
    (X_train, y_train),(X_test, y_test) = cifar10.load_data()
    # floatに型変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # 各画素値を正規化
    X_train /= 255.0
    X_test /= 255.0

    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    # モデルの定義
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation="relu",input_shape=(32,32,3)))
    model.add(Conv2D(32,kernel_size=3,activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,kernel_size=3,activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(1.0))

    model.add(Dense(nb_classes, activation='softmax'))

    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"]
    )

    history = model.fit(
        X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.1
        )
    
if __name__ == '__main__':
    main()