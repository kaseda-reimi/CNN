from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import function as fc

x_len = 40
y_len = 6

lr = 0.001
batch_size = 128
epochs = 10

model_path = os.getcwd()+'/cnn_model'

def main():
    input_data, output_data = fc.get_data()
    #x_train  = x_train.reshape(60000, 28, 28, 1)
    #x_test   = x_test.reshape(10000, 28, 28, 1)
    input_data  = input_data.astype('float32')
    output_data   = output_data.astype('float32')
    #正規化
    input_data /= 2
    output_data[:,0] = fc.normalize(output_data[:,0])
    output_data[:,1] = fc.normalize(output_data[:,1])

    x_train, x_test, y_train, y_test = train_test_split(input_data,output_data,test_size=0.1)
    
    # モデルの定義
    model = Sequential()

    model.add(Conv2D(32,(2, 4),input_shape=(y_len,x_len,1)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,(2, 4)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,(2, 4)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    #model.add(Dropout(1.0))
    model.add(Dense(2, activation=("linear")))

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

    model.save(model_path)

if __name__ == '__main__':
    main()