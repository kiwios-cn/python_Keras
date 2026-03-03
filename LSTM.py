from keras.layers import LSTM
model = Sequential()
model.add(Embedding(max_features , 32))
model.add(LSTM(32))

model.add(Dense( 1 , activation = 'sigmoid') )

model.compile(optimizer = 'rmsprop' ,
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(input_train , y_train ,
                    epochs = 10,
                    batch_size = 128,
                    validation_spilt = 0.2)
