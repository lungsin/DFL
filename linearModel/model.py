from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

def build_model():
    # Build the model 
    model = Sequential()
    
    model.add(Flatten(input_shape=(28, 28, 1)))
    # model.add(Dropout(rate=0.25))
    # model.add(Dense(128))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model