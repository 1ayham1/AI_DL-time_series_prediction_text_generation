import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    elements_range = len(series)- window_size
    
    X = [series[i:i+window_size] for i in range(elements_range)]
    y = [series[i+window_size] for i in range(elements_range)]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    #https://keras.io/layers/recurrent/
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    
    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model
    
    


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    
    import re
    
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    text = re.sub(r'[^a-zA-Z,!.:;?]', ' ', text)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    elements_range = len(text)- window_size
    
    #Question: How can we include a step size in list comprehension?
    i = 0
    while i < elements_range:
        
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        i = i  +step_size 
    
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    
    #https://keras.io/layers/recurrent/
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    # initialize optimizer
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile model --> make sure initialized optimizer and callbacks - as defined above - are used
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
    
    
    
