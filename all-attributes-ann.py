import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Keras: {}'.format(keras.__version__))

#Importin and formatting the data
dataset = pd.read_csv('Input n Capacity.csv')
dataset = dataset.drop(labels=['SampleId'], axis=1)
data = dataset[~dataset.isin(['?'])]
data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)

#Creating the model 
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from keras.layers import Dropout
from keras import regularizers
# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=1, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.15))
    model.add(Dense(7, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(3,kernel_initializer='uniform', activation='sigmoid'))
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_absolute_error','accuracy'])
    return model

model = create_model()
print(model.summary())


#_____________________________________________________________________First Attribute: Cycle______________________________________________________________________________________


Cycle_X = np.array(data.iloc[:,0].values)
y = np.array(data.iloc[:,5].values)
Cycle_X = Cycle_X.astype('float32')
mean = Cycle_X.mean(axis=0)
Cycle_X -= mean
std = Cycle_X.std(axis=0)
Cycle_X /= std

# create X and Y datasets for training
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(Cycle_X, y, random_state=42, test_size = 0.3)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=1)

# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Cycles - Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Cycle - Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


#_____________________________________________________________________Second Attribute: Time Measured______________________________________________________________________________________

Time_X = np.array(data.iloc[:,1].values)
Time_X = Time_X.astype('float32')
y = np.array(data.iloc[:,5].values)
mean = Time_X.mean(axis=0)
Time_X -= mean
std = Time_X.std(axis=0)
Time_X /= std

# create X and Y datasets for training
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(Time_X, y, random_state=42, test_size = 0.3)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=1)

# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Time - Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Time - Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#_____________________________________________________________________Third Attribute: Voltage Measured______________________________________________________________________________________

Voltage_X = np.array(data.iloc[:,2].values)
Voltage_X = Voltage_X.astype('float32')
y = np.array(data.iloc[:,5].values)
mean = Voltage_X.mean(axis=0)
Voltage_X -= mean
std = Voltage_X.std(axis=0)
Voltage_X /= std

# create X and Y datasets for training
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(Voltage_X, y, random_state=42, test_size = 0.3)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=1)

# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Voltage - Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Voltage - Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#_____________________________________________________________________Fourth Attribute: Current Measured______________________________________________________________________________________

Current_X = np.array(data.iloc[:,3].values)
Current_X = Current_X.astype('float32')
y = np.array(data.iloc[:,5].values)
mean = Current_X.mean(axis=0)
Current_X -= mean
std = Current_X.std(axis=0)
Current_X /= std

# create X and Y datasets for training
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(Current_X, y, random_state=42, test_size = 0.3)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=1)

# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Current - Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Current - Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#_____________________________________________________________________Fifth Attribute: Temperature Measured______________________________________________________________________________________

Temperature_X = np.array(data.iloc[:,3].values)
Temperature_X = Temperature_X.astype('float32')
y = np.array(data.iloc[:,5].values)
mean = Temperature_X.mean(axis=0)
Temperature_X -= mean
std = Temperature_X.std(axis=0)
Temperature_X /= std

# create X and Y datasets for training
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(Temperature_X, y, random_state=42, test_size = 0.3)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=1)

# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Temperature - Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Temperature - Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()