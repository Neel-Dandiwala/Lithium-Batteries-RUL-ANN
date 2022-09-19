import sys
import pandas as pd
import numpy as np
import sklearn
import keras
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Keras: {}'.format(keras.__version__))

#Import dataset
dataset = pd.read_csv('Input n Capacity.csv')

#Making alterations
dataset = dataset.drop(labels=['SampleId'], axis=1)
print( 'Shape of DataFrame: {}'.format(dataset.shape))
print (dataset.loc[1])
data = dataset[~dataset.isin(['?'])]
data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)
print(data.dtypes)
data.describe()

#Visualising the attributes 
data.hist(figsize = (12, 12))
plt.show()
plt.pause(3)
plt.close('all')

features=["Voltage Measured(V)","Current Measured","Capacity(Ah)","Temperature Measured"]
data[features].std().plot(kind='bar', figsize=(8,6), title="Features Standard Deviation")

#Splitting the data
X = np.array(data.iloc[:,0:5].values)
y = np.array(data.iloc[:,5].values)

#Levelling the X attributes
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

#Creating testing and training sets
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size = 0.3)

#Converting vector (integers) to binary class matrix
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

#Model Creation
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from keras.layers import Dropout
from keras import regularizers

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=5, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.15))
    model.add(Dense(7, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(3,kernel_initializer='uniform', activation='sigmoid'))
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_absolute_error','accuracy'])
    return model

model = create_model()

print(model.summary())

#Fitting the model 
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=1)

#Plotting the obtained & predicted Mean Absolute Error
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('MAE')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.pause(3)
plt.close()

#Plotting the obtained & predicted Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
plt.pause(3)
plt.close()

#Plotting the obtained & predicted Mean Squared Error
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
plt.pause(3)
plt.close()

#Plotting Cycle vs Capacity
plot_df = data.loc[(data['Cycle']>=1),['Cycle','Capacity(Ah)']]
sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))
plt.plot(plot_df['Cycle'], plot_df['Capacity(Ah)'])
#Draw threshold
plt.plot([0.,len(data)], [1.4, 1.4])
plt.ylabel('Capacity(Ah)')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
plt.xlabel('Cycle')
plt.title('RUL Prediction')
plt.show()
plt.pause(3)
plt.close()

#Normalising and plotting Cycle Vs Capacity 
capacity = np.array(data.iloc[:,5].values)
cycle = np.array(data.iloc[:,0].values)
normalizedCycle = (cycle-min(cycle))/(max(cycle)-min(cycle))
normalizedCapacity = (capacity-min(capacity)) / (max(capacity) - min(capacity))
plt.plot(normalizedCapacity)
plt.plot(normalizedCycle)
plt.title('RUL Prediction')
plt.ylabel('Normalised Capacity')
plt.xlabel('Normalised Cycle')
plt.show()
plt.hist(normalizedCapacity, bins=10)
plt.hist(normalizedCycle, bins=10)
plt.pause(3)
plt.close()


#Plotting the graph of Time Measured Vs Voltage Measured for all cycles
plot_df = data.loc[(data['Cycle']>=1),['Time Measured(Sec)','Voltage Measured(V)']]
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['Time Measured(Sec)'], plot_df['Voltage Measured(V)'])
plt.ylabel('Voltage battery')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
#adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Time Measured(Sec)')
plt.title('Discharge')
plt.show()
plt.pause(3)
plt.close()

#Plotting the observed relationship between Temperature Measured, Voltage Measured and Current Measured
fig, ax1 = plt.subplots()
sns.set_style("white")
plot_df= data.loc[(data['Cycle']==1),['Temperature Measured','Current Measured']]
#plt.plot([126, 127], color="black")
plot_df1 = data.loc[(data['Cycle']==1),['Temperature Measured','Voltage Measured(V)']]
color = 'tab:red'
ax1.set_xlabel('Temperature Measured')
ax1.set_ylabel('Voltage Measured(V)', color=color)
ax1.plot(data['Time Measured(Sec)']/60, data['Voltage Measured(V)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Current Measured', color=color)  # we already handled the x-label with ax1
ax2.plot(data['Time Measured(Sec)']/60, data['Current Measured'],'-', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.set_size_inches(10, 5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.pause(3)
plt.close()
# vertical black line split the graph between charge and discharge operation.