import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Import dataset 
data = pd.read_csv('Input n Capacity.csv')

#Format the data
data = data.drop(labels=['SampleId'], axis=1)
dataset = data[~data.isin(['?'])]
dataset = dataset.dropna(axis=0)
dataset = dataset.apply(pd.to_numeric)

#Mathematical summary of dataset
dataset.describe()

#Standard deviation for all attributes
features=['Cycle','Time Measured(Sec)','Voltage Measured(V)','Current Measured','Temperature Measured']
dataset[features].std().plot(kind='bar', figsize=(8,6), title="Features Standard Deviation")

#Figure below shows capacity changes of the observed samples over the charge-discharge cycles.
plot_df = dataset.loc[(dataset['Cycle']>=1),['Cycle','Capacity(Ah)']]
sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))
plt.plot(plot_df['Cycle'], plot_df['Capacity(Ah)'])
#Draw threshold
plt.plot([0.,len(dataset)], [1.4, 1.4])
plt.ylabel('Capacity(Ah)')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
plt.xlabel('Cycle')
plt.title('Discharge Observed')
plt.show()

#Calculate the State of Health:
import warnings
warnings.filterwarnings('ignore')
attributes=['Cycle','Time Measured(Sec)','Capacity(Ah)']
soh=data[attributes]
C=soh['Capacity(Ah)'][0]
#print(soh['Capacity'][0])
HoS=0
for i in range(len(soh)):
    soh['SoH']=(soh['Capacity(Ah)'])/C
print(soh.head(5))


#SoH Vs Cycle - Discharge Observed
plot_df = soh.loc[(soh['Cycle']>=1),['Cycle','SoH']]
sns.set_style("white")
plt.figure(figsize=(8, 5))
plt.plot(plot_df['Cycle'], plot_df['SoH'])
#Draw threshold
#plt.plot([0.,len(dataset)], [0.70, 0.70])
plt.ylabel('SOH')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
plt.xlabel('Cycle')
plt.title('Discharge Observed')
plt.show()

#Discharge observed pertaining to Time Measured and Voltage Measured
plot_df = dataset.loc[(dataset['Cycle']>=1),['Time Measured(Sec)','Voltage Measured(V)']]
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['Time Measured(Sec)'], plot_df['Voltage Measured(V)'])
plt.ylabel('Voltage')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
#adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Time')
plt.title('Discharge')
plt.show()

#Plotting Temperature Resistance over cycles
plot_df = dataset.loc[(dataset['Cycle']>=1),['Cycle','Temperature Measured']]
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['Cycle'], plot_df['Temperature Measured'])
plt.ylabel('Temperature')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
#adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Cycle')
plt.title('Temperature Resistance')
plt.show()

#Plotting Current observed over cycles
plot_df = dataset.loc[(dataset['Cycle']>=1),['Cycle','Current Measured']]
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['Cycle'], plot_df['Current Measured'])
plt.ylabel('Current')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
#adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Cycle')
plt.title('Observed Current Changes')
plt.show()

#Potting Voltage recorded in training set and testing set
test = pd.read_csv('testing_data.csv')
train = pd.read_csv('training_data.csv')
time12=train['Time Measured(Sec)'][280]
print(time12)
#test['Time Measured(Sec)']=test['Time Measured(Sec)']+time12
plot_df = test.loc[(test['Cycle']),['Time Measured(Sec)','Voltage Measured(V)']]
plot_charge=train.loc[(train['Cycle']),['Time Measured(Sec)','Voltage Measured(V)']]
sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['Time Measured(Sec)'], plot_df['Voltage Measured(V)'])
plt.plot(plot_charge['Time Measured(Sec)'], plot_charge['Voltage Measured(V)'])
#Draw threshold
#plt.plot(dis['cycle'], dis['limt']) 'g'
plt.ylabel('Voltage Observed')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
#adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Cycle')
plt.title('Orange line: Train and  Blue line: Test')
plt.show()

#Charge and discharge in one cycle
fig, ax1 = plt.subplots()
sns.set_style("white")
plot_df= dataset.loc[(dataset['Cycle'] < 10),['Time Measured(Sec)','Current Measured']]
#plt.plot([126, 127], color="black")
plot_df1 = dataset.loc[(dataset['Cycle'] < 10),['Time Measured(Sec)','Voltage Measured(V)']]
color = 'tab:red'
ax1.set_xlabel('Time (Seconds)')
ax1.set_ylabel('Voltage', color=color)
ax1.plot(plot_df1['Time Measured(Sec)'], plot_df1['Voltage Measured(V)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Current', color=color)  # we already handled the x-label with ax1
ax2.plot(plot_df['Time Measured(Sec)'], plot_df['Current Measured'],'-', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.set_size_inches(10, 5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
# vertical black line split the graph between charge and discharge operation.

#The figure below explain (the cylces) that after long-term, repeated charges and discharges, the lifetime of the Li-ion battery will be gradually reduced due to some irreversible reactions.
plot_df = dataset.loc[(dataset['Cycle']<10),['Time Measured(Sec)','Voltage Measured(V)']]
plot_df1 = dataset.loc[(dataset['Cycle'] > 10) & (dataset['Cycle'] < 100),['Time Measured(Sec)','Voltage Measured(V)']]
plot_df2 = dataset.loc[(dataset['Cycle'] > 100) & (dataset['Cycle'] < 120),['Time Measured(Sec)','Voltage Measured(V)']]
plot_df3 = dataset.loc[(dataset['Cycle'] > 120) & (dataset['Cycle'] < 150),['Time Measured(Sec)','Voltage Measured(V)']]
plot_df4 = dataset.loc[(dataset['Cycle'] > 150) & (dataset['Cycle'] <= 160),['Time Measured(Sec)','Voltage Measured(V)']]
sns.set_style("white")
plt.figure(figsize=(20, 8))
plt.plot(plot_df['Time Measured(Sec)'], plot_df['Voltage Measured(V)'],color='red')
plt.plot(plot_df1['Time Measured(Sec)'], plot_df1['Voltage Measured(V)'],'-',color='blue')
plt.plot(plot_df2['Time Measured(Sec)'], plot_df2['Voltage Measured(V)'],color='green')
plt.plot(plot_df3['Time Measured(Sec)'], plot_df3['Voltage Measured(V)'],'-',color='black')
plt.plot(plot_df4['Time Measured(Sec)'], plot_df4['Voltage Measured(V)'],color='Orange')
plt.ylabel('Voltage')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
#adf.scaled[1.0] = '%m-%d-%Y'
plt.xlabel('Time')
plt.title('Discharge')
print(" cycle 10 with color:red\n cycle 100 with color: blue\n cycle 120 with color:green\n cycle 150 with color: black\n cycle 160 with color: orange ")
plt.show()