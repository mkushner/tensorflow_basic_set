from keras.models import Sequential
from keras.layers import Dense
from tensorflow import set_random_seed
import numpy as np
import datetime

print (f"MAIN: init: {datetime.datetime.now()}")
# restore results 
np.random.seed(0)
set_random_seed(0)


# dataset load 
dataset = np.loadtxt(r"dataset_file_location", delimiter=",")
print ("MAIN: csv loaded")

# parameters matrix (X) - target output (Y)
X, Y = dataset[:,0:8], dataset[:,8]
print ("NP: matrix build finished")

# create model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu')) # input layer input_dim
model.add(Dense(15, activation='relu')) # 15-8-10 # 30-16-20
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))     
model.add(Dense(1, activation='sigmoid'))
print ("KERAS: NN build finished")

# gradient compilation
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
print ("KERAS: model compilation finished")

# train set
model.fit(X, Y, epochs = 100, batch_size=10)
print ("KERAS: model training finished")

model.save(r"runmodel.h5")

# evaluate
scores = model.evaluate(X, Y)
print (f"KERAS: model evaluation: {model.metrics_names[1]}, {scores[1]*100}")
print (f"MAIN: end: {datetime.datetime.now()}")

# predict
pred = [[1.0, 103.0, 30.0, 38.0, 83.0, 43.3, 0.183, 33.0]]
pred_array = np.array(pred)

print (f"NP: Model shape: {pred_array.shape}")
print (f"KERAS: Prediction: {model.predict(pred_array)}")