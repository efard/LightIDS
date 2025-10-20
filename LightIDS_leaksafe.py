

print("=251017=================================================================")
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())
print("========================================================================")

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import os, csv, time, pathlib
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report,confusion_matrix
from generator_classifier import DataGenerator, FLOW_to_matrix

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

###################################
#Dataset 2018 time-based leakage-safe splitting
###################################
# Train set
base_path = "<ADDRESS_TO_DATASET_FOLDER>"
subpaths = [
    "/Thursday-01/Benign/",
    "/Thursday-01/Infilteration/",
    "/Friday-02/Benign/",
    "/Friday-02/Bot/",
    "/Wednesday-14/Benign/",
    "/Wednesday-14/FTP-BruteForce/",
    "/Wednesday-14/SSH-Bruteforce/",
    "/Thursday-15/Benign/",
    "/Thursday-15/DoS attacks-GoldenEye/",
    "/Thursday-15/DoS attacks-Slowloris/",
    "/Friday-16/Benign/",
    "/Tuesday-20/Benign/",
    "/Tuesday-20/DDoS attacks-LOIC-HTTP/",
    "/Wednesday-21-tmp/Benign/"
]

subpaths_val = [
    "/Thursday-22/Benign/"
]

subpaths_test = [
    "/Friday-23/Benign/",
    "/Friday-23/Brute Force -Web/",
    "/Friday-23/Brute Force -XSS/",
    "/Friday-23/SQL Injection/",
    "/Friday-23/Web/",
    "/Wednesday-28-tmp/Benign/"
]

#CIC-IDS2017
#base_path = "/home/<ADDRESS_TO_DATASET_FOLDER>"
#subpaths = [
#    "/vectorize_friday/attack_bot/",
#    "/vectorize_friday/attack_DDOS/",
#    "/vectorize_friday/attack_portscan/",
#    "/vectorize_friday/benign3/",
#    "/vectorize_monday/attack/",
#    "/vectorize_monday/benign3/",
#    "/vectorize_tuesday/attack/",
#    "/vectorize_tuesday/benign3/"
#]

#subpaths_val = [
#    "/vectorize_wednesday/attack/",
#    "/vectorize_wednesday/benign3/"
#]

#subpaths_test = [
#    "/vectorize_thursday/attack_infilteration/",
#    "/vectorize_thursday/attack_webattacks/",
#    "/vectorize_thursday/benign3/"
#]

intemp      = "?".join([base_path + sub for sub in subpaths])
intemp_val  = "?".join([base_path + sub for sub in subpaths_val])
intemp_test = "?".join([base_path + sub for sub in subpaths_test])

###################################
#Parameter initialization
###################################
input_path      = intemp
input_path_val  = intemp_val
input_path_test = intemp_test
flow_size = 100
pkt_size = 200
class_num = 2
seed = 7150
np.random.seed(seed)

batch_size = 256
params = { 'dim_x' : flow_size,
           'dim_y' : pkt_size,
           'batch_size': batch_size,
           'shuffle' : True,
           'path' : input_path}
params_val = { 'dim_x' : flow_size,
           'dim_y' : pkt_size,
           'batch_size': batch_size,
           'shuffle' : True,
           'path' : input_path_val}
params_test = { 'dim_x' : flow_size,
           'dim_y' : pkt_size,
           'batch_size': batch_size,
           'shuffle' : True,
           'path' : input_path_test}

list_path      = input_path.split("?")
list_path_val  = input_path_val.split("?")
list_path_test = input_path_test.split("?")
pp      = list()
pp_val  = list()
pp_test = list()

for k in range(0, len(list_path)):
    pp.append(os.listdir(list_path[k]))
for l in range(0, len(list_path_val)):
    pp_val.append(os.listdir(list_path_val[l]))
for m in range(0, len(list_path_test)):
    pp_test.append(os.listdir(list_path_test[m]))

print("Train folders:", len(list_path), "No of files in each folder:", [len(entries) for entries in pp])
print("Validation folders: ", len(list_path_val), "No of files in each folder:", [len(entries) for entries in pp_val])
print("Test folders:", len(list_path_test), "No of files in each folder:", [len(entries) for entries in pp_test])

###################################
#Randomize train, validation, and test sets
###################################
partition2 = list()
for k in range(0, len(pp)):
    for j in range(0, len(pp[k])):
        partition2.append(list_path[k] + pp[k][j])

np.random.seed(seed)
np.random.shuffle(partition2)

partition2_val = list()
for k in range(0, len(pp_val)):
    for j in range(0, len(pp_val[k])):
        partition2_val.append(list_path_val[k] + pp_val[k][j])

np.random.seed(seed)
np.random.shuffle(partition2_val)

partition2_test = list()
for k in range(0, len(pp_test)):
    for j in range(0, len(pp_test[k])):
        partition2_test.append(list_path_test[k] + pp_test[k][j])

np.random.seed(seed)
np.random.shuffle(partition2_test)

trainIDs = partition2
validIDs = partition2_val
testIDs  = partition2_test

print ("\nDataset size:   ", len(partition2 + partition2_val + partition2_test))
print ("Train size:     ", len(trainIDs))
print ("Validation size:", len(validIDs))
print ("Test size:      ", len(testIDs))

init = tf.keras.initializers.truncated_normal()
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
CSV = callbacks.CSVLogger("log-classifier.csv" , separator=",", append=False)
TFboard = callbacks.TensorBoard(log_dir='./logs_newB',write_graph=True, write_images=True)

###################################
#Define Model: 5000, 2
###################################
main_input = Input(shape=(flow_size,pkt_size,), dtype='float32', name='main_input')
x1 = LSTM(50, kernel_initializer=init, return_sequences=True, name='LSTM1')(main_input)
x2 = Flatten()(x1)
y = Dense(class_num, activation = "softmax",kernel_initializer=init, name = 'classifier')(x2)
model = Model(inputs=main_input , outputs=[y])

print("Model Summary========================================================================")
print(model.summary())
print("=====================================================================================")

###################################
#Loss changes per batch
###################################
class LossHisotry(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

###################################
#Compile Model
###################################
model.compile(loss={'classifier':'binary_crossentropy'},optimizer='Nadam',metrics=['acc'])
history = LossHisotry()

###################################
#Train Model
###################################
training_generator   = DataGenerator(**params).generate(list_IDs = trainIDs)
validation_generator = DataGenerator(**params_val).generate(list_IDs = validIDs)

model.fit(
    x=training_generator,
    batch_size=batch_size,
    epochs=1,
    verbose=0,
    callbacks=[CSV, TFboard, earlyStopping, history],
    validation_data=validation_generator,
    shuffle=True,
    steps_per_epoch=len(trainIDs)//batch_size,
    validation_steps=len(validIDs)//batch_size,
    validation_freq=1
)

###################################
#Test Model on test set
###################################
PR = list()
GT = list()
EX = list()
counter = 0

for sample in testIDs:
    DataX, DataY = FLOW_to_matrix(ID = sample, path = input_path)
    GT.append(DataY.argmax())
    start_time = time.time()
    pr = model.predict(DataX, verbose=0)
    EX.append((time.time() - start_time))
    PR.append(pr.argmax())
    counter = counter + 1

###################################
#Print confusion matrix
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX) , np.std(EX)))
print(classification_report(y_true=GT , y_pred=PR))
print(confusion_matrix(y_true=GT , y_pred=PR))