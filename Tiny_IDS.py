
# In[1]:


# QUANTIZE JUST BY tf-nightly-gpu! But Inference by tensorflow-gpu is OK!

# In[2]:


import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import os
import csv
import time
import pathlib
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report,confusion_matrix
from tflite_runtime.interpreter import Interpreter
from generator_classifier import DataGenerator
from generator_classifier import FLOW_to_matrix
import tflite_runtime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

print(tflite_runtime.__version__)
print(tf.__version__)
tf.config.list_physical_devices()


# In[3]:


intemp = "/media/fard/Media/ISCX_IDS_2017/vectorize_thursday/attack_infilteration/"
intemp += "?/media/fard/Media/ISCX_IDS_2017/vectorize_thursday/attack_webattacks/"
intemp += "?/media/fard/Media/ISCX_IDS_2017/vectorize_thursday/benign3/"


# In[4]:


input_path = intemp
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


# In[5]:


###################################
#Load saved dataset and split it
###################################
with open("test_98_new.txt", "r") as file:
    partition2 = eval(file.readline())
    
cut = int(len(partition2)*0.8)
trainvalidIDs = partition2[:cut]
testIDs = partition2[cut:]


# In[6]:


###################################
#Split valid set to train set & validation set
###################################
cut2 = int(len(trainvalidIDs)*0.8)
trainIDs = trainvalidIDs[:cut2]
validIDs = trainvalidIDs[cut2:]
print ("train size:     ", len(trainIDs))
print ("validation size:", len(validIDs))
print ("test size:      ", len(testIDs))


# In[7]:


init = tf.keras.initializers.truncated_normal()

earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
CSV = callbacks.CSVLogger("log-classifier.csv" , separator=",", append=False)
TFboard = callbacks.TensorBoard(log_dir='./logs_newB',write_graph=True, write_images=True)


# In[8]:


###################################
#Define Model: 5000, 2
###################################
main_input = Input(shape=(flow_size,pkt_size,), dtype='float32', name='main_input')

x1 = LSTM(50, kernel_initializer=init, return_sequences=True, name='LSTM1')(main_input)

x2 = Flatten()(x1)

y = Dense(class_num, activation = "softmax",kernel_initializer=init, name = 'classifier')(x2)

model = Model(inputs=[main_input] , outputs=[y])
print(model.summary())


# In[9]:


class LossHisotry(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# In[10]:


###################################
#Compile Model
###################################
model.compile(loss={'classifier':'binary_crossentropy'},optimizer='Nadam',metrics=['acc'])
history = LossHisotry()


# In[11]:


###################################
#Train Model
###################################
training_generator   = DataGenerator(**params).generate(list_IDs = trainIDs)
validation_generator = DataGenerator(**params).generate(list_IDs = validIDs)

model.fit(
    x=training_generator,
    batch_size=batch_size,
    epochs=2,
    verbose=1,
    callbacks=[CSV, TFboard, earlyStopping, history],
    validation_data=validation_generator,
    shuffle=True,
    steps_per_epoch=len(trainIDs)//batch_size,
    validation_steps=len(validIDs)//batch_size,
    validation_freq=1
)


# In[12]:


###################################
#Test Model on test set
###################################
PR = list()
GT = list()
EX = list()
counter = 0

for sample in testIDs:
    print ("%.3f percent!" % float(100*float(counter+1)/float(len(testIDs))))
    DataX, DataY = FLOW_to_matrix(ID = sample, path = input_path)
    GT.append(DataY.argmax())
    start_time = time.time()
    pr = model.predict(DataX)
    EX.append((time.time() - start_time))
    PR.append(pr.argmax())
    counter = counter + 1  


# In[13]:


###################################
#Print confusion matrix
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX) , np.std(EX)))
print(classification_report(y_true=GT , y_pred=PR))
print(confusion_matrix(y_true=GT , y_pred=PR))


# In[14]:


###################################
# Save Model
###################################
model.save("L_5000_2.h5")


# In[15]:


###################################
#Convert saved model to tflite
###################################
run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = flow_size
INPUT_SIZE = pkt_size
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("./tflite_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"L_5000_2.tflite"
tflite_model_file.write_bytes(tflite_model)


# In[16]:


###################################
#Interpret tflite model
###################################

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./tflite_models/L_5000_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input and output indexes.
input_index  = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


# In[17]:


###################################
#Test model on input data
###################################
# Run inference for test set
PR1 = list()
GT1 = list()
EX1 = list()
counter = 0
DataX_list = list()

for sample in testIDs:
    DataX,DataY = FLOW_to_matrix(ID = sample, path = input_path)
    DataX_fp32 = np.float32(DataX)
    GT1.append(DataY.argmax())
    
    interpreter.set_tensor(input_details[0]['index'], DataX_fp32)
    start_time = time.time()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    EX1.append((time.time() - start_time))
    PR1.append(output_data.argmax())
    print(counter+1, '/', len(testIDs), DataY.argmax(), output_data.argmax())
    
    DataX_list.append(DataX_fp32)
    
    counter = counter + 1
    
    # Clean up internal states.
    interpreter.reset_all_variables()


# In[18]:


###################################
#Print confusion matrix for tflite model
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX1) , np.std(EX1)))
print(classification_report(y_true=GT1 , y_pred=PR1))
print(confusion_matrix(y_true=GT1 , y_pred=PR1))


# In[20]:


###################################
#Convert using quantization
###################################
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#Change data from list to array
DataX_array = np.array(DataX_list)
DataX_reshaped = np.reshape(DataX_array, (len(testIDs), flow_size, pkt_size))
ids_ds = tf.data.Dataset.from_tensor_slices(DataX_reshaped).batch(1)

def representative_data_gen():
    for input_value in ids_ds.take(100):
        yield [input_value]

converter.representative_dataset = representative_data_gen

#ensure that the converted model is fully quantized
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

converter.experimental_new_converter = True
#Finally, convert the model to TensorFlow Lite format:
tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir/"L_5000_2_INT.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)


# In[21]:


###################################
#Interpret INT8 tflite model
###################################

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./tflite_models/L_5000_2_INT.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[22]:


###################################
#Test model on input data for INT8 tflite model
###################################

# Generate random input data
input_shape = input_details[0]['shape']
input_random_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)

# Run inference for test set
PR2 = list()
GT2 = list()
EX2 = list()
counter = 0
DataX_list = list()

for sample in testIDs:
    DataX,DataY = FLOW_to_matrix(ID = sample, path = input_path)
    
    GT2.append(DataY.argmax())
    interpreter.set_tensor(input_details[0]['index'], DataX)  #input_random_data
    start_time = time.time()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    EX2.append((time.time() - start_time))
    PR2.append(output_data.argmax())
    print(counter, '/', len(testIDs), "\t", DataY.argmax(), output_data.argmax())
        
    counter = counter + 1
    # Clean up internal states.
    interpreter.reset_all_variables()


# In[23]:


###################################
#Print confusion matrix for tflite model
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX2) , np.std(EX2)))
print(classification_report(y_true=GT2 , y_pred=PR2))
print(confusion_matrix(y_true=GT2 , y_pred=PR2))


# In[ ]:


fh = open("results.txt", "w")
for i in range(len(PR2)):
    fh.write("Index: " + str(i) + "\t" + "Predicted/True: " + str(PR2[i]) + " / " + str(GT2[i]) + "\n")
fh.close()

