
print("========================================================================")
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())
print("========================================================================")

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import os, csv, time, pathlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from generator_classifier import DataGenerator, FLOW_to_matrix

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

###################################
#Dataset address
###################################
#CSE-CIC-IDS2018
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
    "/Thursday-22/Benign/"
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
#    "/vectorize_tuesday/benign3/",
#    "/vectorize_wednesday/attack/",
#    "/vectorize_wednesday/benign3/",
#    "/vectorize_thursday/attack_infilteration/",
#    "/vectorize_thursday/attack_webattacks/",
#    "/vectorize_thursday/benign3/"
#]

###################################
#Parameter initialization
###################################
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

list_path = input_path.split("?")
pp = list()

for k in range(0, len(list_path)):
    pp.append(os.listdir(list_path[k]))

print("No of folders:", len(list_path))
print("No of files in each folder:", [len(entries) for entries in pp])


partition2 = list()
for k in range(0, len(pp)):
    for j in range(0, len(pp[k])):
        partition2.append(list_path[k] + pp[k][j])

np.random.seed(seed)
np.random.shuffle(partition2)

###################################
#Load saved dataset and split it
###################################
cut = int(len(partition2)*0.8)
trainvalidIDs = partition2[:cut]
testIDs = partition2[cut:]

###################################
#Split valid set to train set & validation set
###################################
cut2 = int(len(trainvalidIDs)*0.8)
trainIDs = trainvalidIDs[:cut2]
validIDs = trainvalidIDs[cut2:]
print ("Dataset size:   ", len(partition2))
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
print(model.summary())

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
validation_generator = DataGenerator(**params).generate(list_IDs = validIDs)

model.fit(
    x=training_generator,
    batch_size=batch_size,
    epochs=100,
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
PROB = list()     # predicted probabilities (for ROC)
counter = 0

for sample in testIDs:
    DataX, DataY = FLOW_to_matrix(ID = sample, path = input_path)
    GT.append(DataY.argmax())
    start_time = time.time()
    pr = model.predict(DataX, verbose=0)
    EX.append((time.time() - start_time))
    PR.append(pr.argmax())
    PROB.append(pr[0][1])         # probability of the positive class (index 1)
    counter = counter + 1

###################################
#Compute, plot, and save ROC/PR, AUPRC, Calibration Curve, Confidence intervals
###################################
GT = np.array(GT)
PROB = np.array(PROB)

# Compute, plot and save ROC curve
fpr, tpr, _ = roc_curve(GT, PROB)
roc_auc = roc_auc_score(GT, PROB)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("ROC curve saved as 'roc_curve.png'")
print(f"AUC = {roc_auc:.3f}")

# Compute, plot and save PR curve
precision, recall, thresholds = precision_recall_curve(GT, PROB)
auprc = average_precision_score(GT, PROB)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall (PR) Curve')
plt.legend(loc='lower left')
plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("PR curve saved as 'pr_curve.png'")
print(f"AUPRC = {auprc:.3f}")

# Compute, plot and save Calibration curve
prob_true, prob_pred = calibration_curve(GT, PROB, n_bins=10, strategy='uniform')
brier = brier_score_loss(GT, PROB)
plt.figure()
plt.plot(prob_pred, prob_true, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("True probability (fraction of positives)")
plt.title(f"Calibration Curve (Brier = {brier:.3f})")
plt.legend(loc="best")
plt.savefig("calibration_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("Calibration curve saved as 'calibration_curve.png'")
print(f"Brier Score = {brier:.3f}")

###################################
#Print confusion matrix
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX) , np.std(EX)))
print(classification_report(y_true=GT , y_pred=PR))
print(confusion_matrix(y_true=GT , y_pred=PR))

###################################
#Save Model
###################################
model.save('L_5000_2.keras')

###################################
#Convert saved model to tflite
###################################
run_model = tf.function(lambda x: model(x))
BATCH_SIZE = 1
STEPS = flow_size
INPUT_SIZE = pkt_size
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "<ADDRESS_TO_MODEL_DIRECTORY>"
model.save(MODEL_DIR)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("./tflite_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"L_5000_2.tflite"
tflite_model_file.write_bytes(tflite_model)

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


###################################
#Print confusion matrix for tflite model
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX1) , np.std(EX1)))
print(classification_report(y_true=GT1 , y_pred=PR1))
print(confusion_matrix(y_true=GT1 , y_pred=PR1))

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

###################################
#Interpret INT8 tflite model
###################################
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./tflite_models/L_5000_2_INT.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

###################################
#Print confusion matrix for tflite model
###################################
print("Average Delay per flow is: %.6f sec +/- %.6f sec" % (np.mean(EX2) , np.std(EX2)))
print(classification_report(y_true=GT2 , y_pred=PR2))
print(confusion_matrix(y_true=GT2 , y_pred=PR2))

###################################
#Save results in a file
###################################
fh = open("results.txt", "w")
for i in range(len(PR2)):
    fh.write("Index: " + str(i) + "\t" + "Predicted/True: " + str(PR2[i]) + " / " + str(GT2[i]) + "\n")
fh.close()
