
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow.keras.utils import to_categorical
import np_utils
import os


flow_size = 100
pkt_size = 200
class_num = 2


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, path = "/Final_data/FINAL_subsample_vector/" , dim_x = flow_size , dim_y = pkt_size , batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.path = path

  def generate(self, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y= self.__data_generation(list_IDs_temp,path=self.path)

              yield X,y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp , path):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y))
      y = np.empty((self.batch_size, 2))
      #W = np.empty((self.batch_size))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          X[i, :, :],y[i,:] = FLOW_to_matrix(ID = ID , path = path)


      Y = dict(zip(['classifier','main_output'], [y,X]))
      return X , y

def FLOW_to_matrix(ID,path):
    X = list()
    Y = list()
    tmp_c = 0
    for filename in [ID]:

        with open(filename) as flowfile:
            if filename.__contains__("BENIGN"):
                Y.append(0)
            else:
                Y.append(1)
            flowmatrix = list()
            counter = 0
            firsttime = 0
            for line in flowfile:
                if counter == flow_size:
                    break
                line = line[:-1].split(",")
                for i in range(len(line)):
                    line[i] = float(line[i])

                if line[0] < 0:
                    line[0] = 0.0
                if line[0] > 0.5:
                    line[0] = 1
                else:
                    line[0] = line[0] / 0.5
                #append 0 to packets which are smaller than packetsize(max=1500)
                for i in range(pkt_size - len(line)):
                    line.append(0.0)
                line = line[:pkt_size]
                for i in range(11,21):
                    line[i] = 0.0    
                
                flowmatrix.append(line)
                counter += 1
                
            for i in range(flow_size - len(flowmatrix)):
                flowmatrix.append([0] * pkt_size)
            X.append(np.stack(flowmatrix))

    X_train = np.stack(X)
    X_train = X_train *255
    X_train = X_train.astype(np.int8)
    
    Y.append(0)
    Y.append(1)

    unq, ids = np.unique(Y, return_inverse=True)
    #Y_train = np_utils.to_categorical(ids, len(unq))
    Y_train = to_categorical(ids, len(unq))
    Y_train = Y_train.astype(np.int8)
    return X_train, Y_train[0]
