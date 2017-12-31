
# coding: utf-8

# In[1]:

import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
import pickle
#from six.moves import cPickle as pickle
from six.moves import range
# In[2]:

#read data
no_of_images=42000
dataset = np.ndarray((no_of_images,784), dtype=np.float32)
dataset = genfromtxt('/home/sarthak/Documents/Sarthak_joshi/train.csv', delimiter=',')


#normalizing dataset
dataset=(dataset-127.5)/255.0
    




#taking input of data labels
labels = np.ndarray(no_of_images, dtype=np.int32)
data_labels=genfromtxt('/home/sarthak/Documents/Sarthak_joshi/train_labels.csv',delimiter=',')


# In[7]:

#print(data_labels)


# In[8]:

#dividing dataset into validation set and training set (approx 3:7 ratio)
training_dataset=dataset[0:30000][:]
validation_dataset=dataset[30000:42000][:]
#print(training_dataset.shape,validation_dataset.shape)
#print(training_dataset[0])



#dividing training labels and validation labels
training_labels=data_labels[0:30000]
validation_labels=data_labels[30000:42000]
print(training_labels.shape)



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[11]:

num_labels=10
#reformating labels to 1-hot encoding
def reformat(labels):
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return  labels
training_labels=reformat(training_labels)
validation_labels=reformat(validation_labels)
#print(training_labels.shape,validation_labels.shape)


# In[ ]:

#processing the testing data


# In[12]:

testing_dataset = tf.constant((28000,784), dtype=tf.float32)
testing_dataset= genfromtxt('/home/sarthak/Documents/Sarthak_joshi/test.csv', delimiter=',')
testing_dataset=(testing_dataset-127.5)/255.0
#print (testing_dataset[0])


# In[13]:

#data preprocessing is complete applying learning algorithm (neural network)
batch_size = 128
hidden_layer_size=1024
image_size=28
graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(validation_dataset,tf.float32)
  tf_test_dataset=tf.constant(testing_dataset,tf.float32)
  beta=tf.placeholder(tf.float32)
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_layer_size]))
  biases1 = tf.Variable(tf.zeros([hidden_layer_size]))
  weights2 = tf.Variable(
    tf.truncated_normal([hidden_layer_size, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
 
  layer1_training = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits=tf.matmul(layer1_training,weights2)+biases2  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))+beta*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2))
  
  # Optimizer.
  optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
  train_prediction = tf.nn.softmax(logits)
  layer1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_prediction  = tf.matmul(layer1_valid, weights2) + biases2
  layer1_test=tf.nn.relu(tf.matmul(tf_test_dataset,weights1)+biases1)
  test_prediction=tf.nn.softmax(tf.matmul(layer1_test,weights2)+biases2)  



#this part of code is to used only when the regularization value needs to be found uncomment it to find the value of beta
'''
num_steps = 7000
regul_val = [pow(10, i) for i in np.arange(-4, -1, 0.1)]
accuracy_val = []

for regul in regul_val:
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
      if(step%1000==0):
       print (step)
  
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (training_labels.shape[0] - batch_size)
    # Generate a minibatch.
      batch_data = training_dataset[offset:(offset + batch_size), :]
      batch_labels = training_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta: regul}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
    accuracy_val.append(accuracy(valid_prediction.eval(), validation_labels))


# In[34]:

print(accuracy_val)
print (regul_val)


# In[35]:

plt.semilogx(regul_val, accuracy_val)
plt.grid(True)
plt.title('Test accuracy by regularization')
plt.show()

'''
# In[14]:

num_steps = 50000

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (training_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = training_dataset[offset:(offset + batch_size), :]
    batch_labels = training_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,beta:0.000509}
    _, l, predictions,w1,w2,b1,b2 = session.run(
      [optimizer, loss, train_prediction,weights1,weights2,biases1,biases2], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), validation_labels))
  print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), validation_labels))
  pickle_file = "MNIST.pkl"



  
  f = open(pickle_file, 'wb')
  save = {
  'w1': w1,
  'w2': w2,
  'b1': b1,
  'b2': b2
  }
  pickle.dump(save,f)
  f.close()
 # except Exception as e:
 #  print('Unable to save data to', pickle_file, ':', e)
 #  raise
