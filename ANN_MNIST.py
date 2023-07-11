import numpy as np
import matplotlib.pyplot as plt
import random

"""**LOADING THE DATASETS**"""
# Instead of the paths specified,
# Eg : 'C:\Users\Varshitha\OneDrive\Desktop\train_x.csv'
#  replace them with the paths where the files are located on your local device.

X_train = np.loadtxt(r'C:\Users\Varshitha\OneDrive\Desktop\train_x.csv',delimiter = ',') .T 
Y_train = np.loadtxt(r'C:\Users\Varshitha\OneDrive\Desktop\train_label.csv',delimiter = ',') .T

X_test = np.loadtxt(r'C:\Users\Varshitha\OneDrive\Desktop\test_x.csv',delimiter=',').T
Y_test = np.loadtxt(r'C:\Users\Varshitha\OneDrive\Desktop\test_label.csv',delimiter=',').T

def shape(x,y,x_t,y_t):
  print("Shape of X_train is : {}".format(x.shape))
  print("Shape of Y_train is : {}".format(y.shape))
  print("Shape of X_test is : {}".format(x_t.shape))
  print("Shape of Y_test is : {}".format(y_t.shape))
shape(X_train,Y_train,X_test,Y_test)

"""shape(X_train,Y_train,X_test,Y_test)

X_train = (1000, 784)
  1000  = no.of training instances.
  784   = no.of i/p features.
        = in this case it is the
          (28 x 28) of each digit image.

Y_train = (1000, 10)
  1000  = no.of training instances       
          labels.
  10    = no.of o/p possible.
        = no.of classes.

X_test  = (350, 784)
  350  = no.of testing instances.
  784   = no.of i/p features.
        = in this case it is the
          (28 x 28) of each digit image.

Y_test  = (350, 10)
 1000  = no.of testing instances       
          labels.
  10    = no.of o/p possible.
        = no.of classes.

But we need X_train shape as
(no.of nodes in the i/p layer = no.of features , no.of instances)
and the same applies with the rest.
So, we transpose them."""



# **SHOWING AN EXAMPLE OF A DIGIT.**
# X_train[:,index]== all rows of matrix X_train and specific column specified by the index.
# X_train matrix has 784 rows == i/p features and 1000 columns == each column has a i/p instance.
#So,selecting all rows and a specific column means selecting a specific instance (here a digit ).

'''index = random.randrange(0, X_train.shape[1])
plt.imshow(X_train[:,index].reshape(28,28),cmap='gray')
plt.show()'''


"""**INTIALIZING THE PARAMETERS.**"""

def initializing_parameters(n_x,n_h,n_y):
  w1 = np.random.randn(n_h,n_x) * 0.001
  b1 = np.zeros((n_h,1))

  w2 = np.random.randn(n_y,n_h) * 0.001
  b2 = np.zeros((n_y,1))

  parameters ={
      "w1" : w1,
      "b1" : b1,
      "w2" : w2,
      "b2" : b2
  }
  return parameters

"""**ACTIVATION FUNCTIONS**"""

def tanh(x):
  return np.tanh(x)

def softmax(x):
  expX = np.exp(x)
  return expX/np.sum(expX,axis=0)

def der_tanh(x):
  return (1-np.power(np.tanh(x),2))

"""**FORWARD PROPAGATION**"""

def forward_propagation(x,parameters):
  w1 = parameters["w1"]
  w2 = parameters["w2"]
  b1 = parameters["b1"]
  b2 = parameters["b2"]

  z1 = np.dot(w1,x) + b1
  a1 = tanh(z1)

  z2 = np.dot(w2,a1) + b2
  a2 = softmax(z2)

  forward_res = {
      "z1" : z1,
      "z2" : z2,
      "a1" : a1,
      "a2" : a2
  }
  return forward_res

"""**COST FUNCTION.**"""

def cost_function(a2 , y):
  m = y.shape[1]                #no.of instances.
  loss = y*np.log(a2)           #loss is calculated for one training instances.
  cost = -(1/m)*np.sum(loss)    #Cost is calculated for all the training instances.
  return cost

"""BACK PROPAGATION."""

def back_propagation(x, y, parameters, forward_res):
  w1 = parameters["w1"]
  w2 = parameters["w2"]
  b1 = parameters["b1"]
  b2 = parameters["b2"]

  z1 = forward_res["z1"]
  z2 = forward_res["z2"]
  a1 = forward_res["a1"]
  a2 = forward_res["a2"]

  m = x.shape[1]  # no.of instances.

  dz2 = (a2 - y)
  dw2 = (1/m)* np.dot(dz2,a1.T)
  db2 = (1/m)*np.sum(dz2,axis=1,keepdims =True) # column-wise summation.

  dz1 = np.dot(w2.T,dz2)*der_tanh(a1)
  dw1 = (1/m)*np.dot(dz1,x.T)
  db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)

  gradients ={
      "dw1" : dw1,
      "dw2" : dw2,
      "db1" : db1,
      "db2" : db2
  }
  return gradients

"""**UPDATING PARAMETERS.**"""

def update_parameters(parameters, gradients,learning_rate):
  w1 = parameters["w1"]
  w2 = parameters["w2"]
  b1 = parameters["b1"]
  b2 = parameters["b2"]

  dw1 = gradients["dw1"]
  dw2 = gradients["dw2"]
  db1 = gradients["db1"]
  db2 = gradients["db2"]



  w1 = w1 - learning_rate*dw1
  b1 = b1 - learning_rate*db1
  w2 = w2 - learning_rate*dw2
  b2 = b2 - learning_rate*db2
  parameters = {
      "w1" : w1,
      "b1" : b1,
      "w2" : w2,
      "b2" : b2
      }
  return parameters

"""**TRAINING THE ANN.**"""

def ANN(x,y,n_h, learning_rate,iterations):
  n_x = x.shape[0]
  n_y = y.shape[0]
  n_h = n_h
  parameters = initializing_parameters(n_x,n_h,n_y)

  for i in range(iterations):

    forward_res = forward_propagation(x,parameters)

    cost = cost_function(forward_res["a2"],y)

    gradients = back_propagation(x,y,parameters,forward_res)

    parameters = update_parameters(parameters,gradients,learning_rate)
  return parameters

"""**START.**"""

n_h = 1000
learning_rate = 0.01
iterations = 100
Parameters = ANN(X_train,Y_train,n_h,learning_rate,iterations)


"""**ACCURACY.**"""

def accuracy(x,y,parameters):
  forward_res = forward_propagation(x,parameters)

  a_out = forward_res["a2"]  #a2.shape = (10,1000)
  arr = np.amax(a_out,axis=0,keepdims=True)

  prediction = np.where(a_out==arr)
  prediction = np.array(prediction[0])

  y_true = np.amax(y,axis=0,keepdims=True)
  y_true = np.where(y_true==y)
  y_true = np.array(y_true[0])
  correct_pred=0
  for i in range(0,prediction.shape[0]):
      if(prediction[i]==y_true[i]):
        correct_pred+=1
  accuracy = (100*correct_pred)/y_true.shape[0]

  return prediction,y_true,correct_pred,accuracy


p,tr,c,a = accuracy(X_train,Y_train,Parameters)
print("Shape of prediction array  is : {}".format(p.shape))
print("Shape of true_labels array is : {}".format(tr.shape))
print("Number of correct predictions on training data is : {}".format(c))
print("Accuracy on training_data is : {:.2f} %".format(a))
print()
p_t,tr_t,c_t,a_t =accuracy(X_test,Y_test,Parameters)
print("Shape of prediction array  is : {}".format(p_t.shape))
print("Shape of true_labels array is : {}".format(tr_t.shape))
print("Number of correct predictions on test data is : {}".format(c_t))
print("Accuracy on test data is : {:.2f} %".format(a_t))


# ** TESTING ** 
index = random.randrange(0,X_test.shape[1])  #generating a random testing instance from test data.
plt.imshow(X_test[:,index].reshape(28,28),cmap='gray')  #displaying the digit.
plt.show()

forward_res = forward_propagation(X_test[:,index].reshape(X_test.shape[0],1),Parameters)
a_out = forward_res["a2"]                      #a2.shape = (10,1)

'''np.amax() gives maximum value in the a_out.
Row value of the maximum value in the a_out as it will be our digit predicted by the model,
so we put axis = 0 , it searches for the maximum value column wise.'''
arr = np.amax(a_out,axis=0,keepdims=True)

'''np.where() gives the row and column of the value which satisfies the specified condition.
So, we are checking for the row and column where the maximum value of predictions (arr)
will be equal to the a_out.'''
prediction = np.where(a_out==arr)
prediction = np.array(prediction[0]) #But as we only want row we take prediction[0]

print("Prediction of the model is : {}".format(prediction))




