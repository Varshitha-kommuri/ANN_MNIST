THIS IS A BASIC MACHINE LEARNING PROJECT THAT BUILDS ARTIFICIAL NEURAL NETWORK (ANN) TO PREDICT DIGITS (0-9) TRAINED AND TESTED USING WELL-KNOWN MNIST DATASET.

It will be convinient to broadly break down this project as a SIX step process.

STEP 0: Download the MNIST dataset :

# To , work on the MNIST dataset , you can either download the csv files uploaded in this repository which consist    # of 10,000 training examples and 350 testing examples.
# Or feel free to use MNIST datasets available online, eventhough the  basic process remains same but you should 
# be careful of the path specification, and size of the images.

# ---

STEP 1: Initialize parameters randomly :

#     W1 = np.random.randn(n1,n0)
#     b1 = np.zeros((n1,1))
#     W2 = np.random.randn(n2,n1)
#     b2 = np.zeros((n2,1))
# ---

STEP 2: Forward propagation:
    
#     Z1 = W1 * X + b1
#     A1 = f(Z1) 

#     Z2 = W2 * A1 + b2
#     A2 = softmax(Z2).

f(x) can be basically reLU / tanh but in this project I have considered using tanh, but feel free to experiment around other functions.


# ---

STEP 3: Cost function:

#It is a multi-class classification problem.

#  So, n= no.of nodes in o/p layer.
#      m= no.of observations.

#   loss = -(sigma i=k to n)[ yk * log(ak)]
#   where yk is the one-hot notation of true y values and ak is the predictions made.

#   cost = -(1/m)(i=1->m)(k=1->n)[yk*log(ak)]

# ---

STEP 4 : Back propogation:
   
#    dZ2 = (A2 - Y)
#    dW3 = (1/m)*dZ2*A1.T
#    db2 = (1/m) *sum(dZ2,1)
  
#    dZ1 = W2.T * dZ2 * f1 prime (Z1)
#    dW1 = (1/m)* dZ1*X.T
#    db1 = (1/m)*sum(dZ1,1)

# ---

STEP 5: Updating parameters:

#    W2 = W2 - alpha * (der_cost_W2)
#    b2 = b2 - alpha * (der_cost_b2)

#    W1 = W1 - alpha * (der_cost_W1)
#    b1 = b1 - alpha * (der_cost_b1)
# """
