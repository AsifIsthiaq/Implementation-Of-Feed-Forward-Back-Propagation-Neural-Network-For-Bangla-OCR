####################### IMPORTING LIABRARIES ##################################
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

####################### WEIGHT INITIALIZATION #################################
def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    print("Weights::",weights)
    print("Weights:::",tf.Variable(weights))
    return tf.Variable(weights)

######################## FORWARD PROPAGATION ##################################
def forwardprop(X, w_1, w_2):
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  
    yhat = tf.matmul(h, w_2)  
    return yhat

############################ LODING DATASET ###################################
def load_dataset():    
    dataa = pd.read_csv('train800.csv')
    y = dataa['Unnamed: 0']
    dataa=dataa.drop(['Unnamed: 0'],1)
    print(dataa.keys())
    print(dataa.shape)    

    ##########################
    '''
    y=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
       1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
       1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50
       ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50
       ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50
       ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    '''
    ##########################
    label = pd.read_csv('label800.csv')
    y = label['target']
    y = list(y)
    
    ##########################
    data = dataa
    target = y
    print(target)
    
    ################### Prepend the column of 1s for bias #####################
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    ########### Using Categorical Data with One Hot Encoding ##################
    target = np.array(target)
    target = target.reshape([-1, 1])      # add one extra dimension
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(target)
    encoded = encoder.transform(target)
    print(encoded)
    all_Y = encoded
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_Y, test_size=0.20, random_state=42)
    #return train_test_split(all_X, all_Y, test_size=0.20, random_state=42)
    
    return train_X, test_X, train_y, test_y
    
def main():
    ############## Splitting Dataset into train and test set ##################
    train_X, test_X, train_y, test_y = load_dataset()
    print(train_X)
    print(train_y)
    print(test_X)
    #print("train_y:",train_y[4])
    print("Test_y:",test_y[4])
    ###################################MLP#####################################

    
    
    ########################## Layer's sizes ##################################
    x_size = train_X.shape[1]   # Number of input nodes: 400 features and 1 bias
    h_size = 250                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (50 Characters)

    ########################### Symbols #######################################
    X = tf.placeholder("float", shape=[None, x_size]) #input image shape 20*20 means 400
    y = tf.placeholder("float", shape=[None, y_size]) #50 classes

    ##################### Weight initializations ##############################
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    ###################### Forward propagation ################################
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1) #Get the index of the maximum value

    ##################### Backward propagation ################################
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # Logits means, in particular, the sum of the inputs may not equal 1,
    #that the values are not probabilities
    ############################ Run SGD ######################################
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(25):
        ################## Train with each example ############################
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        
        print("Train: ",np.argmax(train_y, axis=1))
        print("train2:",sess.run(predict, feed_dict={X: train_X, y: train_y}))
        
        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        print("Test: ",np.argmax(test_y, axis=1))
        print("Test2:",sess.run(predict, feed_dict={X: test_X, y: test_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()