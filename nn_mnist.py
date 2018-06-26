import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


train_x, train_y = train_set
train_y = one_hot(train_y,10)

valid_x, valid_y = valid_set
valid_y = one_hot(valid_y,10)

test_x, test_y = test_set
test_y = one_hot(test_y,10)

# ---------------- Visualizing some element of the MNIST dataset --------------
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt
#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])


# TODO: the neural net!!

inputLayer = 784    # 28x28 samples
outputLayer = 10    # 10 classes

x = tf.placeholder("float", [None, inputLayer])
y_ = tf.placeholder("float", [None, outputLayer])

W = tf.Variable(tf.zeros([inputLayer, outputLayer])) #pesos
b = tf.Variable(tf.zeros([outputLayer])) #bias, umbral

y = tf.nn.softmax(tf.matmul(x, W) + b) #funcion de salida

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)




batch_size = 20

currentLoss = 9999
prevLoss = 9999
epoch = 0
valoresGraficaTrain = []
#valoresGraficaValid = []


while (currentLoss <= prevLoss ):
    epoch += 1
    for jj in range(int(len(train_x) / batch_size)):
        batch_xsTrain = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ysTrain = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xsTrain, y_: batch_ysTrain})
    prevLoss = currentLoss
    currentLoss = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    valoresGraficaTrain.append(sess.run(loss, feed_dict={x: valid_x, y_: valid_y}))
    #valoresGraficaValid.append(sess.run(loss, feed_dict={x: test_x, y_: test_y}))
    print("Epoch#",epoch," Current lossValue:", currentLoss,"Previous lossValue:", prevLoss)

misses = 0. #errores
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):

    if (np.argmax(b) != np.argmax(r)):

        misses += 1.


#misses += 0.0
errorPercent = misses/len(test_x)*100
print ("----------------------------------------------------------------------------------")
print ("Error:", errorPercent,"% Total:",misses)
print ("X len:",len(test_x))

plt.title("Grafica")
plt.plot(valoresGraficaTrain)
#plt.plot(valoresGraficaValid)
plt.show()