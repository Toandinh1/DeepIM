import numpy as np
import tensorflow as tf
import keras as kr
from scipy.special import binom
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda, Reshape, Dense, Input, Activation
import keras.losses import binary_crossentropy as bce

#from keras import backend as K




N = 4 # number of sub-carriers
K = 1 # number of active sub-carriers
M = 4 # M-ary modulation order
epsilon=0.005
SNRdb = 7 # Training SNR
SNRtrain = 35
traing_epochs = 50
l_rate = 0.001
total_batch = 100 # number of batches per epoch
batch_size = 1000


n_output_1 = 16
n_output_2 = 32
n_input_1 = N
n_input_2 = 2*N
channel = 1
u = 1
loss = 1

m = int(np.log2(M))
c = int(np.log2(binom(N,K)))
q = K*m + c # number of bits per OFDM-IM symbol
Q= 2**q
n_output = c

c1 = 4
c2 = 1
c3 = 1
SNR = 10**(SNRdb/10)
sigma = np.sqrt(1/SNR)

display_step = 5
qam_factor = (2/3)*(M-1)
q_levels = 44
q_range = 2.75
def quantized(y,q_levels,q_range)
   delta = 2*q_range/q_levels
   value_max = q_range - delta/2
   
   y_quantized_real = delta*np.floor((np.real(y)+q_range)/delta)-value_max
   y_quantized_imag = delta*np.floor((np.imag(y)+q_range)/delta)-value_max
   
   y_quantized_real[np.where(y_quantized_real > value_max)]=value_max
   y_quantized_real[np.where(y_quantized_real < -value_max)]=-value_max
   y_quantized_imag[np.where(y_quantized_imag > value_max)]=value_max
   y_quantized_imag[np.where(y_quantized_imag > -value_max)]=-value_max
   
   y_quantized = y_quantized_real + y_quantized_imag*1j
   return y_quantized
a = 1/np.sqrt(2)

# M-ary modulations
if M==4:
    QAM = np.array([1+0j, 0+1j, -0-1j, -1+0j], dtype=complex) # gray mapping
elif M==8:
    QAM = np.array([1, a+a*1j, -a+a*1j, 1j, a-a*1j, -1j, -1, -a-a*1j], dtype=complex) # 8PSK, not 8QAM indeed
    qam_factor = 1
elif M==16:
    QAM = np.array([-3+3j, -3+1j, -3-3j, -3-1j, 
                    -1+3j, -1+1j, -1-3j, -1-1j, 
                    3+3j, 3+1j, 3-3j, 3-1j, 
                    1+3j, 1+1j, 1-3j, 1-1j], dtype=complex)
else:
    QAM = np.array([1,-1], dtype=complex) #BPSK
    qam_factor = 1


# index patterns for N=4 and K=1,2,3 only
if K==1:
    idx = np.array([[0],[1],[2],[3]])
elif K==2:
    idx = np.array([[0,1],[2,3],[0,2],[1,3]]) 
else:
    idx = np.array([[0,1,2],[1,2,3],[0,2,3],[0,1,3]]) 
def SC_IM_NO_train(bit1,bit2,bit3, SNRdb):
        #user1
    bit_id1 = bit1[0:c:1]
    id_de1 = bit_id1.dot(2**np.arange(bit_id1.size)[::-1])
    bit_sy1 = bit1[c:q:1]   
    bit_K1 = bit_sy1.reshape(-1,m)
    sy_de1 = np.zeros((K,), dtype=int)
    sym1 = np.zeros((K,), dtype=complex)
    one_hot_bit1 = np.zeros((16,),dtype=int)
    for i in range(K):
        bit_sy_i1 = bit_K1[i,:]
        sy_de1[i] = bit_sy_i1.dot(2**np.arange(bit_sy_i1.size)[::-1])
        sym1[i] = QAM[sy_de1[i]]

    tx_sym1 = np.zeros((N,), dtype=complex)
    tx_sym1[idx[id_de1,:]] = sym1
    tx_sym1 = tx_sym1*np.sqrt(c1)
    one_hot1 = 0
    for i in range(M):
      if bit1[i] == 1:
        one_hot1 = one_hot1 + 2**i
      else:
        one_hot1 = one_hot1 + 0
    one_hot_bit1[one_hot1] = 1
  #user2
    bit_id2 = bit2[0:c:1]
    id_de2 = bit_id2.dot(2**np.arange(bit_id2.size)[::-1])
    bit_sy2 = bit2[c:q:1]   
    bit_K2 = bit_sy2.reshape(-1,m)
    sy_de2 = np.zeros((K,), dtype=int)
    sym2 = np.zeros((K,), dtype=complex)
    one_hot_bit2  = np.zeros((16,),dtype=int)
    for i in range(K):
        bit_sy_i2 = bit_K2[i,:]
        sy_de2[i] = bit_sy_i2.dot(2**np.arange(bit_sy_i2.size)[::-1])
        sym2[i] = QAM[sy_de2[i]]

    tx_sym2 = np.zeros((N,), dtype=complex)
    tx_sym2[idx[id_de2,:]] = sym2
    tx_sym2 = tx_sym2*np.sqrt(c2)
    one_hot2 = 0
    for i in range(M):
      if bit2[i] == 1:
        one_hot2 = one_hot2 + 2**i
      else:
        one_hot2 = one_hot2 + 0
    one_hot_bit2[one_hot2] = 1
    #user3
      bit_id3 = bit3[0:c:1]
      id_de3 = bit_id3.dot(2**np.arange(bit_id3.size)[::-1])
      bit_sy3 = bit3[c:q:1]   
      bit_K3 = bit_sy3.reshape(-1,m)
      sy_de3 = np.zeros((K,), dtype=int)
      sym3 = np.zeros((K,), dtype=complex)
      one_hot_bit3  = np.zeros((16,),dtype=int)
      for i in range(K):
          bit_sy_i3 = bit_K3[i,:]
          sy_de3[i] = bit_sy_i3.dot(2**np.arange(bit_sy_i3.size)[::-1])
          sym3[i] = QAM[sy_de3[i]]

      tx_sym3 = np.zeros((N,), dtype=complex)
      tx_sym3[idx[id_de3,:]] = sym3
      tx_sym3 = tx_sym2*np.sqrt(c3)
      one_hot3 = 0
      for i in range(M):
        if bit3[i] == 1:
          one_hot3 = one_hot3 + 2**i
        else:
          one_hot3 = one_hot3 + 0
      one_hot_bit3[one_hot3] = 1
    #transmision
    SNR = 10**(SNRdb/10)
    sigma = np.sqrt(1/SNR)
    noise = sigma*np.sqrt(1/2)*(np.random.randn(*tx_sym1.shape)+1j*np.random.randn(*tx_sym1.shape))
    #noise = np.random.normal(0, 1, tx_sym1.shape)
    #H1 = 1
    #H2 = 1
    H = (1+1j).np.sqrt(2)*np.ones(*tx_sym1.shape)
    #H2 = np.sqrt(1/2)*(np.random.randn(*tx_sym2.shape)+1j*np.random.randn(*tx_sym2.shape))
    e = np.sqrt(epsilon/2)*np.random.radn(*tx_sym1.shape)+1j*np.random.radn(*tx_sym1.shape)
    if channel =1:
        H_est = H
    else:
        H_est= H+e
        
        
    y = H(tx_sym1 + tx_sym2 + tx_sym3) + noise
     
    y_bar = y/ H_est
    y_con = np.concatenate((np.real(y_bar),np.imag(y_bar)))
    y_m = np.absolute(y_bar)
    Y =np.concatenate((y_con,y_m))
    
    return Y,y_con,one_hot_bit1,one_hot_bit2,one_hot_bit3
    
  
    

def SC_IM_NO_test(bit1,bit2,bit3, SNRdb):
        #user1
    bit_id1 = bit1[0:c:1]
    id_de1 = bit_id1.dot(2**np.arange(bit_id1.size)[::-1])
    bit_sy1 = bit1[c:q:1]   
    bit_K1 = bit_sy1.reshape(-1,m)
    sy_de1 = np.zeros((K,), dtype=int)
    sym1 = np.zeros((K,), dtype=complex)
    one_hot_bit1 = np.zeros((16,),dtype=int)
    for i in range(K):
        bit_sy_i1 = bit_K1[i,:]
        sy_de1[i] = bit_sy_i1.dot(2**np.arange(bit_sy_i1.size)[::-1])
        sym1[i] = QAM[sy_de1[i]]

    tx_sym1 = np.zeros((N,), dtype=complex)
    tx_sym1[idx[id_de1,:]] = sym1
    tx_sym1 = tx_sym1*np.sqrt(c1)
    one_hot1 = 0
    for i in range(M):
      if bit1[i] == 1:
        one_hot1 = one_hot1 + 2**i
      else:
        one_hot1 = one_hot1 + 0
    one_hot_bit1[one_hot1] = 1
  #user2
    bit_id2 = bit2[0:c:1]
    id_de2 = bit_id2.dot(2**np.arange(bit_id2.size)[::-1])
    bit_sy2 = bit2[c:q:1]   
    bit_K2 = bit_sy2.reshape(-1,m)
    sy_de2 = np.zeros((K,), dtype=int)
    sym2 = np.zeros((K,), dtype=complex)
    one_hot_bit2  = np.zeros((16,),dtype=int)
    for i in range(K):
        bit_sy_i2 = bit_K2[i,:]
        sy_de2[i] = bit_sy_i2.dot(2**np.arange(bit_sy_i2.size)[::-1])
        sym2[i] = QAM[sy_de2[i]]

    tx_sym2 = np.zeros((N,), dtype=complex)
    tx_sym2[idx[id_de2,:]] = sym2
    tx_sym2 = tx_sym2*np.sqrt(c2)
    one_hot2 = 0
    for i in range(M):
      if bit2[i] == 1:
        one_hot2 = one_hot2 + 2**i
      else:
        one_hot2 = one_hot2 + 0
    one_hot_bit2[one_hot2] = 1
    #user3
      bit_id3 = bit3[0:c:1]
      id_de3 = bit_id3.dot(2**np.arange(bit_id3.size)[::-1])
      bit_sy3 = bit3[c:q:1]   
      bit_K3 = bit_sy3.reshape(-1,m)
      sy_de3 = np.zeros((K,), dtype=int)
      sym3 = np.zeros((K,), dtype=complex)
      one_hot_bit3  = np.zeros((16,),dtype=int)
      for i in range(K):
          bit_sy_i3 = bit_K3[i,:]
          sy_de3[i] = bit_sy_i3.dot(2**np.arange(bit_sy_i3.size)[::-1])
          sym3[i] = QAM[sy_de3[i]]

      tx_sym3 = np.zeros((N,), dtype=complex)
      tx_sym3[idx[id_de3,:]] = sym3
      tx_sym3 = tx_sym2*np.sqrt(c3)
      one_hot3 = 0
      for i in range(M):
        if bit3[i] == 1:
          one_hot3 = one_hot3 + 2**i
        else:
          one_hot3 = one_hot3 + 0
      one_hot_bit3[one_hot3] = 1
    #transmision
    SNR = 10**(SNRdb/10)
    sigma = np.sqrt(1/SNR)
    noise = sigma*np.sqrt(1/2)*(np.random.randn(*tx_sym1.shape)+1j*np.random.randn(*tx_sym1.shape))
    #noise = np.random.normal(0, 1, tx_sym1.shape)
    #H1 = 1
    #H2 = 1
    H = (1+1j).np.sqrt(2)*np.ones(*tx_sym1.shape)
    #H2 = np.sqrt(1/2)*(np.random.randn(*tx_sym2.shape)+1j*np.random.randn(*tx_sym2.shape))
    e = np.sqrt(epsilon/2)*np.random.radn(*tx_sym1.shape)+1j*np.random.radn(*tx_sym1.shape)
    if channel =1:
        H_est = H
    else:
        H_est= H+e
        
        
    y = H(tx_sym1 + tx_sym2 + tx_sym3) + noise
     
    y_bar = y/ H_est
    y_con = np.concatenate((np.real(y_bar),np.imag(y_bar)))
    y_m = np.absolute(y_bar)
    Y =np.concatenate((y_con,y_m))
    
    return Y,y_con,one_hot_bit1,one_hot_bit2,one_hot_bit3
#model     
ini = 'glorot_uniform'
init=tf.global_variables_initializer()
X = tf.placeholder("float", [None, 12])
Y = tf.placeholder("float", [None, 16])
initializer = tf.contrib.layers.xavier_initializer()

def encoder(x):
     layer_1=dense(2*(2**q),activation=tf.nn.tanh,init=ini)(x)
     layer_2=dense(4*(2**q),activation=tf.nn.tanh,init=ini)(layer_1)
     norm =BatchNormalization(momentum=0.99,epsilon=0.0001,center=True,scale=True)(layer_2)
     layer_3=dense(2**q,activation=tf.nn.softmax,init=ini)(norm)
     return layer_3

y_pred= encoder(X)
X1= tf.concat((y_pred,X),axis=-1)
Y1 = tf.placeholder("float", [None, 2**q])
Y2 = tf.placeholder("float", [None, 2**q])
Y3 = tf.placeholder("float", [None, 2**q])
def encoder1(x):
     layer_1=dense(2*(2**q),activation=tf.nn.tanh,init=ini)(x)
     layer_2=dense(4*(2**q),activation=tf.nn.tanh,init=ini)(layer_1)
     norm =BatchNormalization(momentum=0.99,epsilon=0.0001,center=True,scale=True)(layer_2)
     layer_3=dense(2**q,activation=tf.nn.softmax,init=ini)(norm)
     return layer_3
y2_true=Y2
y2_pred=encoder1(X1)
X2=tf.concat((y_pred,y2_pred,X),axis=-1)

def encoder2(x):
     layer_1=dense(2*(2**q),activation=tf.nn.tanh,init=ini)(x)
     layer_2=dense(4*(2**q),activation=tf.nn.tanh,init=ini)(layer_1)
     norm =BatchNormalization(momentum=0.99,epsilon=0.0001,center=True,scale=True)(layer_2)
     layer_3=dense(2**q,activation=tf.nn.softmax,init=ini)(norm)
     return layer_3

y_true = Y1
y_pred = encoder(X)
y2_true = Y2
y2_pred = encoder1(X1)
y3_true = Y3
y3_pred = encoder(X2)

if loss ==1:
    cost = tf.reduce_mean(bce(y3_true,y3_pred))
else:
    cost = tf.reduce_mean(bce(y_true,y_pred)+bce(y2_true,y2_pred)+bce(y3_true,y3_pred))    
#y_pred = encoder(X)
#y_true = Y

#cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()
def detobit(A):
  x= np.zeros((6,))
  y= np.zeros((6,))
  x[0] = A//2
  y[0] = A%2
  x[1] = x[0]//2
  y[1] = x[0]%2
  x[2] = x[1]//2
  y[2] = x[1]%2
  x[3] = x[2]//2
  y[3] = x[2]%2
  bit_est =np.array([y[0],y[1],y[2],y[3]],dtype= int)
  return bit_est

def frange(x,y,jump):
    while x < y:
        yield x
        x +=jump

EbNodB_range = list(frange(0,50,5))
BER1 = [None]*len(EbNodB_range)


with tf.Session() as sess:
#Training
    sess.run(init)
    for epoch in range(traing_epochs):
        avg_cost = 0
        for index_m in range(total_batch):
            input_samples = []
            input_samples1 = []
            input_labels = []
            input_labels1 = []
            input_labels2 = []
            
            for index_k in range(0, batch_size):
                bits1 = np.random.binomial(n=1,p=0.5,size=(q,))
                bits2 = np.random.binomial(n=1,p=0.5,size=(q,))
                bits3 = np.random.binomial(n=1,p=0.5,size=(q,))
                signaloutput,y_con,one_hot_bit1,one_hot_bit2,one_hot_bit3 = SC_IM_NO_train(bits1,bits2,bits3,SNRtrain)
                input_labels.append(one_hot_bit1)
                input_labels1.append(one_hot_bit2)
                input_labels2.append(one_hot_bit3)
                input_samples.append(signaloutput)
                input_samples1.append(y_con)
                
               

            batch_1 = np.asarray(input_samples)
            batch_2 = np.asarray(input_samples1)
            batch_3 = np.asarray(input_labels)
            batch_4 = np.asarray(input_labels1)
            batch_5 = np.asarray(input_labels2)
           
            
            

            _,cs = sess.run([optimizer,cost], feed_dict={X:batch_1,
                                                        Y:batch_3,Y2:batch_4,Y3:batch_5
                                                        learning_rate:l_rate})
            avg_cost += cs / total_batch
        if epoch % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1), "cost=", \
               "{:.9f}".format(avg_cost))
#==========Testing=============
    for n in range(0,len(EbNodB_range)):
      input_samples_test = []
      input_samples1_test = []
      input_labels_test = []
      input_labels1_test = []
      input_labels2_test = []
      
      test_number = 100000
      if n>10:
        test_number = 100000
      for i in range(0, test_number):
        bits1 = np.random.binomial(n=1, p=0.5, size=(q, )) 
        bits2 = np.random.binomial(n=1, p=0.5, size=(q, ))
        signaloutput,y_con,one_hot_bit1,one_hot_bit2,one_hot_bit3 = SC_IM_NO_train(bits1,bits2,bits3,EbNodB_range[n])
        input_labels_test.append(bits1)
        input_labels1_test.append(bits2)
        input_labels2_test.append(bits3)
        input_samples_test.append(signaloutput)
        input_samples1_test.append(y_con)
        
  
  
      batch_x = np.asarray(input_samples_test)
      batch_y = np.asarray(input_samples1_testt)
      batch_z = np.asarray(input_labels_test)
      batch_t = np.asarray(input_labels1_test)
      batch_k = np.asarray(input_labels2_test)
      one_hot_bit1_est= sess.run(y_pred,feed_dict={X:batch_x})
      one_hot_bit2_est= sess.run(y2_pred,feed_dict={X:batch_x})
      one_hot_bit3_est= sess.run(y3_pred,feed_dict={X:batch_x})
      bit_error = 0
      for i in range(0, test_number):
        if u==1:
           ind_est =np.argmax(one_hot_bit1_est[i,])
           bit_est = detobit(ind_est)
           bit_error =bit_error+sum(bit_est!=batch_z[i,])
        elif u==2:
           ind_est =np.argmax(one_hot_bit2_est[i,])
           bit_est = detobit(ind_est)
           bit_error =bit_error+sum(bit_est!=batch_t[i,]) 
        else:
            ind_est =np.argmax(one_hot_bit3_est[i,])
            bit_est = detobit(ind_est)
            bit_error =bit_error+sum(bit_est!=batch_k[i,])
      BER[n] = bit_error/(test_number*q)
    if u==1:
       print("SNR=", EbNodB_range[n], "BER1:", BER[n])
    if u==2:
       print("SNR=", EbNodB_range[n], "BER2:", BER[n])
    if u==3:
       print("SNR=", EbNodB_range[n], "BER3:", BER[n])
    #ML1=[0.133100000000000,	0.0551562500000000,	0.0170187500000000,	0.00188125000000000,	5.62500000000000e-05,	0,	0]

