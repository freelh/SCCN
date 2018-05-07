import keras
import numpy as np
import cv2
import keras.backend as k
from keras.models import Model
import tensorflow as tf
from keras.layers import Input,Conv2D,GaussianNoise,Subtract,Lambda,Add,Dot
import numpy as np
class DAE(object):
    def __init__(self):
        print('创建DAE类')
    def set_par(self,input_img,n_input,n_output,name='default',k_size=1,is_mul=0,step=0.005,times=3000,std=0.01):
        self.shape = input_img.shape
        print(self.shape)
        self.img = input_img.reshape([self.shape[1],self.shape[2],-1])
        self.name = name
        self.k_size = k_size
        self.is_mul = is_mul
        self.n_output = n_output
        self.n_input = n_input
        self.step = step

        self.INPUT = Input(shape = (self.img.shape))
        self.INPUT_N = GaussianNoise(std)(self.INPUT)
        self.layer = Conv2D(filters=self.n_output , kernel_size=self.k_size , padding='same' , activation=k.sigmoid , data_format='channels_last' , use_bias = True )
        self.hidden = self.layer(self.INPUT)
        self.temp  = Conv2D(filters=self.n_input , kernel_size=self.k_size , padding='same' , activation=k.sigmoid , data_format='channels_last' , use_bias = True )(self.hidden)

        dae = Model(inputs=self.INPUT,outputs = self.temp)
        self.dae_h = Model(input=self.INPUT,outputs = self.hidden)
        dae.compile(optimizer='adam',loss='mse')
        dae.fit(input_img,input_img,epochs=times)
        self.z=self.dae_h.predict(input_img)
    def par(self):
        t = self.layer.get_weights()
        return [t[0],t[1],self.z]
    def kill(self):
        pass

class last(object):
    def __init__(self):
        print('创建last类')
    def set_par(self,input_opt,input_sar,n_input,n_output):
        step = 0.1
        times = 3000
        self.w_opt = tf.Variable(tf.truncated_normal(shape=[1, 1, n_input, n_output]))
        self.w_opt_ = tf.reshape(tf.transpose(tf.reshape(self.w_opt, [n_input, n_output])),[1, 1, n_output, n_input])
        self.b_opt = tf.Variable(tf.truncated_normal([n_output]))
        self.b_opt_ = tf.Variable(tf.truncated_normal([n_input]))

        self.w_sar = tf.Variable(tf.truncated_normal(shape=[1, 1, n_input, n_output]))
        self.w_sar_ = tf.reshape(tf.transpose(tf.reshape(self.w_sar, [n_input, n_output])), [1, 1, n_output, n_input])
        self.b_sar = tf.Variable(tf.truncated_normal([n_output]))
        self.b_sar_ = tf.Variable(tf.truncated_normal([n_input]))

        #私搭乱建
        self.hidden_opt = tf.nn.sigmoid(tf.nn.conv2d(input_opt, self.w_opt, strides=[1, 1, 1, 1], padding='SAME') + self.b_opt)
        self.out_opt = tf.nn.sigmoid(tf.nn.conv2d(self.hidden_opt, self.w_opt_, strides=[1, 1, 1, 1], padding='SAME') + self.b_opt_)
        self.hidden_sar = tf.nn.sigmoid(tf.nn.conv2d(input_sar, self.w_sar, strides=[1, 1, 1, 1], padding='SAME') + self.b_sar)
        self.out_sar = tf.nn.sigmoid(tf.nn.conv2d(self.hidden_sar, self.w_sar_, strides=[1, 1, 1, 1], padding='SAME') + self.b_sar_)

        loss_opt = tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(self.out_opt, input_opt.shape) - input_opt)))+0.00000001*tf.reduce_mean(tf.abs(self.w_opt))
        loss_sar = tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(self.out_sar, input_sar.shape) - input_sar))) + 0.00000001 * tf.reduce_mean(tf.abs(self.w_sar))
        cop = tf.reduce_mean(tf.abs(self.hidden_opt-self.hidden_sar),3)
        loss_cop = tf.sqrt(tf.reduce_mean(tf.square(self.hidden_opt-self.hidden_sar)))
        loss = loss_cop+loss_opt+loss_sar

        self.opt = tf.train.AdamOptimizer(step).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for i in range(times):
            self.sess.run(self.opt)
            if i%50 is 0 :
                print(self.sess.run(loss))
        return [self.sess.run(self.w_opt),self.sess.run(self.b_opt),self.sess.run(self.w_sar),self.sess.run(self.b_sar),self.sess.run(cop)]


    def set_parr(self,opt3h,sar3h,n_input,n_output):
        self.shape = opt3h.shape
        print(self.shape)
        self.opt3h = opt3h.reshape([self.shape[1], self.shape[2], -1])
        self.sar3h = sar3h.reshape([self.shape[1], self.shape[2], -1])
        self.INPUT_opt = Input(shape=(self.opt3h.shape))
        self.INPUT_sar = Input(shape=(self.sar3h.shape))
        self.layer_opt4 = Conv2D(filters=n_output, kernel_size=1, padding='same', activation=k.sigmoid,data_format='channels_last', use_bias=True)
        self.layer_opt4_= Conv2D(filters=n_output, kernel_size=1, padding='same', activation=k.sigmoid,data_format='channels_last', use_bias=True)
        self.layer_sar4 = Conv2D(filters=n_output, kernel_size=1, padding='same', activation=k.sigmoid,data_format='channels_last', use_bias=True)
        self.layer_sar4_= Conv2D(filters=n_output, kernel_size=1, padding='same', activation=k.sigmoid,data_format='channels_last', use_bias=True)

        self.opt4h = self.layer_opt4(self.INPUT_opt)
        self.opt4h_= self.layer_opt4_(self.opt4h)
        self.sar4h = self.layer_sar4(self.INPUT_sar)
        self.sar4h_ = self.layer_sar4_(self.sar4h)

        self.sopt = Subtract()([self.INPUT_opt,self.opt4h_])
        self.ssar = Subtract()([self.INPUT_sar,self.sar4h_])
        self.sh = Subtract()([self.opt4h,self.sar4h])

        # def sq(x):
        #     return tf.sqrt(tf.reduce_mean(tf.square(x))) #有问题
        # self.l2_opt = Lambda(sq)(self.sopt)
        # self.l2_sar = Lambda(sq)(self.ssar)
        # self.l2_h = Lambda(sq)(self.sh)
        self.d1 = Dot(4)([self.sopt,self.sopt])
        self.d2 = Dot(4)([self.ssar,self.ssar])
        self.d3 = Dot(4)([self.sh,self.sh])
        self.l2_h = Add()([self.d1,self.d2,self.d3])

        # t=[self.l2_opt,self.l2_sar,self.l2_h]
        # self.add = Add()(t)


        l = Model(inputs=[self.INPUT_opt,self.INPUT_sar], outputs=self.l2_h)
        l.compile(optimizer='adam',loss='mse')
        ex = np.abs(0).astype(np.float32)
        l.fit([opt3h,sar3h],ex, epochs=2000)
        t1 = self.layer_opt4.get_weights()
        t2 = self.layer_sar4.get_weights()
        return [t1[0],t1[1],t2[0],t2[1]]
        # img_opt=cv2.imread('i3.bmp',0)
# shape =img_opt.shape
# img_opt = (img_opt.astype(np.float32)/255).reshape([1,shape[0],shape[1],1])
#
# a= DAE()
# a.set_par(img_opt,1,20,times = 2000)
# b=a.par()
# a.set_par(b[2],20,20,times = 2000)
# b=a.par()
# a.set_par(b[2],20,20,times = 2000)
# b=a.par()
# z=b[2]
# z=z.reshape([z.shape[1],z.shape[2],-1])
# print(z.shape)
# for i in range(20):
#     cv2.imwrite('____'+str(i)+'.tif',(z[:,:,i]*255).astype(np.uint8))