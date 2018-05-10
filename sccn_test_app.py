import tensorflow as tf
import numpy as np
import DAEk
import cv2
import sys
import time
def SCCN(opt_path,sar_path,save_path,noise_std,p_times,p_step,lam,t_times,t_step,TEXT):

    img_opt=cv2.imread(opt_path,0)
    shape =img_opt.shape
    img_opt = (img_opt.astype(np.float32)/255).reshape([1,shape[0],shape[1],1])
    img_sar = (cv2.imread(sar_path,0).astype(np.float32)/255).reshape([1,shape[0],shape[1],1])

    k=100000000#60000
    map = np.array(np.random.rand(shape[0],shape[1]),dtype=np.float32)
    global map
    global phi
    global shape
    print(map.shape)
    ksize = 3
    phi = lam
    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    start_time = time.localtime()
    TEXT.AppendText(now+':start pretrain\n')

    #--------------------------------------初始化---------------------------------------------#
    a=DAEk.DAE(TEXT)
    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : 1st opt_layer\n')
    a.set_par(img_opt,1,20,name='conv_opt_1',k_size=ksize,step=p_step,times=p_times,std=noise_std)
    [opt_1_w,opt_1_b,opt_1_h] = a.par()
    a.kill()

    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : 1st sar_layer\n')
    a.set_par(img_sar,1,20,name='conv_sar_1',is_mul=1,k_size=ksize,step=p_step,times=p_times,std=noise_std)
    [sar_1_w,sar_1_b,sar_1_h] = a.par()
    a.kill()

    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : 2nd opt_layer\n')
    a.set_par(opt_1_h,20,20,name='conv_opt_2',k_size=1,step=p_step,times=p_times,std=noise_std)
    [opt_2_w,opt_2_b,opt_2_h] = a.par()
    a.kill()

    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : 2nd sar_layer\n')
    a.set_par(sar_1_h,20,20,name='conv_sar_2',is_mul=1,k_size=1,step=p_step,times=p_times,std=noise_std)
    [sar_2_w,sar_2_b,sar_2_h] = a.par()
    a.kill()

    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : 3rd opt_layer\n')
    a.set_par(opt_2_h,20,20,name='conv_opt_3',k_size=1,step=p_step,times=p_times,std=noise_std)
    [opt_3_w,opt_3_b,opt_3_h] = a.par()
    a.kill()

    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : 3rd sar_layer\n')
    a.set_par(sar_2_h,20,20,name='conv_sar_3',is_mul=1,k_size=1,step=p_step,times=p_times,std=noise_std)
    [sar_3_w,sar_3_b,sar_3_h] = a.par()
    a.kill()



    #---------------------------------搭建网络-----------------------------------#
    opt1w = tf.Variable(opt_1_w)
    opt1b = tf.Variable(opt_1_b)
    opt1h = tf.nn.sigmoid(tf.nn.conv2d(img_opt , opt1w ,strides=[1,1,1,1],padding='SAME')+opt1b)

    sar1w = tf.Variable(sar_1_w)
    sar1b = tf.Variable(sar_1_b)
    sar1h = tf.nn.sigmoid(tf.nn.conv2d(img_sar , sar1w ,strides=[1,1,1,1],padding='SAME')+sar1b)
    #------------2
    opt2w = tf.Variable(opt_2_w)
    opt2b = tf.Variable(opt_2_b)
    opt2h = tf.nn.sigmoid(tf.nn.conv2d(opt1h , opt2w ,strides=[1,1,1,1],padding='SAME')+opt2b)

    sar2w = tf.Variable(sar_2_w)
    sar2b = tf.Variable(sar_2_b)
    sar2h = tf.nn.sigmoid(tf.nn.conv2d(sar1h , sar2w ,strides=[1,1,1,1],padding='SAME')+sar2b)

    #-------------3
    opt3w = tf.Variable(opt_3_w)
    opt3b = tf.Variable(opt_3_b)
    opt3h = tf.nn.sigmoid(tf.nn.conv2d(opt2h , opt3w ,strides=[1,1,1,1],padding='SAME')+opt3b)

    sar3w = tf.Variable(sar_3_w)
    sar3b = tf.Variable(sar_3_b)
    sar3h = tf.nn.sigmoid(tf.nn.conv2d(sar2h , sar3w ,strides=[1,1,1,1],padding='SAME')+sar3b)

    #---------------4
    l = DAEk.last()
    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain : last layer\n')
    [opt_4_w,opt_4_b,sar_4_w,sar_4_b,map_temp] = l.set_par(opt_3_h,sar_3_h,20,20)

    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':pretrain :over\n')
    print(map_temp)
    map_temp = map_temp/(np.max(map_temp))
    map_temp = map_temp.reshape(shape)
    # map = 1-map_temp
    # cv2.imshow('0',map_temp)
    # cv2.waitKey(-1)
    print(map.shape)

    cv2.imwrite(save_path+'/A.tif',np.array(map_temp*255).astype(np.uint8))
    # map = map_temp.reshape(map.shape)
    # map = map/max(max(map))
    opt4w = tf.Variable(opt_4_w)
    opt4b = tf.Variable(opt_4_b)
    sar4w = tf.Variable(sar_4_w)
    sar4b = tf.Variable(sar_4_b)

    opt4h=tf.sigmoid(tf.nn.conv2d(opt3h , opt4w ,strides=[1,1,1,1],padding='SAME')+opt4b)
    sar4h=tf.sigmoid(tf.nn.conv2d(sar3h , sar4w ,strides=[1,1,1,1],padding='SAME')+opt3b)



    #计算
    sub = tf.sqrt(tf.reduce_mean(tf.square(opt4h-sar4h),3))#run
    sub = tf.reshape(sub,shape)
    def cal(a):
        return 1/(1+2.71828**(k*(a-phi)))  #0->1
    def cal_map(sub):
        global map
        map=cal(sub)



    #loss = tf.reduce_mean(sub * map)  - phi*tf.reduce_mean(map)
    map_p = tf.placeholder(tf.float32)
    loss = tf.reduce_mean(sub * map_p) - phi * tf.reduce_mean(map_p)
    step = t_step
    decay_rate = 0.96  # 衰减率
    global_steps = t_times # 总的迭代次数
    decay_steps = 100  # 衰减次数
    global_ = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(step,global_,decay_steps=global_steps,decay_rate=decay_rate,staircase=False)


    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_)
    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':train\n')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(20):


            cv2.imwrite(save_path +'/before_' +str(i)  + '_opt.tif', ((sess.run(tf.reshape(opt3h[:,:,:,i],shape))) * 255).astype(np.uint8))
            cv2.imwrite(save_path +'/before_' +str(i)  + '_sar.tif', ((sess.run(tf.reshape(sar3h[:,:,:,i],shape))) * 255).astype(np.uint8))
        temp = 0
        TEXT.AppendText('=')
        for j in range(global_steps):
            cal_map(sess.run(sub)/max(max(sess.run(sub))))
            sess.run(opt,feed_dict={map_p:map})

            if(j*10//global_steps == temp):
                pass
            else:
                TEXT.AppendText('=')
                temp = j*10//global_steps

            phi = phi / 1
            # k = k*1.005
            map_n = map / (np.max(map))
            if j is 15:
                print(map)
                print(map_n)
                print(sess.run(sub))
            cv2.imwrite(save_path+'/res'+str(j)+'.tif',(((1-map_n))*255).astype(np.uint8))
        print(save_path + '\\' + str(10) + 'opt.tif')
        for i in range(20):
            cv2.imwrite(save_path + '/after_'+str(i)  +'_opt.tif', ((sess.run(tf.reshape(opt4h[:,:,:,i],shape))) * 255).astype(np.uint8))
            cv2.imwrite(save_path+ '/after_'+str(i)  + '_sar.tif', ((sess.run(tf.reshape(sar4h[:,:,:,i],shape))) * 255).astype(np.uint8))
        print('\n\n\n\n\n\n\n\n')
    now = time.strftime("\n%d-%H:%M:%S  ", time.localtime())
    TEXT.AppendText(now + ':Success!\n')