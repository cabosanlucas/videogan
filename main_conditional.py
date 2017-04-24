import tensorflow as tf
#import numpy as np
from ops import*

"""
TODO:
test2
1. load image, video
2. loss funcion -> think this is done
3. train
4. gpu implementation
"""
opt = {
  dataset = 'video2',   # indicates what dataset load to use (in data.lua)
  nThreads = 32,        # how many threads to pre-fetch data
  batchSize = 32,      # self-explanatory
  loadSize = 128,       # when loading images, resize first to this size
  fineSize = 64,       # crop this size from the loaded image 
  frameSize = 32,
  lr = 0.0002,          # learning rate
  lr_decay = 1000,         # how often to decay learning rate (in epoch's)
  lambda = 10,
  beta1 = 0.5,          # momentum term for adam
  meanIter = 0,         # how many iterations to retrieve for mean estimation
  saveIter = 1000,    # write check point on this interval
  niter = 100,          # number of iterations through dataset
  ntrain = math.huge,   # how big one epoch should be
  gpu = 1,              # which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            # whether to use cudnn or not
  finetune = '',        # if set, will load this network instead of starting from scratch
  name = 'condbeach7',        # the name of the experiment
  randomize = 1,        # whether to shuffle the data file or not
  cropping = 'random',  # options for data augmentation
  display_port = 8000,  # port to push graphs
  display_id = 1,       # window ID when pushing graphs
  mean = {0,0,0},
  data_root = '/data/vision/torralba/crossmodal/flickr_videos/',
  data_list = '/data/vision/torralba/crossmodal/flickr_videos/scene_extract/lists-full/_b_beach.txt.train',
}

#function: load data
opt['batchSize'] = 64
image = tf.truncated_normal(shape= [opt['batchSize'], 64,64,3]) #take first frame of the video
video = tf.truncated_normal(shape = [opt['batchSize'], 32, 64, 64, 3]) #depth, height, width, dim
#encode net(2d encoding)
def encode_net(image):
    en_dim = 128
    h1 = tf.nn.relu(conv2d(image,en_dim, name ='g_en_h1_conv')) #batch,32,32,128 
    h2 = tf.nn.relu(batch_norm(conv2d(h1,en_dim*2, name = 'g_en_h2_conv')))
    h3 = tf.nn.relu(batch_norm(conv2d(h2,en_dim*4, name = 'g_en_h3_conv')))
    h4 = tf.nn.relu(batch_norm(conv2d(h3,en_dim*8, name = 'g_en_h4_conv'))) #batch, 4,4,1024
    return h4

#static net(2d decoding)
def static_net(input_):
    #input = (batch, 4,4,1024)
    input_shape = input_.get_shape().as_list()
    st_dim = input_shape[-1]/2
    st_h = input_shape[1]*2
    st_w = input_shape[2]*2
    h1 = tf.nn.relu(batch_norm(deconv2d(input_, [opt['batchSize'], st_h, st_w, st_dim], name = 'g_st_h1_deconv'))) #512
    h2 = tf.nn.relu(batch_norm(deconv2d(h1, [opt['batchSize'], st_h*2, st_w*2, st_dim/2], name = 'g_st_h2_deconv'))) #256
    h3 = tf.nn.relu(batch_norm(deconv2d(h2, [opt['batchSize'], st_h*4,st_w*4,st_dim/4], name = 'g_st_h3_deconv'))) #128
    h4 = tf.nn.tanh(deconv2d(h3, [opt['batchSize'], st_h*8,st_w*8,3], name = 'g_st_h4_deconv')) #32, 64, 64,3
    return h4

#net video(3d decoding)
def net_video(input_):
    #input = (batch, 4,4,1024)
    input_shape = input_.get_shape().as_list()
    net_dim = input_shape[-1]
    net_h = input_shape[1]
    net_w = input_shape[2]
    net_d = 2
    input_ = tf.reshape(input_, [opt['batchSize'], 1,net_h,net_w,net_dim])
    
    h1 = tf.nn.relu(batch_norm(deconv3d(input_, [opt['batchSize'], net_d, net_h, net_w, net_dim/2], name = 'g_net_h1_deconv')))
    h2 = tf.nn.relu(batch_norm(deconv3d(h1, [opt['batchSize'], net_d*2, net_h*2, net_w*2, net_dim/4], name = 'g_net_h2_deconv')))
    h3 = tf.nn.relu(batch_norm(deconv3d(h2, [opt['batchSize'], net_d*4, net_h*4, net_w*4, net_dim/8], name = 'g_net_h3_deconv')))
    h4 = tf.nn.relu(batch_norm(deconv3d(h3, [opt['batchSize'], net_d*8, net_h*8, net_w*8, net_dim/16], name = 'g_net_h4_deconv')))
    # batch, 16, 32, 32, 64
    return h4

#mask out(3d decoding)
def mask_out(input_):
    #input = (batch, 16, 32, 32, 64)
    input_shape = input_.get_shape().as_list()
    m_dim = g_dim = input_shape[-1]
    m_d = g_d = input_shape[1]
    m_h = g_h = input_shape[2]
    m_w = g_w = input_shape[3]

    #mask net (batch, 32, 64, 64, 1)
    mask_net = tf.nn.sigmoid(deconv3d(input_, [opt['batchSize'], m_d*2, m_h*2, m_w*2, 1], name = 'g_mask_h1_deconv'))

    #gen net (batch, 32, 64, 64, 3)
    gen_net = tf.tanh(deconv3d(input_, [opt['batchSize'], g_d*2, g_h*2, g_w*2, 3], name= 'g_gen_h2_deconv'))
    return mask_net, gen_net

#discriminator net (3d encoding)
def discriminator_net(video, reuse = False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        #input = (batch, 32, 64, 64, 3)
        v_dim = 128
        h1 = lrelu(conv3d(video, v_dim, name = 'disc_h1_conv'))
        h2 = lrelu(batch_norm(conv3d(h1, v_dim*2, name = 'disc_h2_conv')))
        h3 = lrelu(batch_norm(conv3d(h2, v_dim*4, name = 'disc_h3_conv')))
        h4 = lrelu(batch_norm(conv3d(h3, v_dim*8, name = 'disc_h4_conv'))) # batch, 2, 4, 4, 1024
        h5 = conv3d(h4, 2, f_d = 2,f_h = 4,f_w=4,  name = 'disc_h5_conv')
        h6 = conv3d(h5, 2, 1,2,2, name = 'disc_h6_conv') #batch, 1,1,1,2
        return h6

def Generator(image):
    enc = encode_net(image)
    back = static_net(enc)
    mask, fore = mask_out(net_video(enc))
    # dimension of back should be modified
    netG = tf.add(tf.multiply(mask,fore), tf.multiply(tf.subtract(tf.ones_like(mask),mask),back[:,None,:,:,:]))
    return netG

def Discriminator(video, reuse = False):
    #Discriminator
    netD = discriminator_net(video, reuse)
    return netD

#Build Model
def Build_Model_and_Train(image, video):
    gen_dat = Generator(image)
    d_fake_logits = Discriminator(gen_dat)
    d_real_logits = Discriminator(video, reuse = True)
    #L1 distance
    """
    paper uses different cross entropy: logsoftmax + ClassNLLCriterion
    """
    d_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real_logits)))
    d_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake_logits)))
    d_loss = d_real_loss + d_fake_loss

    g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake_logits)))
    reg_gen = tf.transpose(gen_dat, [4,0,1,2,3])
    reg_frame = tf.transpose(video, [4,0,1,2,3])
    reg_loss =  tf.reduce_mean(tf.abs(tf.subtract(reg_gen[0],reg_frame[0])))
    
    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'disc_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()
    
    #load data

    d_optim = tf.train.AdamOptimizer(opt['lr'], beta1 = opt['beta1']).minimize(d_loss, var_list = d_vars)
    g_optim = tf.train.AdamOptimizer(opt['lr'], beta1 = opt['beta1']).minimize(g_loss, var_list = g_vars)
    #learning rate dcay? if requested
    r_optim = tf.train.AdamOptimizer(opt['lr']).minimize(reg_loss, var= g_vars)

    tf.global_variables_initializer().run()

    for epoch in xrange():

Build_Model(image,video)
    #save variables

#train
