import tensorflow as tf
import numpy as np
from ops import*

"""
TODO:
test2
1. load image, video
2. loss funcion
3. train
4. gpu implementation
"""
#load data
batch_size = 64
frame_size = 64
fine_size = 32
opt_lambda = 10
image = tf.truncated_normal(shape= [batch_size, 64,64,3])
video = tf.truncated_normal(shape = [batch_size, 32, 64, 64, 3]) #depth, height, width, dim
#encode net(2d encoding)
def encode_net(image):
    en_dim = 128
    h1 = tf.nn.relu(conv2d(image,en_dim, name ='en_h1_conv')) #batch,32,32,128 
    h2 = tf.nn.relu(batch_norm(conv2d(h1,en_dim*2, name = 'en_h2_conv')))
    h3 = tf.nn.relu(batch_norm(conv2d(h2,en_dim*4, name = 'en_h3_conv')))
    h4 = tf.nn.relu(batch_norm(conv2d(h3,en_dim*8, name = 'en_h4_conv'))) #batch, 4,4,1024
    return h4

#static net(2d decoding)
def static_net(input_):
    #input = (batch, 4,4,1024)
    input_shape = input_.get_shape().as_list()
    st_dim = input_shape[-1]/2
    st_h = input_shape[1]*2
    st_w = input_shape[2]*2
    h1 = tf.nn.relu(batch_norm(deconv2d(input_, [batch_size, st_h, st_w, st_dim], name = 'st_h1_deconv'))) #512
    h2 = tf.nn.relu(batch_norm(deconv2d(h1, [batch_size, st_h*2, st_w*2, st_dim/2], name = 'st_h2_deconv'))) #256
    h3 = tf.nn.relu(batch_norm(deconv2d(h2, [batch_size, st_h*4,st_w*4,st_dim/4], name = 'st_h3_deconv'))) #128
    h4 = tf.nn.tanh(deconv2d(h3, [batch_size, st_h*8,st_w*8,3], name = 'st_h4_deconv')) #32, 64, 64,3
    return h4

#net video(3d decoding)
def net_video(input_):
    #input = (batch, 4,4,1024)
    input_shape = input_.get_shape().as_list()
    net_dim = input_shape[-1]
    net_h = input_shape[1]
    net_w = input_shape[2]
    net_d = 2
    input_ = tf.reshape(input_, [batch_size, 1,net_h,net_w,net_dim])
    
    h1 = tf.nn.relu(batch_norm(deconv3d(input_, [batch_size, net_d, net_h, net_w, net_dim/2], name = 'net_h1_deconv')))
    h2 = tf.nn.relu(batch_norm(deconv3d(h1, [batch_size, net_d*2, net_h*2, net_w*2, net_dim/4], name = 'net_h2_deconv')))
    h3 = tf.nn.relu(batch_norm(deconv3d(h2, [batch_size, net_d*4, net_h*4, net_w*4, net_dim/8], name = 'net_h3_deconv')))
    h4 = tf.nn.relu(batch_norm(deconv3d(h3, [batch_size, net_d*8, net_h*8, net_w*8, net_dim/16], name = 'net_h4_deconv')))
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
    mask_net = tf.nn.sigmoid(deconv3d(input_, [batch_size, m_d*2, m_h*2, m_w*2, 1], name = 'mask_h1_deconv'))

    #gen net (batch, 32, 64, 64, 3)
    gen_net = tf.tanh(deconv3d(input_, [batch_size, g_d*2, g_h*2, g_w*2, 3], name= 'gen_h2_deconv'))
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
def Build_Model(image, video):
    gen_dat = Generator(image)
    d_fake_logits = Discriminator(gen_dat)
    d_real_logits = Discriminator(video, reuse = True)
    #L1 distance
    #reg_loss = 
    #loss function
    """
    paper uses different cross entropy: logsoftmax + ClassNLLCriterion
    """
    d_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real_logits)))
    d_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake_logits)))
    d_loss = d_real_loss + d_fake_loss

    g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake_logits)))
    
    print(d_real_loss.eval())
    print(d_fake_loss)
    print(g_loss)

Build_Model(image,video)
    #save variables

#train
