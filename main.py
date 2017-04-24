import tensorflow as tf
import numpy as np
import utils
from ops import *


opt = {
  "dataset" : 'video2',   #indicates what dataset load to use (in data.lua)
  "nThreads" : 32,        #how many threads to pre-fetch data
  "batchSize" : 64,      #self-explanatory
  "loadSize" : 128,       # when loading images, resize first to this size
  "fineSize" : 64,       #crop this size from the loaded image 
  "frameSize" : 32,
  "lr" : 0.0002,          #learning rate
  "lr_decay" : 1000,      #   -- how often to decay learning rate (in epoch's)
  "lambda" : 0.1,
  "beta1" : 0.5,         #momentum term for adam
  "meanIter" : 0,        # how many iterations to retrieve for mean estimation
  "saveIter" : 1000,    #write check point on this interval
  "niter" : 100,          #number of iterations through dataset
  "ntrain" : float("inf"),   #how big one epoch should be
  "gpu" : 1,              # which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  "cudnn" : 1,            # whether to use cudnn or not
  "finetune" : '',        # if set, will load this network instead of starting from scratch
  "name" : 'beach100',        # the name of the experiment
  "randomize" : 1,        #whether to shuffle the data file or not
  "cropping" : 'random',  #options for data augmentation
  "display_port" : 8001,  # port to push graphs
  "display_id" : 1,       # window ID when pushing graphs
  "mean" : {0,0,0},
  "data_root" : '/data/vision/torralba/crossmodal/flickr_videos/',
  "data_list" : '/data/vision/torralba/crossmodal/flickr_videos/scene_extract/lists-full/_b_beach.txt.train'
  }

#-- one-line argument parser. parses enviroment variables to override the defaults
#for k,v in pairs(opt) do opt[k] : tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

#TODO::
#if opt["gpu"] > 0:


""" NEED TO LOAD DATA TO TENSORS """
def static_net(tensor):

	#may need to add batch size dimiension to tensor
	#see https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
	# five convolutional layers with their channel counts
	batchSize = 64
	input_dim = 4 # 100
	K = 512  # first convolutional layer output depth
	L = 256  # second convolutional layer output depth
	M = 128  # third convolutional layer
	N = 64   # fourth convolutional layer
	O = 3	 # fifth layer

	input_shape = tensor.get_shape().as_list()
	print input_shape
	st_dim = 512
	st_h = 4
	st_w = 4
	tf.Print(tensor, [tensor])
	h1 = tf.nn.relu(batch_norm(deconv2d(tensor, [batchSize, st_h, st_w, st_dim], name = 'st_h1_deconv'))) #512
	h2 = tf.nn.relu(batch_norm(deconv2d(h1, [batchSize, st_h*2, st_w*2, st_dim/2], name = 'st_h2_deconv'))) #256
	h3 = tf.nn.relu(batch_norm(deconv2d(h2, [batchSize, st_h*4,st_w*4,st_dim/4], name = 'st_h3_deconv'))) #128
	h4 = tf.nn.relu(batch_norm(deconv2d(h3, [batchSize, st_h*8,st_w*8,st_dim/8], name = 'st_h4_deconv'))) #64, 
	h5 = tf.nn.tanh(batch_norm(deconv2d(h4, [batchSize, st_h*8,st_w*8, 3], name = 'st_h5_deconv'))) #32, 
	return tf.Print(h5, [h5])


#TODO: see https://github.com/cvondrick/videogan/blob/master/main.lua line 92
def net_video(tensor):
	input_size = 4 #100
	# four convolutional layers with their channel counts
	K = 512  # first convolutional layer output depth
	L = 256  # second convolutional layer output depth
	M = 128  # third convolutional layer
	N = 64   # fourth convolutional layer

	input_shape = tensor.get_shape().as_list()
	print input_shape
	net_h = input_shape[1]
	net_w = input_shape[2]
	net_d = 2
	tensor = tf.reshape(tensor, [64, 1, net_h, net_w, 128])
	
	h1 = tf.nn.relu(batch_norm(deconv3d(tensor, [64, net_d, net_h, net_w, K], name = 'm_net_h1_deconv'))) #100 -> 512
	h2 = tf.nn.relu(batch_norm(deconv3d(h1, [64, net_d*2, net_h*2, net_w*2, L], name = 'm_net_h2_deconv'))) #512 -> 256
	h3 = tf.nn.relu(batch_norm(deconv3d(h2, [64, net_d*4, net_h*4, net_w*4, M], name = 'm_net_h3_deconv'))) #256 ->128
	h4 = tf.nn.relu(batch_norm(deconv3d(h3, [64, net_d*8, net_h*8, net_w*8, N], name = 'm_net_h4_deconv'))) #128 -> 64
	# batch, 16, 32, 32, 64
	return tf.Print(h4, [h4])

	
def mask_net(tensor, L1):
	input_shape = tensor.get_shape().as_list()
	print input_shape
	net_h = input_shape[1]
	net_w = input_shape[2]
	net_d = input_shape[3]

	mask_out = deconv3d(tensor, [64, net_d*2, net_h*2, net_w*2, 1], name = "mask_out_deconv")
	
	L1Penalty = tf.contrib.layers.l1_regularizer(L1)
	return tf.nn.sigmoid(mask_out) #+ L1Penalty


def gen_net(tensor):
	return tf.nn.tanh(tf.layers.deconv3d(tensor, [64, 4, 4, 2, 3], name = "gen_net_deconv"))

def disc_net(tensor):
	# five convolutional layers with their channel counts
	K = 64  # first convolutional layer output depth
	L = 128  # second convolutional layer output depth
	M = 256  # third convolutional layer
	N = 512   # fourth convolutional layer
	O = 2	 # fifth layer (binary classifier)

	W1 = tf.Variable(tf.truncated_normal([2, 2, 2, 3, K], stddev=0.1))  # 4x4 patch, 100 input channels, K output channels

	W2 = tf.Variable(tf.truncated_normal([4, 4, 4, K, L], stddev=0.1))
	B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
	W3 = tf.Variable(tf.truncated_normal([4, 4, 4, L, M], stddev=0.1))
	B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
	W4 = tf.Variable(tf.truncated_normal([4, 4, 4, M, N], stddev=0.1))
	B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
	W5 = tf.Variable(tf.truncated_normal([2, 4, 4, N, O], stddev=0.1))
	B5 = tf.Variable(tf.constant(0.1, tf.float32, [O]))

	#set dropout = True
	dropout = tf.constant(True)
	#Layer 1
	Y1l = tf.nn.conv3d(tensor, W1, strides=[1, 2, 2, 2, 1], padding='SAME')
	Y1r = utils.lrelu(Y1l, leak = 0.2)

	#layer 2
	Y2l = tf.nn.conv3d(Y1r, W2, strides=[1, 2, 2, 2, 1], padding='SAME')
	Y2bn, update_ema2 = utils.batchnorm(Y2l, dropout, iter, B2, convolutional=True)
	Y2r = utils.lrelu(Y2bn, leak = 0.2)

	#layer 3
	Y3l = tf.nn.conv3d(Y2r, W3, strides=[1, 2, 2, 2, 1], padding='SAME')
	Y3bn, update_ema3 = utils.batchnorm(Y3l, dropout, iter, B3, convolutional=True)
	Y3r = utils.lrelu(Y3bn, 0.2)

	#layer 4
	Y4l = tf.nn.conv3d(Y3r, W4, strides=[1, 2, 2, 2, 1], padding='SAME')
	Y4bn, update_ema3 = utils.batchnorm(Y4l, dropout, iter, B4, convolutional=True)
	Y4r = utils.lrelu(Y4bn, 0.2)

	#layer 4
	Y5l = tf.nn.conv3d(Y4r, W5, strides=[1, 1, 1, 1, 1], padding='VALID')
	Y5bn, update_ema3 = utils.batchnorm(Y5l, dropout, iter, B5, convolutional=True)

	#h0 = lrelu(tf.layers.conv3d(inputs = tensor, filters = 64, kernel_size = [4,4,4], strides = [2,2,2], padding = "SAME"), leak = 0.2)
	#h1 = lrelu(tf.contrib.layers.batch_norm(tf.layers.conv3d(inputs = h0, filters = 128, kernel_size = [4,4,4], strides = [2,2,2], padding = "SAME") , decay = .0001), leak = 0.2)
	#h2 = lrelu(tf.contrib.layers.batch_norm(tf.layers.conv3d(inputs = h1, filters = 256, kernel_size = [4,4,4], strides = [2,2,2], padding = "SAME") , decay = .0001), leak = 0.2)
	#h3 = lrelu(tf.contrib.layers.batch_norm(tf.layers.conv3d(inputs = h2, filters = 512, kernel_size = [4,4,4], strides = [2,2,2], padding = "SAME") , decay = .0001), leak = 0.2)
	#h4 = tf.layers.conv3d(inputs = h3, filters = 2, kernel_size = [2,4,4], strides = [1,1,1], padding = "VALID")
	return tf.Print(Y5bn, [Y5bn])

tf.reset_default_graph()

tensor = tf.truncated_normal(shape = [64, 64, 64, 2, 64])
tensor2 = tf.truncated_normal( shape =[64, 64, 64, 2, 64])
sess = tf.Session()

# test flag for batch norm
dropout = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
#
vnet = net_video(tensor)

#motion_net = tf.multiply(vnet, utils.replicate(tf.squeeze(mask_net(vnet, .0001)), sess, 3, 2))

tensor2d = tf.placeholder(tf.float32, [None, 1024])
tensor2d = tf.reshape(tensor2d, [4,4,4,4])

video = tf.truncated_normal(shape = [64, 32, 64, 64, 3]) #depth, height, width, dim

#sta_part = tf.matmul(utils.replicate(static_net(tensor2d, iter), sess, opt["frameSize"], 3), #
#	utils.replicate(tf.add(tf.multiply(tf.squeeze(mask_net(net_video(video), .0001)), -1), 1), sess, 32, 2))



#net = tf.

t = tf.image.decode_jpeg('converted_data/IMG_7491.JPG', 3, name = "import")
tf.Print(t, [t])

