import tensorflow as tf
from ops import *
import imageflow

def static_net(tensor):
	input_shape = tensor.get_shape().as_list()
	batch_size = input_shape[0]
	static_w = input_shape[1]
	static_h = input_shape[2]
	dim = input_shape[-1]
	#tensor = tf.reshape(tensor, [batch_size, static_h, static_w, dim])

	h0 = tf.nn.relu(batch_norm(deconv2d(tensor, [batch_size, static_h, static_w, 512], name = "st_h0_decov")))
	h1 = tf.nn.relu(batch_norm(deconv2d(h0, [batch_size, static_h*2, static_w*2, 256], name = 'st_h1_deconv'))) #512
	h2 = tf.nn.relu(batch_norm(deconv2d(h1, [batch_size, static_h*4, static_w*4, 128], name = 'st_h2_deconv'))) #256
	h3 = tf.nn.relu(batch_norm(deconv2d(h2, [batch_size, static_h*8, static_w*8, 64], name = 'st_h3_deconv'))) #128
	h4 = tf.nn.tanh(deconv2d(h3, [batch_size, static_h*16, static_w*16, 3], name = 'st_h4_deconv')) #32, 64, 64,3
	return h4

def net_video(tensor):
	input_shape = tensor.get_shape().as_list()
	batch_size = input_shape[0]
	net_w = input_shape[1]
	net_h = input_shape[2]
	net_d = input_shape[3]
	dim = input_shape[-1]
	#tensor = tf.reshape(tensor, [batch_size, static_h, static_w, dim])

	h0 = tf.nn.relu(batch_norm(deconv3d(tensor, [batch_size, net_d, net_h, net_w, 512], name = "net_h0_deconv")))
	h1 = tf.nn.relu(batch_norm(deconv3d(h0, [batch_size, net_d*2, net_h*2, net_w*2, 256], name = 'net_h1_deconv'))) #512
	h2 = tf.nn.relu(batch_norm(deconv3d(h1, [batch_size, net_d*4, net_h*4, net_w*4, 128], name = 'net_h2_deconv'))) #256
	h3 = tf.nn.relu(batch_norm(deconv3d(h2, [batch_size, net_d*8, net_h*8, net_w*8, 64], name = 'net_h3_deconv'))) #128
	return h3

def mast_net(tensor):
	input_shape = tensor.get_shape().as_list()
	batch_size = input_shape[0]
	net_w = input_shape[1]
	net_h = input_shape[2]
	net_d = input_shape[3]
	dim = input_shape[-1]

	return tf.nn.sigmoid(deconv3d(tensor, [batch_size, net_d, net_h, net_w, 1], name = "mask_deconv"))

image = tf.truncated_normal(shape= [32, 64, 64, 3])
video = tf.truncated_normal(shape = [32, 32, 64, 64, 3]) #depth, height, width, dim

#print tf.Print(static_net(image), [image])
#print tf.Print(net_video(video), [video])
picture = tf.truncated_normal(shape= [1, 5184, 3456, 10])
label = tf.truncated_normal(shape= [1, 5184, 3456, 10])

filename_queue = tf.train.string_input_producer(['IMG_7491.jpg']) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value)
with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	image = my_img.eval()
	print(image.shape)
	Image.fromarray(np.asarray(image)).show()
#print tf.Print(tf.multiply(static_net(image), mast_net(net_video(video))), [video])
