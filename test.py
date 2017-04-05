import tensorflow as tf
import numpy as np
#import tensorflow.nn.relu

def replicate(tensor, sess, num_replications, dim):
	o_rank = tf.shape(tensor)
	tensor = tf.reshape(tensor, [-1])
	tensor = tf.tile(tensor, [num_replications])

	shape = (o_rank.eval(session = sess))

	free_dim = np.array([-1])
	s = np.append(shape[0], free_dim)
	s = np.append(s, shape[1::])
	tensor = tf.reshape(tensor, s)
	return tf.Print(tensor,[tensor], summarize = 32)


c = tf.constant([[[1.0, 3.0], [2.0, 3.0]], [[4.0, 5.0], [2.0, 3.0]]])

h0 = tf.nn.relu(tf.contrib.layers.batch_norm(c))
#tf.python.layers
#tf.layers.conv3d(inputs = c, filters = 512, kernel_size = [2,4,4], strides = [1,1,1], padding = "VALID")

sess = tf.Session() #o_rank[0], o_rank[1]



sess.run(replicate(c, sess, 3, 1))

#c = tf.reshape(c, [1, -1])
#print c
