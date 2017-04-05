import tensorflow as tf
import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

#tensor is replicated num_replications times, along dim dim 
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

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages