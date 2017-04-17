import numpy as np
import tensorflow as tf
import os
from PIL import Image

opt = {
  'model' : 'models/golf/iter65000_net.t7',
  'batchSize' : 128,
  'gpu' : 1,
  'cudnn' : 1,
}

train = 'false'

tf.set_random_seed(0)

#TODO: How to set device?

sess = tf.Session(log_device_placement=True)
importer = tf.train.import_meta_graph(opt['model'])
importer.restore(sess, tf.train.latest_checkpoint('./'))

noise = tf.random_normal((opt['batchSize'], 100))
gen = sess.run(y, feed_dict={x: noise})
video = sess.run(h4)[0]
mask = sess.run(h4)[1]
static = sess.run(h4)[2]
mask = tf.tile(mask, tf.constant([1,3,1,1,1]))


def WriteGif(filename, movie):
	for i in range(1, movie.shape[2]):
		Image.fromarray(movie.eval()[3]).save()
	cmd = 'ffmpeg -f image2 -i ' + filename + '.%08d.png -y ' + filename
	print('==> ' + cmd)
	os.system(cmd)
	for i in range(1, movie.shape[2]):
		os.remove(filename + '.%08d.png'.format(i))

if not os.path.exists('vis/'):
	os.mkdir('vis/')
WriteGif('vis/gen.gif', gen)
WriteGif('vis/video.gif', video)
WriteGif('vis/videomask.gif', torch.cmul(video, mask))
WriteGif('vis/mask.gif', mask)
image.save("vis/static", "jpg")
