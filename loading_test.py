import tensorflow as tf


def get_batch(batch_size):
	# Make a queue of file names including all the JPEG images files in the relative
	# image directory.
	filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once("./converted_data/*.JPG"))
	# Read an entire image file which is required since they're JPEGs, if the images
	# are too large they could be split in advance to smaller files or use the Fixed
	# reader to split up the file.
	image_reader = tf.WholeFileReader()

	frame_size = 32

	batch = []
	for v in xrange(batch_size):
		video = []
		for x in xrange(frame_size):
			# Read a whole file from the queue, the first returned value in the tuple is the
			# filename which we are ignoring.
			_, image_file = image_reader.read(filename_queue)

			# Decode the image as a JPEG file, this will turn it into a Tensor which we can
			# then use in training.
			image = tf.image.decode_jpeg(image_file)

			#crop to a 64 x 64 image
			image = tf.image.resize_images(image, [64, 64])
			image.set_shape((64, 64, 3))
			video.append(image)
		batch.append(tf.stack(video, name = 'single_video'))

	return tf.stack(batch, name = 'video_batch')

if __name__ == '__main__':
	# Start a new session to show example output.
	with tf.Session() as sess:
		# Required to get the filename matching to run.
		tf.global_variables_initializer().run()
		image = get_batch(2)
		# # Coordinate the loading of image files.
		# coord = tf.train.Coordinator()
		# threads = tf.train.start_queue_runners(coord=coord)
		print image
		tf.Print(image, [image])

		# # Finish off the filename queue coordinator.
		# #images_tensor = sess.run([images])
		# #print(images_tensor)
		# coord.request_stop()
		# coord.join(threads)
