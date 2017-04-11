import tensorflow as tf

opt : {
  'dataset' : 'video3',   #indicates what dataset load to use (in data.lua)
  'nThreads' : 16,        #how many threads to pre-fetch data
  'batchSize' : 256,      #self-explanatory
  'loadSize' : 256,       #when loading images, resize first to this size
  'fineSize' : 64,       #crop this size from the loaded image 
  'frameSize' : 32,
  'lr' : 0.0002,          #learning rate
  'lr_decay' : 1000,         #how often to decay learning rate (in epoch's)
  'lambda' : 0.1,
  'beta1' : 0.5,          #momentum term for adam
  'meanIter' : 0,         #how many iterations to retrieve for mean estimation
  'saveIter' : 100,    #write check point on this interval
  'niter' : 100,          #number of iterations through dataset
  'max_iter' : 1000,
  'ntrain' : math.huge,   #how big one epoch should be
  'gpu' : 1,              #which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  'cudnn' : 1,            #whether to use cudnn or not
  'name' : 'ucf101',        #the name of the experiment
  'randomize' : 1,        #whether to shuffle the data file or not
  'cropping' : 'random',  #options for data augmentation
  'display_port' : 8001,  #port to push graphs
  'display_id' : 1,       #window ID when pushing graphs
  'mean' : (0,0,0),
  'data_root' : '/data/vision/torralba/hallucination/UCF101/frames-stable-nofail/videos',
  'data_list' : '/data/vision/torralba/hallucination/UCF101/gan/train.txt'
}

tf.set_random_seed(0)

if opt['gpu'] > 0:
	#set device gpu



