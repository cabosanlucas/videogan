import tensorflow as tf
from ops import*
from util2 import*

class VideoGAN_Conditional(object):

    def __init__(self, sess, dataset='video2', batchSize=32, 
            loadSize=128, fineSize=64, frameSize=32,
            lr=0.0002,lr_decay=1000,beta1=0.5, niter=100, gpu=1, 
            data_root='/data/vision/torralba/crossmodal/flickr_videos/', 
            data_list='data/vision/torralba/crossmodal/flicr_videos/scene_extract/lists-full/_b_beach.txt.train'):
        self.sess = sess
        self.batchSize = batchSize
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.frameSize = frameSize
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.niter = niter
        self.gpu = gpu
        self.data_root = data_root
        self.data_list = data_list
        
        """
        loadSize: when loading images, resize first to this size
        fineSize: crop this size from the loaded image
        lr: learning rate
        lr_decay: how often to decay learning rate inepochs
        beta1: momentum term for adam
        niter: number of iterations through dataset
        gpu: which GPU to use
        """
        self.build_model()
    
    def encode_net(self,image): 
        #encode net(2d encoding)
        enc_dim = self.loadSize
        h1 = tf.nn.relu(conv2d(image, enc_dim, name = 'g_enc_h1_conv')) #batch, 32, 32, 128
        h2 = tf.nn.relu(batch_norm(conv2d(h1, enc_dim*2, name = 'g_enc_h2_conv')))
        h3 = tf.nn.relu(batch_norm(conv2d(h2, enc_dim*4, name = 'g_enc_h3_conv')))
        h4 = tf.nn.relu(batch_norm(conv2d(h3, enc_dim*8, name = 'g_enc_h4_conv'))) #batch, 4, 4, 1024
        return h4

    def static_net(self, input_):
        #static net(2d decoding)
        input_shape = input_.get_shape().as_list()
        st_dim = input_shape[-1]/2
        st_h = input_shape[1]*2
        sh_w = input_shape[2]*2
        h1 = tf.nn.relu(batch_norm(deconv2d(input_, [self.batchSize, st_h, st_w, st_dim], name = 'g_st_h1_deconv')))
        h2 = tf.nn.relu(batch_norm(deconv2d(h1, [self.batchSize, st_h*2, st_w*2, st_dim/2], name = 'g_st_h2_deconv')))
        h3 = tf.nn.relu(batch_norm(deconv2d(h2, [self.batchSize, st_h*4, st_w*4, st_dim/4], name = 'g_st_h3_deconv')))
        h4 = tf.nn.tanh(deconv2d(h3, [self.batchSize, st_h*8, st_w*8, st_dim/8], name = 'g_st_h4_deconv'))
        return h4

    def net_video(self, input_):
        #net video(3d decoding)
        input_shape = input_.get_shape().as_list()
        net_dim = input_shape[-1]
        net_h = input_shape[1]
        net_w = input_shape[2]
        net_d = 2
        input_ = tf.reshape(input_, [self.batchSize, 1, net_h, net_w, net_dim])
        h1 = tf.nn.relu(batch_norm(deconv3d(input_, [self.batchSize, net_d, net_h, net_w, net_dim/2], name = 'g_net_h1_deconv')))
        h2 = tf.nn.relu(batch_norm(deconv3d(h1, [self.batchSize, net_d*2, net_h*2, net_w*2, net_dim/4], name = 'g_net_h2_deconv')))
        h3 = tf.nn.relu(batch_norm(deconv3d(h2, [self.batchSize, net_d*4, net_h*4, net_w*4, net_dim/8], name = 'g_net_h3_deconv')))
        h4 = tf.nn.relu(batch_norm(deconv3d(h3, [self.batchSize, net_d*8, net_h*8, net_w*8, net_dim/16], name = 'g_net_h4_deconv')))
        return h4   #batch, 16, 32, 32, 64

    def mask_out(self, input_):
        #input = batch_size, 16, 32, 32, 64
        input_shape = input_.get_shape().as_list()
        m_d = g_d = input_shape[1]
        m_h = g_h = input_shape[2]
        m_w = g_w = input_shape[3]

        #mask net (batch, 32, 64, 64, 1)
        mask_net = tf.nn.sigmoid(decov3d(input_, [self.batchSize, m_d*2, m_h*2, m_w*2, 1], name = 'g_mask_h1_deconv'))

        #gen net (batch, 32, 64, 64, 3)
        gen_net = tf.tanh(deconv3d(input_, [self.batchSize, g_d*2, g_h*2, g_w*2, 3], name = 'g_gen_h1_deconv'))
        return mask_net, gen_net

    def discriminator_net(self, video, reuse = False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            #input = batch, 32, 64, 64, 3
            v_dim = self.loadSize
            h1 = lrelu(conv3d(video, v_dim, name = 'disc_h1_conv'))
            h2 = lrelu(batch_norm(conv3d(h1, v_dim*2, name = 'disc_h2_conv')))
            h3 = lrelu(batch_norm(conv3d(h2, v_dim*4, name = 'disc_h3_conv')))
            h4 = lrelu(batch_norm(conv3d(h3, v_dim*8, name = 'disc_h4_conv')))
            h5 = conv3d(h4, 2, f_d = 2, f_h = 4, f_w = 4, name = 'disc_h5_conv')
            h6 = conv3d(h5, 2, 1, 2, 2, name = 'disc_h6_conv') # batch, 1, 1, 1, 2
            return h6
    
    def Generator(self, image):
        enc = encode_net(image)
        back = static_net(enc)
        mask, fore = mask_out(net_video(enc))
        netG = tf.add(tf.multiply(mask,fore), tf.multiply(tf.subtract(tf.ones_like(mask),mask),back[:,None,:,:,:]))
        return netG

    def Discriminator(self, video, reuse = False):
        netD = discriminator_net(video, reuse)
        return netD

    def build_model(self):
        #input data placeholder(image for generator, video for discriminator)
        self.video = tf.placeholder(tf.float32, [self.batchSize, self.frameSize, self.fineSize, self.fineSize, 3], name = 'input_videos')
        self.image = tf.placeholder(tf.float32, [self.batchSize, self.fineSize, self.fineSize, 3], name = 'input_image')
        
        self.gen_dat = Generator(self.image)
        self.d_fake_logits = Discriminator(self.gen_dat)
        self.d_real_logits = Discriminator(self.video, reuse = True)

        """
        paper uses different cross entropy: log softmax + ClassNLLCriterion
        """
        self.d_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.d_real_logits, tf.ones_like(self.d_real_logits)))
        self.d_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.d_fake_logits, tf.zeros_like(self.d_fake_logits)))
        self.d_loss = self.d_real_loss + self.d_fake_loss

        self.g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.d_fake_logits, tf.ones_like(self.d_fake_logits)))
        
        self.reg_gen =  tf.transpose(gen_dat, [4,0,1,2,3])
        self.reg_frame = tf.transpose(self.video, [4,0,1,2,3])
        self.reg_loss = tf.reduce_mean(tf.abs(tf.subtract(self.reg_gen[0], self.reg_frame[0])))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.lr, beta1 = self.beta1).minimize(self.d_loss, var_list = self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1 = self.beta1).minimize(self.g_loss, var_list = self.g_vars)

        self.sess.run(tf.global_variables_initializer())
        
        #load data
        counter = 1
        for epochs in xrange(self.niter):
            for idx in xrange(0, batch_idxs):
                
                batch_videos = tf.truncated_normal(shape = [self.batchSize, self.frameSize, self.fineSize, self.fineSize, 3])
                batch_images = tf.truncated_normal(shape = [self.batchSize, self.fineSize, self.fineSize, 3])
                #update discriminator
                _, d_loss_curr = self.sess.run([d_optim, self.d_loss], feed_dict = {self.video: batch_videos, self.image: batch_images})
                #update generator
                _, g_loss_curr = self.sess.run([g_optim, self.g_loss], feed_dict = {self.image: batch_images})

                if counter %100==0:
                    print('epochs: '+epochs+' d_loss = ' + d_loss_curr + ' g_loss = ' + g_loss_curr)
                    #make gif
                counter +=1
if __name__ == "__main__":
    sess = tf.Session()
    videogan_cond = VideoGAN_Conditional(sess)
    videogan_cond.train()
