# https://github.com/taki0112/ResNet-Tensorflow

import ops_resnet
import tensorflow as tf

class ResNet(object):

    def __init__(self, feature_space_dimension, n_classes, n_res_blocks=18, margin_in_loss=0.25, is_train=True):
        self.img_size = 28
        self.c_dim = 1
        self.res_n = n_res_blocks
        self.feature_space_dimension = feature_space_dimension
        self.n_classes = n_classes

        self.x1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.x1Image = self.x1
        self.label_ = tf.placeholder(tf.int32, [None])

        # Create loss
        if is_train:
            with tf.variable_scope("resnet") as scope:
                self.o1_lastLayer = self.network(self.x1Image, is_training=True, reuse=False)
            self.loss = self.loss_of_network()
        else:
            with tf.variable_scope("resnet") as scope:
                self.o1_lastLayer = self.network(self.x1Image, is_training=False, reuse=False)

    # def load_network_model(self, session_):
    #     # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    #     print(" [*] Reading checkpoints...")
    #     saver = tf.train.Saver()
    #     checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir_)
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         saver.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
    #         print(" [*] Success to read {}".format(ckpt_name))
    #         latest_epoch = int(ckpt_name[-1])
    #         return True, latest_epoch
    #     else:
    #         print(" [*] Failed to find a checkpoint")
    #         return False, 0

    def save_network(self, session_, checkpoint_dir):
        # https://stackoverflow.com/questions/46549056/can-tensorflow-save-the-variables-in-a-certain-variable-scope
        # saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "network"))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "resnet"))
        saver.save(session_, checkpoint_dir + "\\model.ckpt")

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            if self.res_n < 50 :
                residual_block = ops_resnet.resblock
            else :
                residual_block = ops_resnet.bottle_resblock

            residual_list = ops_resnet.get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = ops_resnet.conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = ops_resnet.batch_norm(x, is_training, scope='batch_norm')
            x = ops_resnet.relu(x)

            x = ops_resnet.global_avg_pooling(x)
            x_oneToLast = ops_resnet.fully_conneted(x, units=self.feature_space_dimension, scope='logit_1')

            self.o1 = x_oneToLast

            x_last = ops_resnet.fully_conneted(x_oneToLast, units=self.n_classes, scope='logit_2')

            return x_last

    def loss_of_network(self):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_, logits=self.o1_lastLayer))  #--> needs one-hot-encoding of labels
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_, logits=self.o1_lastLayer))
        return loss
