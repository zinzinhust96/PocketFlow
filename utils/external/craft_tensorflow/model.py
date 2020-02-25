import tensornets as nets
import tensorflow as tf
import numpy as np 

FLAGS = tf.app.flags.FLAGS

def upsample(inputs, name):
    '''
    Up 2x size
    '''
    # net = tf.image.resize_bilinear(inputs, new_size, name=name) # size of image is arbitrary ==> I have to use Conv2d Transpose
    num_filters = inputs.shape[-1]
    kernel_size = 1
    padding = "SAME"
    strides = 2
    net = tf.layers.conv2d_transpose(
        inputs, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)

    return net


def upconvBlock(inputs, midchannels, outchannels, is_training, name):
    with tf.variable_scope(name):
        # default is "VALID" padding
        net = tf.layers.conv2d(inputs, midchannels, 1)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net, name='relu1')
        net = tf.layers.conv2d(net, outchannels, 3, padding='SAME')
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net, name='relu2')
    return net


def arous_conv(x, filter_height, filter_width, num_filters, rate, name):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the arous_conv layer
        weights = tf.get_variable(
            'weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        arousconv = tf.nn.atrous_conv2d(x, weights, rate=rate, padding='SAME')
        bias = tf.nn.bias_add(arousconv, biases)
        return bias

def conv2d(inputs, n_kernels, namescope = 'conv', kernel_size=(3, 3), stride = 1, padding = 'SAME'):
        with tf.variable_scope(namescope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1], inputs.shape[-1], n_kernels],
                            initializer=tf.initializers.truncated_normal(stddev=1e-1), dtype=tf.float32)
            biases = tf.get_variable('biases', shape=[n_kernels], initializer=tf.constant_initializer(), 
                                     dtype=tf.float32)
            
            conv = tf.nn.conv2d(input=inputs, filter=w, strides=[1, stride, stride, 1], padding = padding)
            out = tf.nn.bias_add(conv, biases)   
            return tf.nn.relu(out)

def maxpooling(layer_input, namescope, kernel_size=(2, 2), stride = 2, padding = 'SAME'):
        return tf.nn.max_pool(layer_input, ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=[1, stride, stride, 1], padding=padding, name=namescope)

def craft(inputs, is_training):

    with tf.variable_scope("encode"):
        f = VGG16(inputs)

    net = f[3]
    # VGG end
    with tf.variable_scope("stage6"):
        net = tf.layers.max_pooling2d(
            inputs=net, pool_size=3, strides=1, padding='SAME', name='pool5')
        # net = arous_conv(net, 3, 3, 1024, 6, name='arous_conv')
        net = conv2d(net, 1024, 'arous_conv')
        net = conv2d(net, 1024, namescope='conv6', kernel_size= (1, 1))
        # net = tf.layers.conv2d(inputs=net, filters=1024,
        #                        kernel_size=1, padding='SAME', name='conv6')

    # U-net
    with tf.variable_scope("decode"):
        net = tf.concat([net, f[3]], axis=3, name='concat1')
        net = upconvBlock(
            net, 512, 256, is_training=is_training, name='up_conv_1')

        net = upsample(net, name='upsample1')

        net = tf.concat([net, f[2]], axis=3, name='concat2')  # w/8 256 + 512
        net = upconvBlock(net, 256, 128, is_training=is_training,
                          name='up_conv_2')  # w/8 128
        net = upsample(net, name='upsample2')

        net = tf.concat([net, f[1]], axis=3, name='concat3')    # w/4 128 + 256
        net = upconvBlock(net, 128, 64, is_training=is_training,
                          name='up_conv_3')  # w/4 64
        net = upsample(net, name='upsample3')

        net = tf.concat([net, f[0]], axis=3, name='concat4')  # w/2 64 + 128
        net = upconvBlock(net, 64, 32, is_training=is_training,
                          name='up_conv_4')      # w/2 32
        net = upsample(net, name='upsample4')

        with tf.variable_scope("output"):
            net = tf.layers.conv2d(net, 32, 3, padding='SAME')
            net = tf.layers.conv2d(net, 32, 3, padding='SAME')
            net = tf.layers.conv2d(net, 16, 3, padding='SAME')
            net = tf.layers.conv2d(net, 16, 1, padding='SAME')
            net = tf.layers.conv2d(net, 2, 1, padding='SAME', name='heatmaps')
    return net


def VGG16(inputs):
    # with tf.variable_scope("encode"):
    with tf.variable_scope('vgg16'):
        with tf.variable_scope('conv1'):
            with tf.variable_scope('1'):
                conv1_1 = conv2d(inputs, 64)
            with tf.variable_scope('2'):
                conv1_2 = conv2d(conv1_1, 64)
        pool1 = maxpooling(conv1_2, 'pool1')

        with tf.variable_scope('conv2'):
            with tf.variable_scope('1'):
                conv2_1 = conv2d(pool1, 128)
            with tf.variable_scope('2'):
                conv2_2 = conv2d(conv2_1, 128)
        pool2 = maxpooling(conv2_2, 'pool2')

        with tf.variable_scope('conv3'):
            with tf.variable_scope('1'):
                conv3_1 = conv2d(pool2, 256)
            with tf.variable_scope('2'):
                conv3_2 = conv2d(conv3_1, 256)
            with tf.variable_scope('3'):
                conv3_3 = conv2d(conv3_2, 256)
        pool3 = maxpooling(conv3_3, 'pool3')

        with tf.variable_scope('conv4'):
            with tf.variable_scope('1'):
                conv4_1 = conv2d(pool3, 512)
            with tf.variable_scope('2'):
                conv4_2 = conv2d(conv4_1, 512)
            with tf.variable_scope('3'):
                conv4_3 = conv2d(conv4_2, 512)
        pool4 = maxpooling(conv4_3, 'pool4')

        with tf.variable_scope('conv5'):
            with tf.variable_scope('1'):
                conv5_1 = conv2d(pool4, 512,)
            with tf.variable_scope('2'):
                conv5_2 = conv2d(conv5_1, 512)
            with tf.variable_scope('3'):
                conv5_3 = conv2d(conv5_2, 512)

    return conv2_2, conv3_3, conv4_3, conv5_3
