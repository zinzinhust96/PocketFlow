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
        arousconv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(arousconv, biases)
        return bias

def conv2d(inputs, n_kernels, namescope, kernel_size=(3, 3), stride = 1, padding = 'SAME'):
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
    # with tf.variable_scope("preprocess"):
    #     inputs = tf.subtract(inputs,  [103.939, 116.779, 123.68], name='inputs') # Because we are using VVG16 pretrained model
    # optional

    with tf.variable_scope("encode"):
        # VGG16 = nets.VGG16(inputs, is_training=is_training, stem=True)
        # VGG16, _ = vgg16(inputs, checkpoint=FLAGS.vgg_ckpt, is_input_trainable=is_training)
        # vgg16_model = VGG16(inputs)
        f = VGG16(inputs)

    # print('>>>>>> debug')
    # for n in tf.get_default_graph().get_operations():
    #     if 'decode' in n.name:
    #         print(n.values())
    # print('>>>>>> debug')
    # f = [
    #     tf.get_default_graph().get_tensor_by_name("model/encode/VGG16/conv2_2/Relu:0"),
    #     tf.get_default_graph().get_tensor_by_name("model/encode/VGG16/conv3_3/Relu:0"),
    #     tf.get_default_graph().get_tensor_by_name("model/encode/VGG16/conv4_3/Relu:0"),
    #     tf.get_default_graph().get_tensor_by_name("model/encode/VGG16/conv5_3/Relu:0"),
    # ]  # name end points

    net = f[3]
    # VGG end
    with tf.variable_scope("stage6"):
        net = tf.layers.max_pooling2d(
            inputs=net, pool_size=3, strides=1, padding='SAME', name='pool5')
        net = arous_conv(net, 3, 3, 1024, 6, name='conv6')
        # net = conv2d(net, 1024, 'conv6')
        net = tf.layers.conv2d(inputs=net, filters=1024,
                               kernel_size=1, padding='SAME', name='conv7')

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
    # return tf.get_default_graph().get_tensor_by_name('model/decode/output/heatmaps/BiasAdd:0')
    
    return net

def VGG16(inputs):
    # with tf.variable_scope("encode"):
    with tf.variable_scope('VGG16'):
        conv1_1 = conv2d(inputs, 64, 'conv1_1')
        conv1_2 = conv2d(conv1_1, 64, 'conv1_2')
        pool1 = maxpooling(conv1_2, 'pool1')
        conv2_1 = conv2d(pool1, 128, 'conv2_1')
        conv2_2 = conv2d(conv2_1, 128, 'conv2_2')
        pool2 = maxpooling(conv2_2, 'pool2')

        conv3_1 = conv2d(pool2, 256, 'conv3_1')
        conv3_2 = conv2d(conv3_1, 256, 'conv3_2')
        conv3_3 = conv2d(conv3_2, 256, 'conv3_3')
        pool3 = maxpooling(conv3_3, 'pool3')

        conv4_1 = conv2d(pool3, 512, 'conv4_1')
        conv4_2 = conv2d(conv4_1, 512, 'conv4_2')
        conv4_3 = conv2d(conv4_2, 512, 'conv4_3')
        pool4 = maxpooling(conv4_3, 'pool4')

        conv5_1 = conv2d(pool4, 512, 'conv5_1')
        conv5_2 = conv2d(conv5_1, 512, 'conv5_2')
        conv5_3 = conv2d(conv5_2, 512, 'conv5_3')
    
    # with tf.variable_scope("stage6"):
    #     net = maxpooling(conv5_3, 'pool5', (3,3), 1)
    #     net = conv2d(net, 1024, 'conv6')
    #     net = conv2d(net, 1024, 'conv7', kernel_size=(1,1))

    return conv2_2, conv3_3, conv4_3, conv5_3



# class VGG16(object):
#     def __init__(self, inputs, n_classes=1000):
#         self.n_classes = n_classes
#         self.inputs = inputs        
#         with tf.variable_scope('VGG16'):
#             self.build()
#         # self.output = tf.nn.softmax(self.fc3)

#     def build(self):
#         self.conv1_1 = self.conv2d(self.inputs, 64, 'conv1_1')
#         self.conv1_2 = self.conv2d(self.conv1_1, 64, 'conv1_2')
#         self.pool1 = self.maxpooling(self.conv1_2, 'pool1')
#         self.conv2_1 = self.conv2d(self.pool1, 128, 'conv2_1')
#         self.conv2_2 = self.conv2d(self.conv2_1, 128, 'conv2_2')
#         self.pool2 = self.maxpooling(self.conv2_2, 'pool2')

#         self.conv3_1 = self.conv2d(self.pool2, 256, 'conv3_1')
#         self.conv3_2 = self.conv2d(self.conv3_1, 256, 'conv3_2')
#         self.conv3_3 = self.conv2d(self.conv3_2, 256, 'conv3_3')
#         self.pool3 = self.maxpooling(self.conv3_3, 'pool3')

#         self.conv4_1 = self.conv2d(self.pool3, 512, 'conv4_1')
#         self.conv4_2 = self.conv2d(self.conv4_1, 512, 'conv4_2')
#         self.conv4_3 = self.conv2d(self.conv4_2, 512, 'conv4_3')
#         self.pool4 = self.maxpooling(self.conv4_3, 'pool4')

#         self.conv5_1 = self.conv2d(self.pool4, 512, 'conv5_1')
#         self.conv5_2 = self.conv2d(self.conv5_1, 512, 'conv5_2')
#         self.conv5_3 = self.conv2d(self.conv5_2, 512, 'conv5_3')
#         # self.pool5 = self.maxpooling(self.conv5_3, 'pool5')     

#         # shape = int(np.prod(self.pool5.get_shape()[1:]))
#         # self.flat = tf.reshape(self.pool5, [-1, shape])
#         # self.fc1 = self.fc(self.flat, shape, 4096, 'fc1')
#         # self.fc2 = self.fc(self.fc1, 4096, 4096, 'fc2')
#         # self.fc3 = self.fc(self.fc2, 4096, self.n_classes, 'fc3', relu=False)


#     def conv2d(self, inputs, n_kernels, namescope, kernel_size=(3, 3), stride = 1, padding = 'SAME'):
#         with tf.variable_scope(namescope, reuse=tf.AUTO_REUSE):
#             w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1], inputs.shape[-1], n_kernels],
#                             initializer=tf.initializers.truncated_normal(stddev=1e-1), dtype=tf.float32)
#             biases = tf.get_variable('biases', shape=[n_kernels], initializer=tf.constant_initializer(), 
#                                      dtype=tf.float32)
            
#             conv = tf.nn.conv2d(input=inputs, filter=w, strides=[1, stride, stride, 1], padding = padding)
#             out = tf.nn.bias_add(conv, biases)   
#             return tf.nn.relu(out)

#     def maxpooling(self, layer_input, namescope, kernel_size=(2, 2), stride = 2, padding = 'SAME'):
#         return tf.nn.max_pool(layer_input, ksize=[1, kernel_size[0], kernel_size[1], 1],
#                             strides=[1, stride, stride, 1], padding=padding, name=namescope)

