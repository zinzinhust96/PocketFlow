import tensornets as nets
import tensorflow as tf

def upsample(inputs, name):
    '''
    Up 2x size
    '''
    #net = tf.image.resize_bilinear(inputs, new_size, name=name) # size of image is arbitrary ==> I have to use Conv2d Transpose
    num_filters = inputs.shape[-1]
    kernel_size = 1
    padding = "SAME"
    strides = 2
    net = tf.layers.conv2d_transpose(inputs, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)

    return net

def upconvBlock(inputs, midchannels, outchannels, is_training, name):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(inputs, midchannels, 1) # default is "VALID" padding
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
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        arousconv = tf.nn.atrous_conv2d(x, weights, rate=rate, padding='SAME')
        bias = tf.nn.bias_add(arousconv, biases)
        return bias

def craft(inputs, is_training):
    # with tf.variable_scope("preprocess"):
    #     inputs = tf.subtract(inputs,  [103.939, 116.779, 123.68], name='inputs') # Because we are using VVG16 pretrained model
    # optional

    with tf.variable_scope("encode"):
        VGG16 = nets.VGG16(inputs, is_training=is_training, stem=True)

    f = [
        tf.get_default_graph().get_tensor_by_name("encode/vgg16/conv2/2/Relu:0"), 
        tf.get_default_graph().get_tensor_by_name("encode/vgg16/conv3/3/Relu:0"), 
        tf.get_default_graph().get_tensor_by_name("encode/vgg16/conv4/3/Relu:0"), 
        tf.get_default_graph().get_tensor_by_name("encode/vgg16/conv5/3/Relu:0"), 
    ] # name end points

    net = f[3]

    # VGG end
    with tf.variable_scope("stage6"):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=1, padding='SAME',name='pool5')
        net = arous_conv(net, 3, 3, 1024, 6, name='arous_conv')
        net = tf.layers.conv2d(inputs=net, filters=1024, kernel_size=1, padding='SAME', name='conv6')

    # U-net
    with tf.variable_scope("decode"):
        net = tf.concat([net, f[3]], axis=3, name='concat1')
        net = upconvBlock(net, 512, 256, is_training=is_training, name='up_conv_1')
        
        net = upsample(net, name='upsample1')

        net = tf.concat([net, f[2]], axis=3, name='concat2')  # w/8 256 + 512
        net = upconvBlock(net, 256, 128, is_training=is_training, name='up_conv_2')  # w/8 128
        net = upsample(net, name='upsample2')

        net = tf.concat([net, f[1]], axis=3, name='concat3')    # w/4 128 + 256
        net = upconvBlock(net, 128, 64, is_training=is_training, name='up_conv_3')  # w/4 64
        net = upsample(net, name='upsample3')

        net = tf.concat([net, f[0]], axis=3, name='concat4')  # w/2 64 + 128
        net = upconvBlock(net, 64, 32, is_training=is_training, name='up_conv_4')      # w/2 32
        net = upsample(net, name='upsample4')

        with tf.variable_scope("output"):
            net = tf.layers.conv2d(net, 32, 3, padding='SAME')
            net = tf.layers.conv2d(net, 32, 3, padding='SAME')
            net = tf.layers.conv2d(net, 16, 3, padding='SAME')
            net = tf.layers.conv2d(net, 16, 1, padding='SAME')
            net = tf.layers.conv2d(net, 2, 1, padding='SAME', name='heatmaps')
    return VGG16, net
