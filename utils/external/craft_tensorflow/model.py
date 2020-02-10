import tensornets as nets
import tensorflow as tf

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


def craft(inputs, is_training):
    # with tf.variable_scope("preprocess"):
    #     inputs = tf.subtract(inputs,  [103.939, 116.779, 123.68], name='inputs') # Because we are using VVG16 pretrained model
    # optional

    with tf.variable_scope("encode"):
        # VGG16 = nets.VGG16(inputs, is_training=is_training, stem=True)
        VGG16, _ = vgg16(inputs, checkpoint=FLAGS.vgg_ckpt, is_input_trainable=is_training)
    # print('>>>>>> debug')
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    # print('>>>>>> debug')
    f = [
        tf.get_default_graph().get_tensor_by_name("model/encode/conv2_2:0"),
        tf.get_default_graph().get_tensor_by_name("model/encode/conv3_3:0"),
        tf.get_default_graph().get_tensor_by_name("model/encode/conv4_3:0"),
        tf.get_default_graph().get_tensor_by_name("model/encode/conv5_3:0"),
    ]  # name end points

    net = f[3]

    # VGG end
    with tf.variable_scope("stage6"):
        net = tf.layers.max_pooling2d(
            inputs=net, pool_size=3, strides=1, padding='SAME', name='pool5')
        net = arous_conv(net, 3, 3, 1024, 6, name='arous_conv')
        net = tf.layers.conv2d(inputs=net, filters=1024,
                               kernel_size=1, padding='SAME', name='conv6')

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
    return VGG16, net


def vgg16(inputs, checkpoint, is_input_trainable=False, fine_tune_last=False,
          n_classes=1000, input_shape=[None, 224, 224, 3]):
    """
    @params:
    is_input_trainable: True in case of Neural Style Transfer to generate images
    fine_tune_last: Make True if you want to fine tune the model for transfer learning
    n_classes: number of output classes
    input_shape:shape of input
    @returns
    model: dictionary of layers of model, example: conv1_1, conv1_2, ...
    params: dictionary of parameters of layers, example params['conv1_1']['W'] or params['conv1_1']['b']

    Note: model['input'] -> input layer of shape provided as argument
    model['out'] -> output layer of shape [1, 1, 1, n_classes] if input shape is [1, 224, 224, 3]
    """

    # checkpoint contains values of weights and biases as:
    # ckpt['vgg_16/conv1/conv1_1/weights'] or
    # ckpt['vgg_16/conv1/conv1_1/biases']
    path_conv = 'vgg_16/conv'
    path_fc = 'vgg_16/fc'
    ckpt_path = checkpoint
    file = tf.train.NewCheckpointReader(ckpt_path)

    def _weights(stage, block=None, type_code=0):
        if type_code == 0:
            path = path_conv + str(stage) + '/conv' + \
                str(stage) + '_' + str(block)
        else:
            path = path_fc + str(stage)
        w = file.get_tensor(path + '/weights')
        b = file.get_tensor(path + '/biases')
        return w, b

    def _conv2d(A_prev, W, strides=[1, 1], padding='SAME'):
        strides = [1, strides[0], strides[1], 1]
        return tf.nn.conv2d(A_prev, W, strides=strides, padding=padding)

    def conv_layer(A_prev, stage, block=None,
                   strides=[1, 1], padding='SAME',
                   freeze=True):
        w, b = _weights(stage, block)
        if freeze:
            w = tf.constant(w)
            b = tf.constant(b)
        else:
            w = tf.Variable(w)
            b = tf.Variable(b)
        c = _conv2d(A_prev, w, strides=strides, padding=padding)
        A = tf.nn.relu(tf.add(c, b), name='conv'+str(stage)+'_'+str(block))
        params = {'W': w, 'b': b}
        return A, params

    # freeze = False if you want to make a layer trainable.
    def fc_layer_wo_nonlin(A_prev, stage, is_final_layer=False, freeze=True):
        w, b = _weights(stage, type_code=1)
        if freeze:
            w = tf.constant(w)
            b = tf.constant(b)
        else:
            w = tf.Variable(w)
            b = tf.Variable(b)
        c = _conv2d(A_prev, w, padding='VALID')
        if is_final_layer:
            Z = tf.add(c, b, name='fc'+str(stage))
        else:
            Z = tf.add(c, b)
        params = {'W': w, 'b': b}
        return Z, params

    def fc_layer(A_prev, stage, freeze=True):
        Z, params = fc_layer_wo_nonlin(A_prev, stage, freeze=freeze)
        A = tf.nn.relu(Z, name='fc'+str(stage))
        params['Z'] = Z
        return A, params

    model = {}
    params = {}
    # max pool hyperparams
    KSIZE = [1, 2, 2, 1]
    STRIDES = [1, 2, 2, 1]
    PAD = 'VALID'

    # if is_input_trainable:
    #     X = tf.get_variable(name='input', shape=input_shape)
    # else:
    X = inputs
    model['input'] = X

    # conv1_1
    model['conv1_1'], params['conv1_1'] = conv_layer(X, 1, block=1)

    # conv1_2
    model['conv1_2'], params['conv1_2'] = conv_layer(
        model['conv1_1'], 1, block=2)

    # pool 1
    model['pool_1'] = tf.nn.max_pool(
        model['conv1_2'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv2_1
    model['conv2_1'], params['conv2_1'] = conv_layer(
        model['pool_1'], 2, block=1)

    # conv2_2
    model['conv2_2'], params['conv2_2'] = conv_layer(
        model['conv2_1'], 2, block=2)

    # pool_2
    model['pool_2'] = tf.nn.max_pool(
        model['conv2_2'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv3_1
    model['conv3_1'], params['conv3_1'] = conv_layer(
        model['pool_2'], 3, block=1)

    # conv3_2
    model['conv3_2'], params['conv3_2'] = conv_layer(
        model['conv3_1'], 3, block=2)

    # conv3_3
    model['conv3_3'], params['conv3_3'] = conv_layer(
        model['conv3_2'], 3, block=3)

    # pool_3
    model['pool_3'] = tf.nn.max_pool(
        model['conv3_3'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv4_1
    model['conv4_1'], params['conv4_1'] = conv_layer(
        model['pool_3'], 4, block=1)

    # conv4_2
    model['conv4_2'], params['conv4_2'] = conv_layer(
        model['conv4_1'], 4, block=2)

    # conv4_3
    model['conv4_3'], params['conv4_3'] = conv_layer(
        model['conv4_2'], 4, block=3)

    # pool_4
    model['pool_4'] = tf.nn.max_pool(
        model['conv4_3'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # conv5_1
    model['conv5_1'], params['conv5_1'] = conv_layer(
        model['pool_4'], 5, block=1)

    # conv5_2
    model['conv5_2'], params['conv5_2'] = conv_layer(
        model['conv5_1'], 5, block=2)

    # conv5_3
    model['conv5_3'], params['conv5_3'] = conv_layer(
        model['conv5_2'], 5, block=3)

    # pool_5
    model['pool_5'] = tf.nn.max_pool(
        model['conv5_3'], ksize=KSIZE, strides=STRIDES, padding=PAD)

    # fc6
    model['fc6'], params['fc6'] = fc_layer(model['pool_5'], 6)

    # fc7
    model['fc7'], params['fc6'] = fc_layer(model['fc6'], 7)

    # fc8
    if fine_tune_last:
        w = tf.get_variable('out_W', shape=[1, 1, 4096, n_classes])
        b = tf.get_variable('out_b', shape=[n_classes])
        model['out'] = tf.add(_conv2d(model['fc7'], w, padding='VALID'), b)
        params['out'] = {'W': w, 'b': b}
    else:
        model['out'], params['out'] = fc_layer_wo_nonlin(model['fc7'], 8)

    return model, params
