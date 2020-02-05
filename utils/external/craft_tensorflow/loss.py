import tensorflow as tf 
import numpy as np


FLAGS = tf.app.flags.FLAGS

def MSE_OHEM_Loss(output_imgs, target_imgs, confident_maps):
    loss_every_sample = []
    batch_size = FLAGS.batch_size

    for i in range(batch_size):
        output_img = tf.reshape(output_imgs[i], [-1])
        target_img = tf.reshape(target_imgs[i], [-1])
        conf_map = tf.reshape(tf.stack([confident_maps[i], confident_maps[i]], -1), [-1])
        positive_mask = tf.cast(tf.greater(target_img, CONFIG.threshold_positive), dtype=tf.float32)
        # fix
        sample_loss = tf.square(tf.subtract(output_img, target_img)) * conf_map
        
        num_all = output_img.get_shape().as_list()[0]
        num_positive = tf.cast(tf.reduce_sum(positive_mask), dtype=tf.int32)
        
        positive_loss = tf.multiply(sample_loss, positive_mask)
        positive_loss_m = tf.reduce_sum(positive_loss)/tf.cast(num_positive, dtype=tf.float32)
        negative_mask = tf.cast(tf.less_equal(target_img, CONFIG.threshold_negative), dtype=tf.float32)
        nagative_loss = tf.multiply(sample_loss, negative_mask) # fix
        # nagative_loss_m = tf.reduce_sum(nagative_loss)/(num_all - num_positive)

        k = num_positive * 3        
        #nagative_loss_topk, _ = tf.nn.top_k(nagative_loss, k)
        # tensorflow 1.13存在bug，不能使用以下语句 Orz。。。
        k = tf.cond((k + num_positive) > num_all, lambda: tf.cast((num_all - num_positive), dtype=tf.int32), lambda: k)
        k = tf.cond(k>0, lambda: k, lambda: k+1)   
        nagative_loss_topk, _ = tf.nn.top_k(nagative_loss, k)
        res = tf.cond(k < 10, lambda: tf.reduce_mean(sample_loss),
                              lambda: positive_loss_m + tf.reduce_sum(nagative_loss_topk)/tf.cast(k, dtype=tf.float32))
        loss_every_sample.append(res)
    return tf.reduce_mean(tf.convert_to_tensor(loss_every_sample))