import tensorflow as tf


def lrelu(x, leak=0.1):
    return tf.maximum(x, leak * x)


def creating_embedding_matrix(input_size, embedding_dim, name):
    init = tf.contrib.layers.xavier_initializer()
    embedding_matrix = tf.get_variable(
        name=name,
        shape=[input_size, embedding_dim],
        initializer=init,
        trainable=True,
        dtype=tf.float32
    )
    return embedding_matrix


def compute_penalty(inter_sample_pred, inter_sample):
    gradients = tf.gradients(inter_sample_pred, [inter_sample])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    return tf.reduce_mean((slopes - 1.) ** 2)


def masking(inputs, length, phn_size):
    mask = tf.sequence_mask(length, tf.shape(inputs)[1])
    mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, phn_size])
    paddings = tf.zeros_like(inputs)
    return tf.where(mask, inputs, paddings)


def frame2phn_network(input_frame, config):
    # Reshape to [-1, feature_len]
    frame_shape = input_frame.get_shape().as_list()
    input_frame = tf.reshape(input_frame, [-1, int(frame_shape[2])])

    with tf.variable_scope('frame2phn_network') as scope:
        # Phoneme Classifier
        outputs = tf.layers.dense(input_frame, config.gen_hidden_size, use_bias=True)
        outputs = tf.nn.relu(outputs)
        outputs = tf.layers.dense(outputs, config.phn_size, use_bias=True)
    # Reshape to [batch_size, seq_len, phn_size]
    log_prob = tf.reshape(outputs, shape=[-1, int(frame_shape[1]), config.phn_size])
    return log_prob


def frame2phn(input_frame, config, temp, input_len=None, reuse=False):
    with tf.variable_scope('frame2phn') as scope:
        if reuse: scope.reuse_variables()
        # Phoneme Classifier
        log_prob = frame2phn_network(input_frame, config)
        # Softmax / Gumbel Softmax
        soft_prob = softmax(log_prob, temp)
        hard_prob = gumbel_sampling(log_prob, 0.9, hard=False)
        # Masking
        soft_prob = masking(soft_prob, input_len, config.phn_size)
        hard_prob = masking(hard_prob, input_len, config.phn_size)
    return soft_prob, hard_prob, log_prob


def generate_real_sample(input_idx, input_len, phn_size):
    real_sample = tf.one_hot(input_idx, phn_size, dtype=tf.float32)
    real_sample = masking(real_sample, input_len, phn_size)
    return real_sample


def generate_inter_sample(real, fake):
    alpha = tf.random_uniform(shape=[tf.shape(real)[0], 1, 1], maxval=1.)
    inter_sample = real + alpha * (fake - real)
    return inter_sample


def sample_noise(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def softmax(logits, temperature, sample=True):
    y = logits + sample_noise(tf.shape(logits)) if sample else logits
    return tf.nn.softmax(y / temperature)


def gumbel_sampling(logits, temperature=0.9, hard=False):
    y = softmax(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def intra_segment_loss(frame_prob, bound_weight):
    unstack_frame_prob = tf.unstack(frame_prob, axis=1)
    segment_loss = tf.square(
        tf.stack(unstack_frame_prob[:-1], axis=1) - tf.stack(unstack_frame_prob[1:], axis=1)) * tf.expand_dims(
        bound_weight, -1)
    return tf.reduce_sum(segment_loss) / tf.cast(tf.reduce_sum(bound_weight), tf.float32)


def segment_loss(start_prob, end_prob, repeat_num=None):
    loss = tf.square(start_prob - end_prob)
    if repeat_num is None:
        return tf.reduce_mean(loss)
    else:
        return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(repeat_num), tf.float32)


def sequence_loss(decoder_outputs, target_inputs, target_length):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_inputs, logits=decoder_outputs)
    mask = tf.sequence_mask(target_length, tf.shape(target_inputs)[1])
    paddings = tf.zeros_like(loss)
    loss = tf.where(mask, loss, paddings)
    mean_loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(target_length), tf.float32)
    return mean_loss
