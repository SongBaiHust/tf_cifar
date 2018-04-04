import tensorflow as tf
from tensorflow.contrib import slim
import os
import time
import math
from tensorflow.contrib.layers.python.layers import utils
#from tensorflow.contrib.slim.nets import resnet_v1
import cifarnet
from DatasetCifar import *
################################################################################
#######################
# Training&Test Flags #
#######################
tf.app.flags.DEFINE_integer(
    'log_freq', 50, 'log frequence')
tf.app.flags.DEFINE_integer(
    'trace_freq', 1000, 'trace frequence')
tf.app.flags.DEFINE_integer(
    'sum_freq', 10, 'summary frequence')
tf.app.flags.DEFINE_integer(
    'save_freq', 1000, 'summary frequence')
tf.app.flags.DEFINE_string(
    'logdir', './train', 'Saving and loging dir')
tf.app.flags.DEFINE_string(
    'mode', 'train', 'process mode: train or test')
tf.app.flags.DEFINE_integer(
    'top_k', 1, 'set top k accuracy in test mode')
#######################
# Learning Rate Flags #
#######################
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
#tf.app.flags.DEFINE_bool(
#    'sync_replicas', False,
#    'Whether or not to synchronize the replicas during training.')
#tf.app.flags.DEFINE_integer(
#    'replicas_to_aggregate', 1,
#    'The Number of gradients to collect before updating params.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
#######################
#    Dataset Flags    #
#######################
tf.app.flags.DEFINE_integer(
    'class_number', 10, 'class number of cifar dataset, 10 or 100')
#tf.app.flags.DEFINE_string(
#    'dataset_name', 'imagenet', 'The name of the dataset to load.')
#tf.app.flags.DEFINE_string(
#    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
#tf.app.flags.DEFINE_integer(
#    'labels_offset', 0,
#    'An offset for the labels in the dataset. This flag is primarily used to '
#    'evaluate the VGG and ResNet architectures which do not use a background '
#    'class for the ImageNet dataset.')
#tf.app.flags.DEFINE_string(
#    'model_name', 'inception_v3', 'The name of the architecture to train.')
#tf.app.flags.DEFINE_string(
#    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
#    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'num_epoch', 1, 'training number of epoch')
tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
FLAGS = tf.app.flags.FLAGS
################################################################################

def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.
    Returns:
        A `Tensor` representing the learning rate.
    Raises:
        ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
    #if FLAGS.sync_replicas:
    #decay_steps /= FLAGS.replicas_to_aggregate
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)
def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
        learning_rate: A scalar or `Tensor` learning rate.
    Returns:
        An instance of an optimizer.
    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer
'''
def create_model(images, labels):
    with tf.name_scope('Network'):
        with slim.arg_scope(cifarnet.cifarnet_arg_scope()):
            logits, end_points = cifarnet.cifarnet(images)
    with tf.name_scope('X_entropy_loss'):
        slim.losses.softmax_cross_entropy(
            labels, logits, label_smoothing=FLAGS.label_smoothing, weights=1.0)
    with tf.name_scope('total_loss'):
        total_loss = slim.losses.get_total_loss()
    with tf.name_scope('global_step'):
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)
    with tf.name_scope('train'):
        learning_rate = _configure_learning_rate(50000, global_step) #to be modified with val
        optimizer = _configure_optimizer(learning_rate)
        var_to_train = [var for var in tf.trainable_variables()]
        gradients = optimizer.compute_gradients(total_loss, var_list=var_to_train)
        train_op = optimizer.apply_gradients(gradients)
    return train_op, total_loss, global_step, incr_global_step
'''

def preprocessing_image(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return (image, label)

def setup_train_input(sess, capacity_factor=10):
    dataset = DatasetCifar(FLAGS.dataset_dir, one_hot=True,  class_number=FLAGS.class_number)
    train_images, train_labels = dataset.read_train()
    train_images = np.array(train_images).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)
    image_placeholder = tf.placeholder(train_images.dtype, train_images.shape)
    label_placeholder = tf.placeholder(train_labels.dtype, train_labels.shape)
    train_data = tf.data.Dataset.from_tensor_slices((image_placeholder, label_placeholder))
    train_data = train_data.map(preprocessing_image)
    train_data = train_data.shuffle(buffer_size=FLAGS.batch_size*capacity_factor)
    train_data = train_data.repeat(FLAGS.num_epoch)
    train_data = train_data.batch(FLAGS.batch_size)
    iterator = train_data.make_initializable_iterator()
    sess.run(iterator.initializer, feed_dict={image_placeholder: train_images, label_placeholder: train_labels})
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

def should_log(freq, step, max_step):
    return freq > 0 and ((step + 1) % freq == 0 or step == max_step -1)

def _train(sess):
    with tf.name_scope('data_input'):
        image_batch, label_batch = setup_train_input(sess)

    with tf.name_scope('Network'):
        with slim.arg_scope(cifarnet.cifarnet_arg_scope()):
            logits, end_points = cifarnet.cifarnet(image_batch)
    with tf.name_scope('X_entropy_loss'):
        x_entropy_loss = slim.losses.softmax_cross_entropy(
            label_batch, logits, label_smoothing=FLAGS.label_smoothing, weights=1.0)
    with tf.name_scope('total_loss'):
        total_loss = slim.losses.get_total_loss()
    with tf.name_scope('global_step'):
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)
    with tf.name_scope('train'):
        learning_rate = _configure_learning_rate(50000, global_step) #to be modified with val
        optimizer = _configure_optimizer(learning_rate)
        var_to_train = [var for var in tf.trainable_variables()]
        gradients = optimizer.compute_gradients(total_loss, var_list=var_to_train)
        train_op = optimizer.apply_gradients(gradients)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver(max_to_keep=5)

    with tf.name_scope('Summary'):
        # Add summaries for end_points.
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point,
                                          tf.nn.zero_fraction(x))
        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            tf.summary.scalar('losses/%s' % loss.op.name, loss)
        tf.summary.scalar('total_loss', total_loss)
        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        tf.summary.scalar('learning_rate', learning_rate)
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)
        summary_op = tf.summary.merge_all()
    step = 0
    max_step = math.ceil(50000*FLAGS.num_epoch*10 / FLAGS.batch_size)
    start_time = time.time()
    print('Start training...')
    print('Batch size: %d, number of epoch: %d' % (FLAGS.batch_size, FLAGS.num_epoch))
    while True:
        try:
            options = None
            run_metadata = None
            if should_log(FLAGS.trace_freq, step, max_step):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            fetches = {
                'train': train_op,
                'global_step': global_step,
                'incr_global_step': incr_global_step
            }
            if should_log(FLAGS.log_freq, step, max_step):
                fetches['x_entropy_loss'] = x_entropy_loss
                fetches['total_loss'] = total_loss
            if should_log(FLAGS.sum_freq, step, max_step):
                fetches['summary'] = summary_op
            results = sess.run(
                fetches,
                options=options,
                run_metadata=run_metadata)
            current_time = time.time()
            process_time = current_time - start_time
            remain_time = (max_step - step + 1) * process_time / (step + 1)
            if should_log(FLAGS.log_freq, step, max_step):
                print('-------------------------------------------------------------------------------')
                print('Global step: %d, X_loss: %.4f, total loss: %.4f, process time: %d mins, remain time: %d mins' %
                      (results['global_step'], results['x_entropy_loss'], results['total_loss'], process_time/60, remain_time/60))
            if should_log(FLAGS.sum_freq, step, max_step):
                summary_writer.add_summary(results['summary'])
            if should_log(FLAGS.trace_freq, step, max_step):
                print('Recording trace...')
                summary_writer.add_run_metadata(run_metadata, 'step_%d' % results['global_step'])
            if should_log(FLAGS.save_freq, step, max_step):
                print('Saving model...')
                saver.save(sess, os.path.join(FLAGS.logdir, 'model'),
                    global_step=results['global_step'])
            step = step + 1
        except tf.errors.OutOfRangeError:
            print('----------------------------------------------------------------------------------')
            print('Done training!')
            current_time = time.time()
            process_time = current_time - start_time
            print('Total training time is %d hours and %d minutes' %
                 (math.floor(process_time/3600),
                 (process_time - math.floor(process_time/3600)*3600)/60))
            break

def _test(sess):
    image_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3])
    label_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.class_number])
    with slim.arg_scope(cifarnet.cifarnet_arg_scope()):
        logits, endpoints = cifarnet.cifarnet(image_placeholder)
    top_k_op = tf.nn.in_top_k(logits, label_placeholder, FLAGS.top_k)
    variable_to_restore = slim.get_model_variables()
    restorer = tf.train.Saver(variable_to_restore)
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    restorer.restore(sess, checkpoint)
    print('---Successfully restored from %s---' % FLAGS.lordir)
    dataset = DatasetCifar(FLAGS.dataset_dir, one_hot=True,  class_number=FLAGS.class_number)
    batch_count = 0
    true_count = 0
    while True:
        try:
            test_images, test_labels = dataset.next_test_batch()
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            predictions = sess.run(top_k_op, feed_dict={
                image_placeholder: test_iamges,
                label_placeholder: test_labels})
            true_count += np.sum(predictions)
            batch_count += 1
        except OutOfRangeError:
            precision = true_count / (batch_count * FLAGS.batch_size)
            print('-----------------------------')
            print('| Precision is %f |' % precision)
            print('-----------------------------')
            break


def main(_):
    session_config=tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
    with tf.Session(config=session_config) as sess:
        if FLAGS.mode == 'train':
            _train(sess)
        elif FLAGS.mode == 'test':
            _test(sess)
        else:
            raise ValueError('Wrong mode [%s], mode must be train or test', FLAGS.mode)

if __name__=='__main__':
    tf.app.run()
