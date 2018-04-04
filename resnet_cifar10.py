import tensorflow as tf
from tensorflow.contrib import slim

class ResNetCifar():
    def __init__(self, block_factor=3, num_class=10):
        self.block_factor=block_factor
        self.global_pool=True
        self.num_class=num_class
    def resnet_arg_scope(self,
                        weight_decay=0.0001,
                        batch_norm_decay=0.997,
                        batch_norm_epsilon=1e-5,
                        batch_norm_scale=True,
                        activation_fn=tf.nn.relu,
                        use_batch_norm=True):
        batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
        }
        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params
            padding='SAME'):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arc_sc


    def res_block_same(self, input, num_channels=64, scope=None):
        net = slim.repeat(input, 2, slim.conv2d, num_channels, [3, 3], scope=scope)
        net = input + net
        return net

    def res_block_downsampel(self, input, stride=2, num_channels=64, scope=None):
        residual = slim.conv2d(input, num_channels, [3, 3], stride=stride, scope='%s/%s_res' % (scope, scope))
        net = slim.conv2d(input, num_channels, [3, 3], stride=stride, scope='%s/%s_1' % (scope, scope))
        net = slim.conv2d(net, num_channels, [3, 3], scope='%s/%s_2' % (scope, scope))
        net = net + residual
        return net
    def resnet_cifar(self, input):
        with slim.argscope(self.resnet_arg_scope()):
            end_points_collection = sc.original_name_scope + '_end_points'
            net = slim.conv2d(input, 16, [3, 3], scope='conv1')
            for group in range(3):
                for block in range(self.block_factor):
                    if group == 0:
                        net = self.res_block_same(input, num_channels=16, scope='group_%d/block_%d' % (group + 1, block + 1))
                    else:
                        if block == 0:
                            net = self.res_block_downsampel(input, num_channels=16*(2**block), scope='group_%d/block_%d' % (group + 1, block + 1))
                        else:
                            net = self.res_block_same(input, num_channels=16*(2**block), scope='group_%d/block_%d' % (group + 1, block + 1))
             end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if self.global_pool:
                net = slim.reduce_mean(net, [1, 2])
                end_points['global_pool'] = net
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            end_points['logits'] = net
        return net, end_points
