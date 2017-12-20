import tensorflow as tf


class LstmCell(object):
    def __init__(self, input_size, num_hidden, scope_name):
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.scope_name = scope_name
        self.param = {}

        with tf.variable_scope(self.scope_name):
            self.W = tf.get_variable(name='W', dtype=tf.float32, shape=[self.input_size + self.num_hidden, self.num_hidden * 4])
            self.b = tf.get_variable(name='b', dtype=tf.float32, shape=[self.num_hidden * 4])
        self.param.update({'W': self.W, 'b': self.b})

    def __call__(self, x, s, finished=None):
        h_prev, c_prev = s
        x = tf.concat([x, h_prev], 1)
        i, j, f, o = tf.split(tf.nn.xw_plus_b(x, self.W, self.b), 4, 1)

        # Final Memory cell
        c = tf.sigmoid(f + 1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j)
        h = tf.sigmoid(o) * tf.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            out = tf.where(finished, tf.zeros_like(h), h)
            state = (tf.where(finished, h_prev, h), tf.where(finished, c_prev, c))
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #         tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state


class RruCell(object):
    def __init__(self, input_size, num_hidden, scope_name):
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.scope_name = scope_name
        self.param = {}

        with tf.variable_scope(self.scope_name):
            self.W = tf.get_variable(dtype=tf.float32, shape=[self.input_size + self.num_hidden, self.num_hidden * 2])
            self.b = tf.get_variable(dtype=tf.float32, shape=[self.num_hidden * 2])
            self.W_j = tf.get_variable(dtype=tf.float32, shape=[self.input_size + self.num_hidden, self.num_hidden])
            self.b_j = tf.get_variable(dtype=tf.float32, shape=[self.num_hidden])
        self.param.update({'W': self.W, 'b': self.b, 'W_j': self.W_j, 'b_j': self.b_j})

    def __call__(self, x, h_prev, finished=None):
        x_1 = tf.concat([x, h_prev], 1)
        z, r = tf.split(tf.sigmoid(tf.nn.xw_plus_b(x_1, self.W, self.b)), 2, 1)
        x_2 = tf.concat([x, tf.multinomial(h_prev, r)], 1)
        j = tf.nn.xw_plus_b(x_2, self.W_j, self.b_j)
        h = tf.multiply(1 - z, h_prev) + tf.multiply(z, j)

        if finished is not None:
            h = tf.where(finished, tf.zeros_like(h), h)

        return h

