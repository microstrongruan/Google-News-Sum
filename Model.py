import tensorflow as tf
import pickle
from LstmUnit import LstmUnit
from OutputUnit import OutputUnit

class Model(object):
    def __init__(self):
        self.batch_size = 10
        self.num_hidden = 133
        self.emb_size = 199
        self.source_vocab = 2000
        self.target_vocab = 2000
        self.grad_clip = 5.0
        self.start_token = 1
        self.stop_token = 2
        self.encoder_max_time_step = 20
        self.decoder_max_time_step = 20

        self.units = {}
        self.params = {}

        self._create_placeholer()
        self._create_embedding()
        self._create_cell()

        en_outputs, en_state = self.encoder(self.encoder_input_embed, self.encoder_time_step)
        de_outputs, de_state = self.training_decoder(en_state, self.decoder_input_embed, self.decoder_time_step)
        # inference_outputs = self.inference_decoder(en_state)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=de_outputs, labels=self.decoder_output)
        mask = tf.sign(tf.to_float(self.decoder_output))
        losses = mask * losses
        self.mean_loss = tf.reduce_mean(losses)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def _create_placeholer(self):
        self.encoder_input = tf.placeholder(tf.int32, [None, self.encoder_max_time_step])
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_max_time_step])
        self.encoder_time_step = tf.placeholder(tf.int32, [None])
        self.decoder_time_step = tf.placeholder(tf.int32, [None])
        self.decoder_output = tf.placeholder(tf.int32, [None, self.decoder_max_time_step])

    def _create_cell(self):
        self.enc_lstm = LstmUnit(self.num_hidden, self.emb_size, 'encoder_lstm')
        self.dec_lstm = LstmUnit(self.num_hidden, self.emb_size, 'decoder_lstm')
        self.dec_out = OutputUnit(self.num_hidden, self.target_vocab, 'decoder_output')
        self.units.update({'encoder_cell': self.enc_lstm, 'decoder_cell': self.dec_lstm,
                           'decoder_output': self.dec_out})

    def _create_embedding(self):
        self.embedding = tf.get_variable('embedding', [self.source_vocab, self.emb_size])
        self.encoder_input_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
        self.decoder_input_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
        self.params.update({'embedding': self.embedding})

    def encoder(self, inputs, inputs_len):
        batch_size = tf.shape(self.encoder_input)[0]
        max_time = tf.shape(self.encoder_input)[1]
        num_hidden = self.num_hidden

        time = tf.constant(0, dtype=tf.int32)
        s0 = (tf.zeros([batch_size, num_hidden], dtype=tf.float32),
              tf.zeros([batch_size, num_hidden], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.enc_lstm(x_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), s0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state


    def training_decoder(self, initial_state, inputs, inputs_len):
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        s0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            o_t = self.dec_out(o_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, s0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state


    def inference_decoder(self, initial_state):
        batch_size = tf.shape(self.encoder_input)[0]

        time = tf.constant(0, dtype=tf.int32)
        s0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)

            o_t = self.dec_out(o_t, finished)
            emit_ta = emit_ta.write(t, o_t)

            next_token = tf.argmax(o_t, 1)
            x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.decoder_max_time_step))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, s0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        pred_tokens = tf.argmax(outputs, 2)
        return pred_tokens

model = Model()
import numpy as np
with tf.Session() as sess:
    for i in range(10):
        sess.run(tf.global_variables_initializer())
        _1, _2 = sess.run([model.train_op, model.mean_loss], feed_dict={
            model.encoder_input: np.random.randint(0, 100, [10, 20]),
            model.decoder_input: np.random.randint(0, 100, [10, 20]),
            model.encoder_time_step: [18]*10,
            model.decoder_time_step: [18]*9+[19],
            model.decoder_output: np.random.randint(0, 100, [10, 20])})
        print(_2)