import tensorflow as tf
from .rnn_cell import LstmCell

class UniOneLayerLstmEncoder(object):
    def __init__(self, num_hidden, emb_size):
        self.num_hidden = num_hidden
        self.emb_size = emb_size
        self.cell = LstmCell(emb_size, num_hidden, 'encoder_lstm_cell')

    def __call__(self, encoder_input, encoder_time_step):
        batch_size = tf.shape(encoder_input)[0]
        max_time = tf.shape(encoder_input)[1]
        emb_size = tf.shape(encoder_input)[2]
        num_hidden = self.num_hidden

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, num_hidden], dtype=tf.float32),
              tf.zeros([batch_size, num_hidden], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(encoder_input, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.cell(x_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, encoder_time_step)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))
        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state


class BiOneLayerLstmEncoder(object):
    def __init__(self, num_hidden, emb_size):
        self.num_hidden = num_hidden
        self.emb_size = emb_size
        self.cell_fw = LstmCell(emb_size, num_hidden, 'encoder_fw_cell')
        self.cell_bw = LstmCell(emb_size, num_hidden, 'encoder_bw_cell')

    def __call__(self, encoder_input, encoder_time_step):

        def fw(cell, emb_size, encoder_input, encoder_time_step):
            batch_size = tf.shape(encoder_input)[0]
            max_time = tf.shape(encoder_input)[1]
            num_hidden = self.num_hidden

            time = tf.constant(0, dtype=tf.int32)
            h0 = (tf.zeros([batch_size, num_hidden], dtype=tf.float32),
                  tf.zeros([batch_size, num_hidden], dtype=tf.float32))
            f0 = tf.zeros([batch_size], dtype=tf.bool)
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
            inputs_ta = inputs_ta.unstack(tf.transpose(encoder_input, [1, 0, 2]))
            emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

            def loop_fn(t, x_t, s_t, emit_ta, finished):
                o_t, s_nt = cell(x_t, s_t, finished)
                emit_ta = emit_ta.write(t, o_t)
                finished = tf.greater_equal(t + 1, encoder_time_step)
                x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                               lambda: inputs_ta.read(t + 1))
                x_nt.set_shape([None, emb_size])
                # x_nt.set_shape([10, 200])
                return t + 1, x_nt, s_nt, emit_ta, finished

            _, _, state, emit_ta, _ = tf.while_loop(
                cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
                body=loop_fn,
                loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))
            outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
            return outputs, state
        outputs_fw, state_fw = fw(self.cell_bw, self.emb_size, encoder_input, encoder_time_step)
        outputs_bw, state_bw = fw(self.cell_bw, self.emb_size, tf.reverse(encoder_input, [2]), [15]*10)
        return outputs_bw, state_bw
# ########################################################################
#         batch_size = tf.shape(encoder_input)[0]
#         batch_size = tf.shape(encoder_input)[0]
#         max_time = tf.shape(encoder_input)[1]
#         num_hidden = self.num_hidden
#
#         inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
#         inputs_ta = inputs_ta.unstack(tf.transpose(encoder_input, [1,0,2]))
#         h0 = (tf.zeros([batch_size, num_hidden], dtype=tf.float32),
#               tf.zeros([batch_size, num_hidden], dtype=tf.float32))
#
#         #cell_fw
#         time_fw = tf.constant(0, dtype=tf.int32)
#         f0_fw = tf.zeros([batch_size], dtype=tf.bool)
#         emit_ta_fw = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
#
#         def loop_fn_fw(t, x_t, s_t, emit_ta_fw, finished):
#             o_t, s_nt = self.cell_fw(x_t, s_t, finished)
#             emit_ta_fw = emit_ta_fw.write(t, o_t)
#             finished = tf.greater_equal(t+1, encoder_time_step)
#             x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
#                                      lambda: inputs_ta.read(t+1))
#             return t+1, x_nt, s_nt, emit_ta_fw, finished
#
#         _, _, state_fw, emit_ta_fw, _ = tf.while_loop(
#             cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
#             body=loop_fn_fw,
#             loop_vars=(time_fw, inputs_ta.read(0), h0, emit_ta_fw, f0_fw))
#         outputs_fw = tf.transpose(emit_ta_fw.stack(), [1,0,2])
#
#         # return outputs_fw, state_fw
#         #cell_bw
#         tf.reverse()
#         time_bw = max_time
#         index_bw = tf.constant(0, dtype=tf.int32)
#         f0_bw = tf.ones([batch_size], dtype=tf.bool)
#         emit_ta_bw = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
#
#         def loop_fn_bw(t, i, x_t, s_t, emit_ta_bw, finished):
#             o_t, s_nt = self.cell_bw(x_t, s_t, finished)
#             emit_ta_bw = emit_ta_bw.write(i, o_t)
#             finished = tf.greater(t-1, encoder_time_step)
#             x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
#                                      lambda: inputs_ta.read(t-1))
#             return t-1, i+1, x_nt, s_nt, emit_ta_bw, finished
#
#         _, _, _, state_bw, emit_ta_bw, _ = tf.while_loop(
#             cond=lambda t, _1, _2, _3, _4, _5:  tf.reduce_all(tf.less(t, 0)),
#             body=loop_fn_bw,
#             loop_vars=(time_bw, index_bw, inputs_ta.read(max_time), h0, emit_ta_bw, f0_bw)
#         )
#
#         outputs_bw = tf.transpose(emit_ta_bw.stack(), [1,0,2])
#
#         return tf.concat([outputs_fw, outputs_bw], 2), tf.concat([state_fw, state_bw], 1)
