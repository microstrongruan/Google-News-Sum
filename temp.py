from utils.encoder_decoder import UniOneLayerLstmEncoder, BiOneLayerLstmEncoder
import tensorflow as tf

encoder = BiOneLayerLstmEncoder(num_hidden=100, emb_size=200)
x = tf.placeholder(dtype=tf.float32, shape=[None, 15, 200])
x_time_step = tf.placeholder(dtype=tf.int32, shape=[None,])
outputs, state = encoder(x, x_time_step)
import numpy as np
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    time_step = np.random.randint(0, 15, 10)
    out, s = sess.run([outputs, state], feed_dict={x: np.random.random([10,15,200]), x_time_step:time_step})
    print(out.shape, s[0].shape, s[1].shape)
    # print(time_step,out.shape, s[0].shape, s[1].shape)
    # print(out[:, 10:, :])
# import tensorflow as tf
# from tensorflow.contrib import seq2seq
# from tensorflow import contrib
# import numpy as np
# from tensorflow.python.layers.core import Dense
#
# encoder_input = tf.placeholder(dtype=tf.float32, shape = [10,20,100])
# decoder_input = tf.placeholder(dtype=tf.float32, shape= [10, 20, 100])
# encoder_state = np.random.random([10,100])
# # cell= tf.contrib.rnn.LSTMCell(100, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
# # train_helper = seq2seq.TrainingHelper(inputs=decoder_input,
# #     sequence_length=[20] * 10,
# #     time_major=False
# # )
# # decoder_cell = tf.contrib.rnn.MultiRNNCell([cell])
# #
# # train_decoder = seq2seq.BasicDecoder(
# #     cell=decoder_cell,
# #     helper=train_helper,
# #     initial_state=cell.zero_state(batch_size=10, dtype=tf.float32),
# #     output_layer=Dense(20))
# # logits, final_state, final_sequence_lengths = seq2seq.dynamic_decode(train_decoder)
#
# encoder_cell = tf.contrib.rnn.LSTMCell(100)
#
# encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
#     encoder_cell, encoder_input,
#     dtype=tf.float32)
#
# decoder_cell = tf.contrib.rnn.LSTMCell(100)
#
# decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
#     decoder_cell, decoder_input,
#
#     initial_state=encoder_state,
#
#     dtype=tf.float32,scope="plain_decoder",
# )
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(decoder_outputs, feed_dict={decoder_input:np.random.random[10, 20, 100],
#                                                encoder_input:np.random.random[10, 20, 100]}))