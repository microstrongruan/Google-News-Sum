from utils.Model import Model
from utils.DataLoder import DataLoader
import tensorflow as tf
import argparse, os, time

def build_parse():
    parser = argparse.ArgumentParser(description='seq2seq')
    # train para
    parser.add_argument('--train_data_dir', default='resource/data/train_data.json')
    parser.add_argument('--test_data_dir', default='resource/data/test_data.json')
    parser.add_argument('--evaluate_data_dir', default='resource/data/evaluate_data.json')
    parser.add_argument('--initial_dir', default='resource/data/initial_word.json')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--device', type = str, default='cpu')
    parser.add_argument('--display_step', default=100)
    # model para
    parser.add_argument('--cell_type', default='lstm')
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--max_grad_norm', default=15)
    parser.add_argument('--batch_size', type = int, default=20)
    parser.add_argument('--dropout_keep', default=0.6)
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--source_vocab', default=20003)
    parser.add_argument('--embedding_size', default=200)
    parser.add_argument('--time_steps', default=200)
    parser.add_argument('--num_hidden', default=200)
    parser.add_argument('--num_class', default=2)
    parser.add_argument('--num_layers', default=2)
    parser.add_argument('--time_steps_encoder', default=212)
    parser.add_argument('--time_steps_decoder', default=212)
    parser.add_argument('--decay_rate', default=0.01)
    parser.add_argument('--decay_steps', default=100)
    parser.add_argument('--max_iter', default=300)

    return parser.parse_args()


def check(args):
    if args.mode == 'evaluate':
        assert args.save_dir != ''
    if args.save_dir == '':
        args.save_dir = 'save/'+'_'.join(time.asctime(time.localtime(time.time())).split())
        os.mkdir(args.save_dir)


def write_log(args, s):
    print(s)
    with open(args.save_dir+'/log.txt', 'a') as f:
        f.write(s)
        f.write('\n')


def main(args):
    import numpy as np
    data = DataLoader(args)
    model = Model()
    print('finished load')
    with tf.Session() as sess:
        for epoch_step in range(args.epoch):
            for batch_data in DataLoader.batch_iter(data.train_set,args.time_steps_encoder, args.time_steps_decoder, 20, True):
                sess.run([model.train_op, model.mean_loss], feed_dict={
        model.encoder_input: np.random.randint(0, 100, [10, 20]),
        model.decoder_input: np.random.randint(0, 100, [10, 20]),
        model.encoder_len:np.random.randint(1,20, [10]),
        model.decoder_len:np.random.randint(1,20, [10]),
        model.decoder_output:np.random.randint(0, 100, [10, 20])})

if __name__ == '__main__':
    args = build_parse()
    check(args)
    write_log(args, str(args))
    main(args)