import json
import time
import numpy as np

# 标签：1是留下，0 是去掉， 0是padding
# 词向量： 0是padding， 1-19998， 19999是OOV 20001是</s>, 20002是</e>
class DataLoader(object):
    def __init__(self, args):
        self.train_data_path = args.train_data_dir
        self.test_data_path = args.test_data_dir
        self.evaluate_data_path = args.evaluate_data_dir
        start_time = time.time()

        self.train_set = self.load_data(self.train_data_path)
        self.test_set = self.load_data(self.test_data_path)
        self.evaluate_set = self.load_data(self.evaluate_data_path)  # evaluate虽然有tags，但是是不用的，只是为了batch_iter，和训练中可以跑acc
        print('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

    def load_data(self, path):
        data = json.load(open(path, 'r'))
        texts, tags = [], []
        for i in range(len(data)):
            texts.append(data[i][0])
            tags.append(data[i][1])
        return (texts, tags)

    def batch_iter(data, time_steps_encoder, time_steps_decoder, batch_size, shuffle):
        texts, tags = data
        data_size = len(texts)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            texts = np.array(texts)[shuffle_indices]
            tags = np.array(tags)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_data = {'encoder_input': [], 'decoder_input': [], 'decoder_target':[]}
            for text, tag in zip(texts[start_index:end_index], tags[start_index:end_index]):
                length = len(text)
                summary = []
                for i in range(length):
                    if tag[i] == 1:
                        summary.append(text[i])
                encoder_input = text + [0] * (time_steps_encoder - length)
                decoder_input = [20001]+summary+[0]*(time_steps_decoder-len(summary)-1)
                decoder_target = summary+[20002]+[0]*(time_steps_decoder-len(summary)-1)
                # encoder_input = text
                # decoder_input = summary
                # decoder_target = summary + [20002]
                batch_data['encoder_input'].append(encoder_input)
                batch_data['decoder_input'].append(decoder_input)
                batch_data['decoder_target'].append(decoder_target)
            yield batch_data
