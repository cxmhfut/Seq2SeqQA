# coding = utf-8
import os
import pandas as pd
from itertools import chain
import numpy as np
import pickle as pkl
import tqdm
import nltk
import random


class Batch:
    """
    Struct containing batches info
    """

    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    def __init__(self, args):
        self.args = args

        self.word2id = None
        self.id2word = None
        self.line_id = None

        self.train_samples = []

        self.padToken = '<pad>'  # Padding
        self.goToken = '<go>'  # Start of sequence
        self.eosToken = '<eos>'  # End of sequence
        self.unKnowToken = '<unk>'  # Word dropped from vocabulary
        self.numToken = 4  # Num of above tokens

    def load_data(self):
        if not os.path.exists(self.args.word_id_path) or not os.path.exists(self.args.train_samples_path):
            # 读取 movie_line.txt 和 movie_conversations.txt 两个文件
            print("开始读取数据")

            self.lines = pd.read_csv(self.args.line_path, sep="\+\+\+\$\+\+\+", usecols=[0, 4],
                                     names=["line_id", "utterance"], dtype={"utterance": str}, engine="python")
            self.conversations = pd.read_csv(self.args.conv_path, usecols=[3], names=["line_ids"],
                                             sep="\+\+\+\$\+\+\+", dtype={"line_ids": str}, engine="python")
            self.lines.utterance = self.lines.utterance.apply(lambda conv: self.word_tokenizer(conv))
            self.conversations.line_ids = self.conversations.line_ids.apply(lambda li: eval(li))
            # print("数据读取完毕")

    def build_word_dict(self):
        if not os.path.exists(self.args.word_id_path):
            # 得到 word2id 和 id2word 两个词典
            print("开始构建词典")
            words = self.lines.utterance.values
            words = list(chain(*words))
            # 将全部 words 转为小写
            words = list(map(str.lower, words))
            print("转化小写完毕")

            words_count = pd.Series(words).value_counts()
            # 筛选出出现次数大于1的词作为 vocabulary
            words_size = np.where(words_count.values > self.args.vocab_filter)[0].size
            words_index = words_count.index[0:words_size]

            self.word2id = pd.Series(range(self.numToken, self.numToken + words_size), index=words_index)
            self.id2word = pd.Series(words_index, index=range(self.numToken, self.numToken + words_size))
            self.word2id[self.padToken] = 0
            self.word2id[self.goToken] = 1
            self.word2id[self.eosToken] = 2
            self.word2id[self.unKnowToken] = 3
            print("词典构建完毕")
            with open(os.path.join(self.args.word_id_path), 'wb') as handle:
                data = {
                    'word2id': self.word2id,
                    'id2word': self.id2word
                }
                pkl.dump(data, handle, -1)
        else:
            print("从{}载入词典".format(self.args.word_id_path))
            with open(self.args.word_id_path, 'rb') as handle:
                data = pkl.load(handle)
                self.word2id = data['word2id']
                self.id2word = data['id2word']

    def replace_word_with_id(self, conv):
        conv = list(map(str.lower, conv))
        conv = list(map(self.get_word_id, conv))
        return conv

    def get_word_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id[self.unKnowToken]

    def get_id_word(self, id):
        if id in self.id2word:
            return self.id2word[id]
        else:
            return self.unKnowToken

    def generate_conversations(self):
        if not os.path.exists(self.args.train_samples_path):
            print("开始生成训练样本")
            self.line_id = pd.Series(self.lines.utterance.values, index=self.lines.line_id.values)
            for line_id in tqdm(self.conversations.line_ids.values, ncols=10):
                for i in range(len(line_id) - 1):
                    first_conv = self.line_id[line_id[i]]
                    second_conv = self.line_id[line_id[i + 2]]

                    first_conv = self.replace_word_with_id(first_conv)
                    second_conv = self.replace_word_with_id(second_conv)
                    valid = self.filter_conversations(first_conv, second_conv)

                    if valid:
                        temp = [first_conv, second_conv]
                        self.train_samples.append(temp)

            print("生成训练样本结束")
            with open(self.args.train_samples_path, 'wb') as handle:
                data = {
                    'train_samples': self.train_samples
                }
                pkl.dump(data, handle, -1)
        else:
            with open(self.args.train_samples_path, 'rb') as handle:
                data = pkl.load(handle)
                self.train_samples = data['train_samples']
            print("从{}导入训练样本".format(self.args.train_samples_path))

    def word_tokenizer(self, sentence):
        # 英文分词
        words = nltk.word_tokenize(sentence)
        return words

    def filter_conversations(self, first_conv, second_conv):
        # 筛选样本 去掉长度大于maxLength的样本和含有unk的样本
        valid = True
        valid &= len(first_conv) <= self.args.maxLength
        valid &= len(second_conv) <= self.args.maxLength
        valid &= second_conv.count(self.word2id[self.unKnowToken]) == 0

        return valid

    def get_next_batches(self):
        """
        Prepare the batches for the current epoch
        :return:
            List<Batch>:Get a list of the batches for the next epoch
        """
        self.shuffle()
        batches = []

        def gen_next_samples():
            """
            Generate over the mini-batch training samples
            :return:
            """
            for i in range(0, len(self.train_samples, self.args.batch_size)):
                yield self.train_samples[i:min(i + self.args.batch_size, len(self.train_samples))]

        # TODO: Should replace that by generator (better by tf.queue)

        for samples in gen_next_samples():
            batch = self.create_batch(samples)
            batches.append(batch)

        return batches

    def shuffle(self):
        """
        Shuffle the training samples
        :return:
        """
        print("Shuffling the dataset...")
        random.shuffle(self.train_samples)

    def create_batch(self, samples):
        batch = Batch()
        batch_size = len(samples)

        # Create the batch tensor
        for i in range(batch_size):
            # Unpack the sample
            sample = samples[i]
            batch.encoderSeqs.append(list(reversed(sample[0])))
            batch.decoderSeqs.append([self.word2id[self.goToken]] + sample[1] + [self.word2id[self.eosToken]])
            # Same as decoder, but shifted to the left (ignore the <go>)
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:])

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.encoderSeqs[i] = [self.word2id[self.padToken]] * (
                    self.args.maxLengthEnco - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]
            batch.weights.append(
                [1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2id[self.padToken]] * (
                    self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.word2id[self.padToken]] * (
                    self.args.maxLengthDeco - len(batch.decoderSeqs[i]))

        # Simple hack to reshape the batch
        encoderSeqsT = []
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batch_size):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []

        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batch_size):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)

        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        return batch

    def sentence2enco(self, sentence):
        """
        Encode a sequence and return a batch as an input for the model
        :param sentence:
        :return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """
        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        #Second step: Covert the token in word ids
        wordIds = []
        for token in tokens:
            if token in self.word2id:
                wordIds.append(self.word2id[token])
            else:
                wordIds.append(self.word2id[self.unKnowToken])

        #Third step: creating the batch (add padding, reverse)
        batch = self.create_batch([[wordIds,[]]])

        return batch

    def deco2sentence(self,decoderOutputs):
        """
        Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>)
        :param decoderOutputs:
        :return:
        """

        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out)) #Adding each prediction word ids

        return sequence