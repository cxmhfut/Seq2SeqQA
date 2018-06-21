import tensorflow as tf
import os
import datetime
import math
from tqdm import tqdm

from args import args
from data_util import TextData
from seq2seq_model2 import seq2seq


class ChatBot:
    def __init__(self):
        self.args = args()
        self.text_data = None
        self.seq2seq_model = None
        self.writer = None
        self.saver = None
        self.sess = None
        self.train_op = None
        self.global_step = 0
        self.SENTENCE_PREFIX = ['Q:', 'A:']

        self.main()

    def main(self):
        self.text_data = TextData(self.args)

        with tf.Graph().as_default():
            # build seq2seq model
            self.seq2seq_model = seq2seq(self.args, self.text_data)

            # Saver/summaries
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "save/model"))
            print(out_dir)
            self.writer = tf.summary.FileWriter(out_dir)
            self.saver = tf.train.Saver()

            session_conf = tf.ConfigProto(
                allow_soft_placement=self.args.allow_soft_placement,
                log_device_placement=self.args.log_device_placement
            )
            self.sess = tf.Session(config=session_conf)

            if self.args.test is not None:
                pass

            if self.args.test == 'interactive':
                pass
            else:
                self.train()

    def train(self):
        mergedSummaries = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(
            self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            epsilon=self.args.epsilon
        )
        grads_and_vars = optimizer.compute_gradients(self.seq2seq_model.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        self.sess.run(tf.global_variables_initializer())

        try:  # If the user exit while training, we still try to save the model
            for i in range(self.args.epoch_nums):
                # Generate batches
                tic = datetime.datetime.now()
                batches = self.text_data.get_next_batches()
                for next_batch in tqdm(batches, desc="Training"):
                    # step, summaries, loss = self.seq2seq_model.step(next_batch)
                    feed_dict = self.seq2seq_model.step(next_batch)

                    _, summaries, loss = self.sess.run(
                        (self.train_op, mergedSummaries, self.seq2seq_model.loss),
                        feed_dict)

                    self.global_step += 1
                    self.writer.add_summary(summaries, self.global_step)

                    # Output training status
                    if self.global_step % 100 == 0:
                        perplexity = math.exp(float(loss) if loss < 300 else float("inf"))
                        tqdm.write(
                            "----- step %d -- Loss %.2f -- Perplexity %.2f" % (self.global_step, loss, perplexity))
                    if self.global_step % self.args.checkpoint_every == 0:
                        self.save_session(self.sess, self.global_step)
                toc = datetime.datetime.now()
                print("Epoch finished in {}".format(toc - tic))
        except(KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')
            self.save_session(self.sess, self.global_step)

    def save_session(self,sess,step):
        tqdm.write('Checkpoint reached: saving model')
        model_name = os.path.join('save/model','model.ckpt')
        self.saver.save(sess,model_name,global_step=step)
        tqdm.write('Model saved.')

if __name__ == '__main__':
    chatbot = ChatBot()