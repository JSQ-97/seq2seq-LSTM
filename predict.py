import numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense
# from seq2seq import *


def read_file(file1, file2):
    with open(file1, "r", encoding="utf-8") as f:
        source_data = f.read()
    with open(file2, "r", encoding="utf-8") as f:
        target_data = f.read()
    return source_data, target_data


file1, file2 = "./data/letters_source.txt", "./data/letters_target.txt"
source_data, target_data = read_file(file1=file1, file2=file2)


def extract_character_vocab(data):
    special_words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]  # 特殊标识符
    set_words = list(set([character for line in data.split("\n") for character in line]))
    id2label = {ID: label for ID, label in enumerate(special_words + set_words)}  # {4:"a"}
    label2id = {label: ID for ID, label in id2label.items()}  # {"a":4}
    return id2label, label2id


source_id2label, source_label2id = extract_character_vocab(source_data)
target_id2label, target_label2id = extract_character_vocab(target_data)

source_int = [[source_label2id.get(letter, source_label2id["<UNK>"]) for letter in line]
              for line in source_data.split("\n")]  # 将数据用标号形式表示
target_int = [[target_label2id.get(letter, target_label2id["<UNK>"]) for letter in line] + [target_label2id["<EOS>"]]
              for line in target_data.split("\n")]
# print(target_int[:10])


def source_to_seq(text):
    sequence_length = 7
    return [source_label2id.get(word, source_label2id['<UNK>']) for word in text] + [source_label2id['<PAD>']]*(sequence_length-len(text))

batch_size = 128
input_text = input()
text = source_to_seq(input_text)
checkpoint = "./train_model.ckpt"
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint+".meta")
    loader.restore(sess, checkpoint)
    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_text)] * batch_size,
                                      source_sequence_length: [len(input_text)] * batch_size})[0]


pad = source_label2id["<PAD>"]

print('原始输入:', input_text)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_id2label[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_id2label[i] for i in answer_logits if i != pad])))
