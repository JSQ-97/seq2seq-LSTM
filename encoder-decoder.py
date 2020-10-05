import numpy as np
import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001


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


# input_layer
def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

# 在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
# 在Embedding中，我们使用tf.contrib.layers.embed_sequence，它会对每个batch执行embedding操作。


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size, encoding_embedding_size):
    '''
    构造Encoder层
    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    '''
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    # 上式直接将输入转为embedding

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    encoder_output, encoder_stage = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length,
                                                      dtype=tf.float32)
    return encoder_output, encoder_stage


# Decoder
def process_decoder_input(data, label2id, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])  # 切片删除最后一个元素
    decoder_input = tf.concat([tf.fill([batch_size, 1], label2id["<GO>"]), ending], 1)
    # 生成batch_size的一列"<GO>"标号,用tf.concat拼接到输入的第一列
    # concat将[1, 2] 内的1 和 2 按轴系1拼接起来
    return decoder_input


def decoding_layer(target_id2label, decoding_embedding_size, num_layers, rnn_size, target_sequence_length,
                   max_target_sequence_length, encoder_state, decoder_input):
    '''
    构造Decoder层

    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''
    target_vocab_size = len(target_id2label)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))  # 随机初始化生成向量
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)  # nn.embedding__lookup

    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    # decoder端LSTM网络
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # contrib.seq2seq.TrainingHelper：提供一个next函数,能够提供下一个时间步的输出?
        # 接受[batch_size, sequence_length, embedding_size]的输入, 每个句子的真实输入长度target_sequence_length
        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,             # decoder端的网络结构
                                                           training_helper,  # Helper分两种一种为training_Helper用于训练
                                                           #一种为inference_Helper用于预测
                                                           encoder_state,    # initial_state, 一般为encoder最后输出的隐藏状态
                                                           output_layer)     # 输出层：一个全连接层
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,   # 类似dynamic_rnn函数
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_label2id['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')   # predict的时候decoder只输入一个batch的"<GO>"的标号
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     # 专门用于predict的seq2seq Helper, 与train_Helper不同
                                                                     # 输入整体的embedding矩阵 方便预测时embedding_lookup
                                                                     start_tokens,  # decoder的输入"<GO>"的标号
                                                                     target_label2id['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


# Seq-Seq 将encoder—decoder连接起来
def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
    _, encoder_state = get_encoder_layer(input_data,
                                  rnn_size,
                                  num_layers,
                                  source_sequence_length,
                                  source_vocab_size,
                                  encoding_embedding_size)
    decoder_input = process_decoder_input(targets, target_label2id, batch_size)
    training_decoder_output, predicting_decoder_output = decoding_layer(target_label2id,
                                                                       decoding_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state,
                                                                       decoder_input)
    return training_decoder_output[0], predicting_decoder_output[0]


# 构造graph
train_graph = tf.Graph()

with train_graph.as_default():
    # 获得模型输入
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_label2id),
                                                                       len(target_label2id),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)

    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    # tf.identity相当于复制
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    # tf.sequence_mask将不同长度输入都映射到max_sequence长度

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


def pad_sequence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence-len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sequence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sequence_batch(targets_batch, target_pad_int))
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))
        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


# 将数据集分割为train和validation
train_source = source_int[batch_size:]
train_target = target_int[batch_size:]
# 留出一个batch进行验证
valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                           source_label2id['<PAD>'],
                           target_label2id['<PAD>']))

display_step = 50
checkpoint = "train_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(1, epochs+1):
        for batch_i, (targets_batch, sources_batch, targets_length, source_length) in enumerate(get_batches(train_target, train_source, batch_size,
                                                                                                             source_label2id["<PAD>"], target_label2id["<PAD>"])):
            _, loss = sess.run([train_op, cost], {input_data: sources_batch,
                                                  targets: targets_batch,
                                                  lr: learning_rate,
                                                  target_sequence_length: targets_length,
                                                  source_sequence_length: source_length})  # tf.placeholder的都要feed_dict
            if batch_i % display_step == 0:
                # 计算validation loss
                validation_loss = sess.run(
                    [cost],
                    {input_data: valid_sources_batch,
                     targets: valid_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: valid_targets_lengths,
                     source_sequence_length: valid_sources_lengths})

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))
    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')
