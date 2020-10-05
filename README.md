# seq2seq-LSTM
源代码地址：
https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb
简易版的seq2seq网络
编码和解码均仅用到了LSTM,具体实现的功能为将输入单词,按字典序重新排序后输出
decoder分为train_decoder和predict_decoder
其中train_decoder用到了target(除最后结尾<eos>)的每个元素作为输入得到下一时间步的预测结果,保证训练模型预测结果的准确性
    predict_decoder仅用到了target的<go>起始符以及encoder最后输出的隐藏状态
