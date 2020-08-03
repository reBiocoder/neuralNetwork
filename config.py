class Config:
    num_layers = 1  # LSTM层数
    data_path = './test_poetry.txt'  # 数据文本
    lr = 0.1  # 学习率
    use_gpu = False
    epoch = 50  # 一共循环50次
    batch_size = 16  # mini batch
    embedding_dim = 256
    hidden_dim = 512
    plot_every = 20
    max_gen_len = 200  # 生成诗歌的最长长度
    device = 'cuda'
    prefix_words = "横看成岭侧成峰，远近高低各不同。"
