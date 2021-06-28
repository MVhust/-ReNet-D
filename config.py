class DefaultConfig(object):
    model = 'CAE8'       # 使用模型，名字与models/__init__.py中的名字一致
    pattern_index = '6'

    train_raw_data_root = './data/pattern/pattern'    # 训练原始数据集（正样本）的存放路径
    #val_raw_data_root = './data/val/'
    test_raw_data_root = './data/bad_pattern/bad_pattern'   # 测试原始数据集（负样本）的存放路径
    load_model_path = './checkpoints/'      # 加载、保存训练模型的路径
    img_show_path = './img_show/'
    log_dir = './logs/'

    train_height = 512      # 原始训练数据（正样本）的图像高度
    train_width = 512       # 原始训练数据（正样本）的图像宽度
    test_height = 512       # 原始测试数据（负样本）的图像高度
    test_width = 512       # 原始测试数据（负样本）的图像高度
    patch_size = 32
    train_patch_stride = 4
    test_patch_stride = 32
    channel = 3
    use_gpu = True
    train_batch_size = 512
    print_freq = 20
    max_epoch = 4000
    lr = 0.0001
    lr_decay = 0.95
    weight_decay = 1e-5
    momentum = 0.9

    def parse(self, dicts):
        """
        根据传入字典dicts更新config参数
        :param dicts:
        :return: None
        """
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)
