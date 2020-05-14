class GLOBAL_CONFIG(object):
    def __init__(self):
        super(GLOBAL_CONFIG, self).__init__()

        ###### data location
        self.train = '/tank/space/chen_zheng/data/cmr_nlvr2/data/nlvr2_preprocessing_data/train.json'
        self.valid = '/tank/space/chen_zheng/data/cmr_nlvr2/data/nlvr2_preprocessing_data/valid.json'
        self.test = '/tank/space/chen_zheng/data/cmr_nlvr2/data/nlvr2_preprocessing_data/test.json'

        ###### img feat location
        self.train_img_feat = '/tank/space/chen_zheng/data/cmr_nlvr2/data/nlvr2_imgfeat/train_obj36.tsv'
        self.valid_img_feat = '/tank/space/chen_zheng/data/cmr_nlvr2/data/nlvr2_imgfeat/valid_obj36.tsv'
        self.test_img_feat = '/tank/space/chen_zheng/data/cmr_nlvr2/data/nlvr2_imgfeat/test_obj36.tsv'

        ###### output dir
        # self.output = '/tank/space/chen_zheng/data/cmr_nlvr2/experiments/'  ## gpu 7
        self.output = '/tank/space/chen_zheng/data/cmr_nlvr2/experiments_new/'    ## gpu 1

        ###### load model
        # self.load_cmr = '/tank/space/chen_zheng/data/cmr_nlvr2/checkpoints/cmr_finetune.pth' ### chen
        self.load_cmr = '/tank/space/chen_zheng/data/cmr_nlvr2/experiments_new/BEST.pth' ### chen
        self.from_scratch = None

        ###### hyper-parameters
        self.epochs = 20
        self.lr = 1e-5
        self.batch_size = 16
        self.optim = 'bert'
        self.dropout = 0.1
        self.max_seq_length = 20
        self.seed = 9595

        ###### experiment hardware parameters
        self.visable_gpus = 7,
        self.no_cuda = False
        self.multiGPU = False
        self.numWorkers = 0
        self.tqdm = True
        self.mce_loss = False
