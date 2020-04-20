class GLOBAL_CONFIG(object):
    def __init__(self):
        super(GLOBAL_CONFIG, self).__init__()
        
        ###### data location
        self.train = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/data/nlvr2_preprocessing_data/train.json'
        self.valid = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/data/nlvr2_preprocessing_data/valid.json'
        self.test = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/data/nlvr2_preprocessing_data/valid.json'

        ###### img feat location
        self.train_img_feat = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/data/nlvr2_imgfeat/train_obj36.tsv'
        self.valid_img_feat = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/data/nlvr2_imgfeat/valid_obj36.tsv'
        self.test_img_feat = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/data/nlvr2_imgfeat/test_obj36.tsv'

        ###### output dir
        self.output = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/experiments/'

        ###### load model
        self.load_cmr = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/checkpoints/cmr_finetune.pth' ### chen
        self.load_pretrain = '/home/hlr/shared/data/chenzheng/data/cmr_nlvr2/checkpoints/cmr_pretrain.pth'
        self.from_scratch = None

        self.load_lxmert_qa = None


        ###### hyper-parameters
        self.epochs = 50
        self.lr = 1e-5
        self.batch_size = 16
        self.optim = 'bert'
        self.dropout = 0.1
        self.max_seq_length = 20
        self.seed = 9595

        ###### experiment hardware parameters
        self.visable_gpus = 1,
        self.no_cuda = False
        self.multiGPU = False
        self.numWorkers = 0
        self.tqdm = True
        self.mce_loss = False

        ###### delete later:
        self.task_matched = False
        self.task_mask_lm = False
        self.task_obj_predict = False
        self.task_qa = False
        self.visual_losses = 'obj,attr,feat'
        self.qa_sets = None
        self.word_mask_rate = 0.15
        self.obj_mask_rate = 0.15


