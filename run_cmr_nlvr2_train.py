import os
import collections

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

# import sys
# sys.path.append('../')
from configs.global_config import GLOBAL_CONFIG
from model.cmr_nlvr2_model import Cross_Modality_Relevance
from BERT_related.optimization import BertAdam
from data_preprocessing.cmr_nlvr2_data import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator


class CMR:
    def __init__(self, cfg):
        self.train_tuple = get_tuple(
            cfg.train, bs=cfg.batch_size, shuffle=True, drop_last=True
        )
        if cfg.valid != "":
            valid_bsize = 2048 if cfg.multiGPU else 512
            self.valid_tuple = get_tuple(
                # cfg.valid, bs=valid_bsize,
                cfg.valid, bs=cfg.batch_size,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = Cross_Modality_Relevance(cfg)

        if cfg.load_cmr is not None:
            self.model.load_state_dict(torch.load(cfg.load_cmr))

        if cfg.multiGPU:
            self.model.bert_encoder.multi_gpu()
        self.model = self.model.cuda()

        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in cfg.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * cfg.epochs)
            print("Total Iters: %d" % t_total)
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=cfg.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = cfg.optimizer(list(self.model.parameters()), cfg.lr)
        # self.optim = BertAdam(list(self.model.parameters()), lr=cfg.lr, warmup=0.1,)

        self.output = cfg.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if cfg.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(cfg.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label) in iter_wrapper(enumerate(loader)):
                self.model.train()

                self.optim.zero_grad()
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)

                loss = self.mce_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if cfg.tqdm else (lambda x: x)
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, label) in iter_wrapper(enumerate(loader)):
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)
        self.model.load_state_dict(state_dict)


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = NLVR2Dataset(splits)
    tset = NLVR2TorchDataset(dset)
    evaluator = NLVR2Evaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)



if __name__ == "__main__":
    #### gpu environment
    cfg = GLOBAL_CONFIG()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in cfg.visable_gpus)
    
    cpu = torch.device('cpu')

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not cfg.no_cuda else "cpu")

    n_gpu = torch.cuda.device_count()
    print('number of the gpu devices------------------------------------->:' , device, n_gpu)

    #### initial the model
    nlvr2 = CMR(cfg)

    # Train 
    print('CMR: Load Train data!')
    if nlvr2.valid_tuple is not None:
        print('CMR: Load Valid data!')
    else:
        print("CMR: No valid data, only train data!")
    nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)
