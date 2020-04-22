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

        self.model = Cross_Modality_Relevance(cfg)
        self.model.load_state_dict(torch.load(cfg.load_cmr))
        self.model = self.model.cuda()
        self.output = cfg.output

        os.makedirs(self.output, exist_ok=True)


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
    print('number of the gpu devices------------------------------------->:' , n_gpu)

    #### initial the model
    nlvr2 = CMR(cfg)

    # Test
    if cfg.test is not None:
        print('CMR: Begin valid or test dataset prediction')
        result = nlvr2.evaluate(
            get_tuple(cfg.test, bs=cfg.batch_size,
                        shuffle=False, drop_last=False),
            dump=os.path.join(cfg.output, '%s_predict.csv' % cfg.test)
        )
        print(result)

        nlvr2.predict(
            get_tuple(cfg.test, bs=cfg.batch_size,
                    shuffle=False, drop_last=False),
                    dump=os.path.join(cfg.output, 'final_predict.csv'))
    else:
        print('CMR: Please provide the correct test dataset path!!!')