# coding=utf-8
import json

import numpy as np
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils.nlvr2_utils import load_obj_tsv
from configs.global_config import GLOBAL_CONFIG

class NLVR2Dataset:
    def __init__(self, data_loc: str):

        # Loading datasets to data
        self.data = []
        self.data.extend(json.load(open(data_loc)))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['uid']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)

class NLVR2TorchDataset(Dataset):
    def __init__(self, dataset: NLVR2Dataset):
        super().__init__()
        self.raw_dataset = dataset
        cfg = GLOBAL_CONFIG()
        # Loading detection features to img_data
        img_data = []
        img_data.extend(load_obj_tsv(cfg.train_img_feat, topk=-1))
        img_data.extend(load_obj_tsv(cfg.valid_img_feat, topk=-1))
        img_data.extend(load_obj_tsv(cfg.test_img_feat, topk=-1))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        ques_id = datum['uid']
        ques = datum['sent']

        # Get image info
        boxes2 = []
        feats2 = []
        for key in ['img0', 'img1']:
            img_id = datum[key]
            img_info = self.imgid2img[img_id]
            boxes = img_info['boxes'].copy()
            feats = img_info['features'].copy()
            assert len(boxes) == len(feats)

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
            boxes[..., (0, 2)] /= img_w
            boxes[..., (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

            boxes2.append(boxes)
            feats2.append(feats)
        feats = np.stack(feats2)
        boxes = np.stack(boxes2)

        # Create target
        if 'label' in datum:
            label = datum['label']
            return ques_id, feats, boxes, ques, label
        else:
            return ques_id, feats, boxes, ques


class NLVR2Evaluator:
    def __init__(self, dataset: NLVR2Dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (idt, ans))
