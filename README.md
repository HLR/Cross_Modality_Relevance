# Cross_Modality_Relevance
The source code of ACL 2020 paper: "Cross-Modality Relevance for Reasoning on Language and Vision"

Author: Chen Zheng, Quan Guo, Parisa Kordjamshidi

ArXiv pre-print version link: https://arxiv.org/abs/2005.06035

## data download link:
The data link is the same as the nlvr data page. We only need to download nlvr2. Please open the below page link and download it.
```
https://github.com/lil-lab/nlvr
```

## image bounding box feature download link:
Since the training image bounding box feature file is too large, we only provide the valid and test image feature files in this time.
After we find a larger storage space, we will consider to upload the training image bounding box feature file.
```
https://drive.google.com/file/d/1Ywpe-Vq5FKHCIPMMEToIDytlG_BMbxpt/view?usp=sharing
```

## model checkpoint download link:
The checkpoint file of nlvr2 parameter weights:
```
https://drive.google.com/file/d/10SBGpAXQ-tV0qpEjlxatyWYbwND5u0Hd/view?usp=sharing
```

## Tips
>- The image bounding box feature files are very large, espeically the training bounding box file has around 40GB.
>- Make sure both CPU and GPU memory are enough to load the data and model.

## experiment environment:
>- Machine: Lambda GPU machine.
>- GPU: TITAN RTX.

## Load conda environment:
>- conda env create -f cmr.yaml
>- source activate cmr

## How to run the code?
>- before runing the code, please make sure your config file is correct: configs/global_config.py
```python
python run_cmr_nlvr2_test.py
```
