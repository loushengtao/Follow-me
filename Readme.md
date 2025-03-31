# [**Follow-me: Deceiving Trackers with Fabricated Paths**](https://dl.acm.org/doi/10.1145/3581783.3611935)<br>


<div align="center">
<img src="https://github.com/loushengtao/Follow-me/blob/main/doc/demo.gif">
</div>



## Introduction
In this work, we propose a novel attack algorithm
that is capable of fooling victim models by enforcing their predictions to be consistent with arbitrary counterfeit paths.

Key contributions include:
- Our attack algorithm is the first
one that is capable of satisfying various generic requests,
e.g., following arbitrary counterfeit paths.
- Our novel attack algorithm is able to fool SiamRPN-based
trackers such that their outputs are smooth yet consistent
with pre-define counterfeit paths.
-  We beat the SOTA methods by a large margin under conventional tracking-based evaluation metrics. Numbers on
our novel evaluation metrics further showcase the path following ability of our algorithm.



## Framework

<p align="center">
<img src="https://github.com/loushengtao/Follow-me/blob/main/doc/framework.jpg" width=100%>
<p>

## Installation

1. Clone the repository

```sh
git clone https://github.com/loushengtao/Follow-me.git
```

2. Create a conda environment and install the required packages

```sh
conda create -n fm python==3.9
conda activate fm
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
3. Download the victim model

Download the ``model.pth`` for the ``siamrpn_r50_l234_dwxcorr`` model from the [PySOT Model Zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md) and place it in the corresponding folder within the ``experiment`` directory.

## Victim data preparation


We currently support several victim data types: ``frames``, ``avi/mp4``, and ``dataset``. For the ``frames`` and ``avi/mp4`` types, the data should be organized within the ``testdata/`` folder according to the structure outlined below. 

```
testdata/
|-- victim_name/
|   |-- ori_frames/
|   |   |--[frames data]
|
|-- victim_name/
|   |-- victim.avi(mp4)

```

Data for the ``dataset/`` type does not need to be placed in ``testdata/`` manually. Instead, please follow the instructions at [pysot](https://github.com/STVIR/pysot) to download the SOT datasets.

## Attack

Refer to the ``attack.sh`` for the attack script.

1. Victim data of the ``frames`` type:

```
python main.py \
    --mode attack \   # attack mode
    --victim_dir testdata/ants \  # victim data path
    --data_type frames \  # victim data type
    --cases arb case1 case2 case3   # attack cases
```

2. Victim data of the ``avi/mp4`` type: 

```
python main.py \
    --mode attack \
    --victim_dir testdata/bag \ 
    --data_type avi \
    --cases arb case3
```

2. Victim data of the ``dataset`` type: 

```
python main.py \
    --mode attack \
    --victim_dir sotdataset/VOT2018 \   # SOT dataset path
    --data_type dataset \
    --cases arb case1 case2 case3 \
    --dataset_name VOT2018 \  # SOT dataset name
    --datait_name rabbit  # selected sample

```


## Visualize

In ``vis.sh``, select the script corresponding to your data type and run it.

```
bash vis.sh
```

## BibTeX
```bibtex
@inproceedings{lou2023follow,
  title={Follow-me: Deceiving Trackers with Fabricated Paths},
  author={Lou, Shengtao and Liu, Buyu and Bao, Jun and Ding, Jiajun and Yu, Jun},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8808--8818},
  year={2023}
}

```

