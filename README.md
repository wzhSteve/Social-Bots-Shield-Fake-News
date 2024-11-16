# Data and Code for "Bots Shield Fake News: Adversarial Attack on User Engagement based Fake News Detection"

## Device
We conduct all experiments on a server running Ubuntu 20.04 with an NVIDIA RTX 3090 GPU. 

## Requirements
```
python==3.8.16
numpy==1.24.3
torch==1.10.1+cu113
torch-geometric==2.4.0
torch-scatter==2.0.9
torch-sparse==0.6.12
deeprobust==0.2.8
tqdm==4.64.1
scipy==1.10.1
pytz==2023.3
```

## Data
Our work relies on the `PolitiFact` and `GossipCop` datasets from the [Decor](https://github.com/jiayingwu19/DECOR), which is built upon the [FakeNewsNet benchmark](https://github.com/KaiDMML/FakeNewsNet).

Limited by the maximum file size of supplementary material, we only provide the `PolitiFact` and `GossipCop1` datasets.

If you need to reproduce the results for other sub-datasets of GossipCop, you download the full dataset from [Decor](https://github.com/jiayingwu19/DECOR) and then utilize `attack_method/split_dataset.py` to divide the dataset. The command is as follows:
```
python -u attack_method/split_dataset.py
```

## Run Attack Methods
To run QSA-AC on a dataset, the corresponding command is as follows:
```bash
python -u attack_method/test_qsa_ac.py --dataset [dataset_name] --subset [subset_id] --victim_model [victim_model_name] --topk [size_of_candidate_set] --m [threshold] --alpha [alpha] --constrain --retrain
```

`[dataset_name]`: politifact / gossipcop

`[subset_id]`: subset1 / subset2 / subset3

`[vicitm_model_name]`: gcn / gat / decor / mid-gcn

`[size_of_candidate_set]`: 30 / 50 / 70

`[threshold]`: 1 / 2

`[alpha]`: 1 / 2 / 3 / 4


We provide detailed commands in `attack.sh` and the checkpoint for all surrogate models and victim models in the folder `attack_method/results/`.
## Cite us

```
@inproceedings{wang2024bots,
  title={Bots Shield Fake News: Adversarial Attack on User Engagement based Fake News Detection},
  author={Wang, Lanjun and Wang, Zehao and Wu, Le and Liu, An-An},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={2369--2378},
  year={2024}
}
```
