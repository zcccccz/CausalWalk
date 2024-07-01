# Causal Walk
Code and data for our paper "[Causal Walk: Debiasing Multi-hop Fact Verification with Front-door Adjustment](https://doi.org/10.1609/aaai.v38i17.29925)" in AAAI 2024.
![CausalWalk](CausalWalk.svg)

# Download
To reproduce the results in the paper, you need to download the dataset and our pre-trained model from [here](https://drive.google.com/drive/folders/19IdPsf0hZCnoHVnfHIelzEmnbkU3EF_6?usp=sharing).

Then you should organize them in the following format.
```
CausalWalk
    ├── data
    ├── pretrained_models
    ├── outputs
    ├── data_load_utils.py
    ├── models.py
    ├── train_causal_walk_fever_2way.py
    ├── train_causal_walk_politihop_2way.py
    ├── train_causal_walk_politihop_3way.py
    ├── train.sh
    └── utils.py
```

# Environment
Install python dependencies.
```
pip install -r requirements.txt
```

# Reproduction
Trained on the FEVER dataset and evaluated on Adversarial FEVER. (2-way)
```
CUDA_VISIBLE_DEVICES="0" python train_causal_walk_fever_2way.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128
```
Trained on the PolitiHop dataset and evaluated on Adversarial PolitiHop. (3-way)
```
CUDA_VISIBLE_DEVICES="0" python train_causal_walk_politihop_3way.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 
```
Trained on the PolitiHop dataset and evaluated on Symmetric PolitiHop. (2-way)
```
CUDA_VISIBLE_DEVICES="0" python train_causal_walk_politihop_2way.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 10 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 
```
# Citation
```
@inproceedings{zhang2024causal,
  title={Causal Walk: Debiasing Multi-Hop Fact Verification with Front-Door Adjustment},
  author={Zhang, Congzhi and Zhang, Linhai and Zhou, Deyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={19533--19541},
  year={2024}
}
```