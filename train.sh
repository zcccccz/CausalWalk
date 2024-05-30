# cd /data1/zhangcongzhi/causal_walk/CausalWalk

# causal walk fever 2-way (table 1 in the paper)
CUDA_VISIBLE_DEVICES="0" python train_causal_walk_fever_2way.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 5 \
--max_seq_length 128 

# causal walk politihop 3-way (table 1 in the paper)
CUDA_VISIBLE_DEVICES="0" python train_causal_walk_politihop_3way.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 20 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 

# causal walk politihop 2-way (table 3 in the paper)
CUDA_VISIBLE_DEVICES="0" python train_causal_walk_politihop_2way.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 10 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 