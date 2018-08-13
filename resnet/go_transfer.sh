export CUDA_VISIBLE_DEVICES=2
python main.py --lr=0.1 --id Res18_trans_sp3_no --sp 3 --transfer --pretrained checkpoint/ckpt.t7 --lr_step 80 --epoch 150
python main.py --lr=0.1 --id Res18_trans_sp2_no --sp 2 --transfer --pretrained checkpoint/ckpt.t7 --lr_step 80 --epoch 150
python main.py --lr=0.1 --id Res18_trans_sp1_no --sp 1 --transfer --pretrained checkpoint/ckpt.t7 --lr_step 80 --epoch 150
python main.py --lr=0.1 --id Res18_trans_sp0_no --sp 0 --transfer --pretrained checkpoint/ckpt.t7 --lr_step 80 --epoch 150
