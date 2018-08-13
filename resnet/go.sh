export CUDA_VISIBLE_DEVICES=1
python main.py --lr=0.01 --id Res18_sp3_no --sp 3 --lr_step 60 --epoch 200
python main.py --lr=0.01 --id Res18_sp2_no --sp 2 --lr_step 60 --epoch 200
python main.py --lr=0.01 --id Res18_sp1_no --sp 1 --lr_step 60 --epoch 200
python main.py --lr=0.01 --id Res18_sp0_no --sp 0 --lr_step 60 --epoch 200
