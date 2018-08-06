# Pathology_Analysis
Pathology_Analysis

# files
./read: take network.bin and generate data.csv
./Preprocess.ipynb: Use all 1430 pos and sample 2860 neg to generate 3300 (train) + 990 (test) dataset split.
./resnet/go.sh: train from scratch and show the result
./resnet/go_transfer.sh: transfer from pre-trained model on cifar10 and train the model for target pathology data

#bash usage 
(under the resnet folder) (dependency: pytorch)
bash go.sh
bash go_transfer.sh

#command line guidance
Please try python main.py -h

PyTorch cifar10 for pathology

>optional arguments:
>  -h, --help            show this help message and exit
>  --lr LR               learning rate
>  --lr_step             lr_step
>  --epoch EPOCH         epoch
>  --resume, -r          resume from checkpoint
>  --sp SP               splits
>  --img_dir             img dir
>  --id ID               model id
>  --transfer            transfer to new task with pretrained
>  --pretrained          the pretrained model

