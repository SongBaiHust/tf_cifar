CUDA_VISIBLE_DEVICES=$1 python train_val.py --learning_rate_decay_type=fixed --learning_rate=0.1 --optimizer=momentum --dataset_dir=/data3/xchen/workspace/datasets/cifar-10-batches-py --mode=train 
