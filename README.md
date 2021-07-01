# CaDSI_Implementation_Code
We did experiment under the following environment:
Python 3.6.5
Tensorflow Version: 1.14.0

Running example for douban book data set:
python CaDSI.py --dataset Douban_Book --batch_size 2000 --n_layers 1 --n_iterations 2 --corDecay 0.01 --n_factors 4 --show_step 3 --lr 0.001 --epoch 100 --embed_size 128 --cor_flag 0  --Ks '[10,15,20,25,30,35,40,60,80,100]' --verbose 1 --save_flag 1 --save_name
