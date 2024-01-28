# pathological heterogeneous setting
python main.py --dataset cifar100tpds --alg feddwa  --model cnn --client_num 20 --Tg 100 --E 1 --lr 0.01 --non_iidtype 8 --client_frac 1 --seed 2 --times 1 --feddwa_topk 5 --gpu 1
python main.py --dataset cifar10tpds --alg feddwa  --model cnn --client_num 20 --Tg 100 --E 1 --lr 0.01 --non_iidtype 8 --client_frac 1 --seed 2 --times 1 --feddwa_topk 5 --gpu 3

# practical heterogeneous setting 1
python main.py --dataset cifar100tpds --alg feddwa --ratio_noniid10 0.8 --model cnn --client_num 20 --Tg 100 --E 1 --lr 0.01 --non_iidtype 10 --client_frac 1 --seed 2 --times 1 --feddwa_topk 5 --gpu 0
python main.py --dataset cifar10tpds --alg feddwa --ratio_noniid10 0.8 --model cnn --client_num 20 --Tg 100 --E 1 --lr 0.01 --non_iidtype 10 --client_frac 1 --seed 2 --times 1 --feddwa_topk 5 --gpu 2

# practical heterogeneous setting 2
python main.py --dataset cifar100tpds --alg feddwa  --model cnn --client_num 100 --Tg 100 --E 1 --lr 0.01 --non_iidtype 9 --client_frac 0.2 --seed 2 --times 1 --feddwa_topk 5 --gpu 2
python main.py --dataset cifar10tpds --alg feddwa  --model cnn --client_num 100 --Tg 100 --E 1 --lr 0.01 --non_iidtype 9 --client_frac 0.2 --seed 2 --times 1 --feddwa_topk 5 --gpu 3
