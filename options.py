import argparse
import os

global A
NYU = 1
UM116 = 2
ADNI = 3
nyu200 = 4
um200 = 5
SITE1 = 8
SITE6 = 6

# 在程序开始时创建必要的目录
os.makedirs('./runs', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

# 重点说一下这里的情况
A = 3


def getargs():
    parser = argparse.ArgumentParser(description='xx')
    parser.add_argument('--exp_name', type=str, default='H undec conv down brain', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--minibatch_size', type=int, default=16, metavar='batch_size',
                        help='Size of minibatch_size)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N', #60,90
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=int, default=0,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0009, metavar='LR', #0.0008,0.00009
                        help='learning rate (default: 0.006,0.001 0.1 if using sgd)')
    parser.add_argument('--layer', type=int, default=1, help='layer of model')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.006,0.001 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='enables CUDA training')
    parser.add_argument('--no_cuda', type=bool, default='False',
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout rate')
    parser.add_argument('--mult_num', type=int, default=8, help=' ')
    parser.add_argument('--ratio', type=float, default=0.35, help=' ')
    parser.add_argument('--num_pooling', type=int, default=2, help='')
    parser.add_argument('--k_fold', default=5, help='fold = 0,1,2,3,4')
    parser.add_argument('--cluster', type=int, default=16, help='num of roi')  # 16的时候很不错
    parser.add_argument('--dataset', type=int, default=A, help='num of roi')

    parser.add_argument('--model_path', type=str, default='', metavar='N')
    parser.add_argument('--Num', type=int, default=0, help=' ')
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--Gbias', type=bool, default=False, help='if bias ')
    if (A == 1):
        parser.add_argument('--data_path', type=str, default="data/NYU116(1)")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=175, help=' ')
        parser.add_argument('--input_size', type=int, default=175, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help=' ')
    if (A == 2):
        parser.add_argument('--data_path', type=str, default="data/UM116")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=295, help=' ')
        parser.add_argument('--input_size', type=int, default=295, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help=' ')
    if (A == 3):
        parser.add_argument('--data_path', type=str, default="data/adni2")
        parser.add_argument('--output_size', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--input_size', type=int, default=137)
        parser.add_argument('--norm_num', type=int, default=116)
    if (A == 4):
        parser.add_argument('--data_path', type=str, default="data/NYUcc200")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=175, help=' ')
        parser.add_argument('--input_size', type=int, default=175, help=' ')
        parser.add_argument('--norm_num', type=int, default=200, help=' ')
    if (A == 5):
        parser.add_argument('--data_path', type=str, default="data/UMcc200")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=295, help=' ')
        parser.add_argument('--input_size', type=int, default=295, help=' ')
        parser.add_argument('--norm_num', type=int, default=200, help=' ')

    if (A == 8):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/MDD/SITE1")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=1833, help=' ')
        parser.add_argument('--input_size', type=int, default=1833, help=' ')
        parser.add_argument('--norm_num', type=int, default=3200, help=' ')
    if (A == 6):
        parser.add_argument('--data_path', type=str, default="D:/wjz/data/MDD/SITE20")
        parser.add_argument('--output_size', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=232, help=' ')
        parser.add_argument('--input_size', type=int, default=232, help=' ')
        parser.add_argument('--norm_num', type=int, default=116, help=' ')

    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='tensorboard log directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='model save directory')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='early stopping patience')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging')
    parser.add_argument('--normalize_data', type=bool, default=True,
                        help='whether to normalize the data')
    parser.add_argument('--pad_data', type=bool, default=True,
                        help='whether to pad the data')
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                        help='mask ratio for view generation')

    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='latent dimension size')

    # 添加编译相关参数
    parser.add_argument('--use_compile', type=bool, default=True,
                        help='whether to use torch.compile')
    parser.add_argument('--compile_backend', type=str, default='aot_eager',
                        help='backend for torch.compile')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        help='compilation mode')

    # 添加内存管理相关参数
    parser.add_argument('--memory_fraction', type=float, default=0.95,
                        help='fraction of GPU memory to use')
    parser.add_argument('--allow_tf32', type=bool, default=True,
                        help='whether to allow TF32 on Ampere')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='whether to use cudnn benchmark')

    # 优化相关参数
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--persistent_workers', type=bool, default=True)

    # 添加掩码相关参数
    parser.add_argument('--node_mask_ratio', type=float, default=0.1,
                        help='节点掩码比率')
    parser.add_argument('--edge_mask_ratio', type=float, default=0.1,
                        help='边掩码比率')
    parser.add_argument('--feature_mask_ratio', type=float, default=0.1,
                        help='特征掩码比率')

    # 添加num_classes参数
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes in the dataset')

    args = parser.parse_args()

    # 修改这部分，统一使用二分类
    if args.dataset == 3:  # ADNI
        args.num_classes = 2
    elif args.dataset in [1, 2]:  # NYU, UM116
        args.num_classes = 2
    elif args.dataset == 4:  # NYU200
        args.num_classes = 2
    elif args.dataset == 5:  # UM200
        args.num_classes = 2
    elif args.dataset in [6, 8]:  # SITE1, SITE6
        args.num_classes = 2

    return args

# 站点0适合 150  epoch    0.0007或0.0006的lr