import argparse


SUPPORTED_DATASETS = ('RITE-train', 'LES-AV', 'HRF-Karlsson-w1024')
SUPPORTED_BASE_CRITERIA = ('BCE3Loss', 'BCE3wminmaxLoss')
SUPPORTED_ADD_CRITERIA = ('SpCLLoss',)


def optional_choice(value, choices):
    if value in (None, '', 'None', 'none', 'null'):
        return None
    if value not in choices:
        raise argparse.ArgumentTypeError(
            f"expected one of {', '.join(choices)} or None, got {value!r}"
        )
    return value


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='RITE-train', choices=SUPPORTED_DATASETS)
parser.add_argument('--num_iterations', type=int, default=5)
parser.add_argument('--criterion', type=str, default='RRLoss', choices=('RRLoss',))
parser.add_argument('--base_criterion', type=str, default='BCE3Loss', choices=SUPPORTED_BASE_CRITERIA)
parser.add_argument(
    '--add_criterion',
    type=lambda value: optional_choice(value, SUPPORTED_ADD_CRITERIA),
    default=None,
)
parser.add_argument('--model', type=str, default='RRWNet', choices=('RRWNet',))
parser.add_argument('--num_folds', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-04)
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--base_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--n_proc', type=int, default=1)
parser.add_argument('--data_folder', type=str, default='./_Data/')
parser.add_argument('--version', type=str, default='')
parser.add_argument('--use_checkpoint', type=int, default=0, choices=(0, 1))
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--scheduler_path', type=str, default=None)
args = parser.parse_args()
if args.use_checkpoint and (not args.checkpoint_path or not args.scheduler_path):
    parser.error('--use_checkpoint 1 requires --checkpoint_path and --scheduler_path')

dataset = args.dataset
training_folder = f'/mnt/nasv3/zs/RRWNet_zs/train/__training/{args.version}/{dataset}'

num_folds = args.num_folds
active_folds = [0]
learning_rate = args.learning_rate
num_epochs = args.num_epochs

model = args.model
in_channels = args.in_channels
out_channels = args.out_channels
base_channels = args.base_channels
num_iterations = args.num_iterations

criterion = args.criterion
base_criterion = args.base_criterion
add_criterion = args.add_criterion

n_proc = args.n_proc
gpu_id = args.gpu_id
use_checkpoint = args.use_checkpoint
checkpoint_path = args.checkpoint_path
scheduler_path = args.scheduler_path


if dataset == 'RITE-train':
    images = [
        33, 24, 36, 30, 25, 29, 40, 21, 37, 34, 35, 32, 27, 39, 26, 38, 28, 23,
        31, 22
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': 'RITE/train/av3',
            'pattern': '[0-9]+[.]png'
        },
        'original': {
            'path': 'RITE/train/enhanced',
            'pattern': '[0-9]+[.]png'
        },
        'mask': {
            'path': 'RITE/train/enhanced_masks',
            'pattern': '[0-9]+[.]png'
        },
    }
    num_imgs_val = 4
elif dataset == 'LES-AV':
    images = [
        12, 15, 33, 34, 37, 42, 49, 53, 90, 111, 119,
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': 'LES-AV/train/av3',
            'pattern': '[0-9]+[.]png'
        },
        'original': {
            'path': 'LES-AV/train/enhanced',
            'pattern': '[0-9]+[.]png'
        },
        'mask': {
            'path': 'LES-AV/train/enhanced_masks',
            'pattern': '[0-9]+[.]png'
        },
    }
    num_imgs_val = 2
elif dataset == 'HRF-Karlsson-w1024':
    images = [
        '06_dr', '06_g', '06_h', '07_dr', '07_g', '07_h', '08_dr', '08_g',
        '08_h', '09_dr', '09_g', '09_h', '10_dr', '10_g', '10_h', '11_dr',
        '11_g', '11_h', '12_dr', '12_g', '12_h', '13_dr', '13_g', '13_h',
        '14_dr', '14_g', '14_h', '15_dr', '15_g', '15_h',
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': 'HRF_AVLabel_191219/train_karlsson_w1024/av3',
            'pattern': '[0-9]+_.+[.]png'
        },
        'original': {
            'path': 'HRF_AVLabel_191219/train_karlsson_w1024/enhanced',
            'pattern': '[0-9]+_.+[.]png'
        },
        'mask': {
            'path': 'HRF_AVLabel_191219/train_karlsson_w1024/enhanced_masks',
            'pattern': '[0-9]+_.+[.]png'
        },
    }
    num_imgs_val = 6
