import argparse
import sys
import paddle


def getArgs():
    parser = argparse.ArgumentParser(description='CGCNN')
    parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                        help='dataset options, started with the path to root dir, '
                             'then other options')
    parser.add_argument('--task', choices=['regression', 'classification'],
                        default='regression', help='complete a regression or '
                                                       'classification task (default: regression)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of dataConfig loading workers (default: 0)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default: '
                                           '0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                        metavar='N', help='milestones for scheduler (default: '
                                          '[100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                        help='number of training dataConfig to be loaded (default none)')
    train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                             help='number of training dataConfig to be loaded (default none)')
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of validation dataConfig to be loaded (default '
                             '0.1)')
    valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                             help='number of validation dataConfig to be loaded (default '
                                  '1000)')
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test dataConfig to be loaded (default 0.1)')
    test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                            help='number of test dataConfig to be loaded (default 1000)')

    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                        help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                        help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                        help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                        help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N',
                        help='number of hidden layers after pooling')

    args = parser.parse_args(sys.argv[1:])

    args.cuda = not args.disable_cuda and paddle.device.is_compiled_with_cuda()

    return args