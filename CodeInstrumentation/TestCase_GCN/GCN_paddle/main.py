import os
import sys
import time
import warnings
from random import sample
import numpy as np
import warnings

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.regularizer import L2Decay

from dataset import CIFData, collate_pool, get_train_val_test_loader, get_dataset
from model import CrystalGraphConvNet
from utils import save_checkpoint, AverageMeter, mae, class_eval, Normalizer
from argParser import getArgs

args = getArgs()
warnings.filterwarnings("ignore", category=Warning)
current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def main():
    gpu_device = paddle.device.set_device('gpu:0')
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    test_loader = get_dataset(dataset, collate_fn, batch_size=8, num_workers=args.workers)
    if args.task == 'classification':
        normalizer = Normalizer(paddle.to_tensor([0., 0.], dtype='float32'))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 dataConfig points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    structures, _, _ = dataset[0]
    input_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(input_atom_fea_len, nbr_fea_len)
    # paddle 会根据下载的paddle包自动选择模型的处理器，无需手动指定
    model.to(gpu_device)

    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    print("---------Evaluate Model on TestSet---------------")
    best_checkpoint = paddle.load(current_dir+'model_best.pth.tar')
    model.set_state_dict(best_checkpoint['state_dict'])
    normalizer.set_state_dict(best_checkpoint['normalizer'])
    validate(test_loader, model, criterion, normalizer, gpu_device, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer, gpu_device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.cuda:
            input_var = (input[0].to(gpu_device), input[1].to(gpu_device),
                         input[2].to(gpu_device),
                         [crys_idx.to(gpu_device) for crys_idx in input[3]])
        else:
            input_var = (input[0], input[1], input[2], input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.reshape([-1]).long()
        if args.cuda:
            target_var = target_normed.to(gpu_device)
        else:
            target_var = target_normed

        output = model(*input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.cpu()), target)
            losses.update(loss.cpu())  # , target.size(0)
            mae_errors.update(mae_error)  # , target.size(0)
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.cpu(), target)
            losses.update(loss.cpu())  # , target.size(0))
            accuracies.update(accuracy)  # , target.size(0))
            precisions.update(precision)  # , target.size(0))
            recalls.update(recall)  # , target.size(0))
            fscores.update(fscore)  # , target.size(0))
            auc_scores.update(auc_score)  # , target.size(0))

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)))
                # print('Test: [{0}][{1}/{2}]\t'
                #       'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                #       'Data {data_time_val:.3f} ({data_time_avg:.3f})\t'
                #       'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                #       'MAE {mae_errors_val:.3f} ({mae_errors_avg:.3f})'
                #       .format(epoch, i, len(train_loader),
                #               batch_time_val=batch_time.val, batch_time_avg=batch_time.avg,
                #               data_time_val=data_time.val, data_time_avg=data_time.avg,
                #               loss_val=losses.val, loss_avg=losses.avg,
                #               mae_errors_val=mae_errors.val, mae_errors_avg=mae_errors.avg))
            else:
                print("HI")
                # print('Epoch: [{0}][{1}/{2}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                #       'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                #       'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                #       'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                #       'AUC {auc.val:.3f} ({auc.avg:.3f})'
                #       .format(
                #         epoch, i, len(train_loader), batch_time=batch_time,
                #         data_time=data_time, loss=losses, accu=accuracies,
                #         prec=precisions, recall=recalls, f1=fscores,
                #         auc=auc_scores))


def validate(val_loader, model, criterion, normalizer, gpu_device, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with paddle.no_grad():
                input_var = (input[0].to(gpu_device), input[1].to(gpu_device),
                             input[2].to(gpu_device),
                             [crys_idx.to(gpu_device) for crys_idx in input[3]])
        else:
            with paddle.no_grad():
                input_var = (input[0], input[1], input[2], input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.reshape([-1]).long()
        if args.cuda:
            with paddle.no_grad():
                target_var = target_normed.to(gpu_device)
        else:
            with paddle.no_grad():
                target_var = target_normed

        output = model(*input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.cpu()), target)
            losses.update(loss.cpu())  # , target.size(0)
            mae_errors.update(mae_error)  # , target.size(0)
            if test:
                test_pred = normalizer.denorm(output.cpu())
                test_target = target
                test_preds += test_pred.reshape([-1]).tolist()
                test_targets += test_target.reshape([-1]).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.cpu(), target)
            losses.update(loss.cpu())  # , target.size(0))
            accuracies.update(accuracy)  # , target.size(0))
            precisions.update(precision)  # , target.size(0))
            recalls.update(recall)  # , target.size(0))
            fscores.update(fscore)  # , target.size(0))
            auc_scores.update(auc_score)  # , target.size(0))
            if test:
                test_pred = paddle.exp(output.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.reshape([-1]).tolist()
                test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'.format(i, len(val_loader)))
                # print('Test: [{0}/{1}]\t'
                #       'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                #       'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                #       'MAE {mae_errors_val:.3f} ({mae_errors_avg:.3f})'
                #       .format(i, len(val_loader),
                #               batch_time_val=batch_time.val, batch_time_avg=batch_time.avg,
                #               loss_val=losses.val, loss_avg=losses.avg,
                #               mae_errors_val=mae_errors.val, mae_errors_avg=mae_errors.avg))
            else:
                print("HI")
                # print('Test: [{0}/{1}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                #       'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                #       'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                #       'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                #       'AUC {auc.val:.3f} ({auc.avg:.3f})'
                #       .format(
                #         i, len(val_loader), batch_time=batch_time, loss=losses,
                #         accu=accuracies, prec=precisions, recall=recalls,
                #         f1=fscores, auc=auc_scores))
    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'

    if args.task == 'regression':
        print(f' {star_label} MAE {mae_errors.avg}')
        return mae_errors.avg
    else:
        print(f' {star_label} AUC {auc_scores.avg}')
        return auc_scores.avg


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
