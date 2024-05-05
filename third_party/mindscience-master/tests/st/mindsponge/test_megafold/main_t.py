# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval script"""
import pickle
import argparse
import os
import stat
import json
import time
import numpy as np

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, load_checkpoint
from mindsponge.cell.amp import amp_convert
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction

from data import Feature, RawFeatureGenerator, get_raw_feature
from model import MegaFold, compute_confidence
from module.fold_wrapcell import TrainOneStepCell, WithLossCell
from module.lr import cos_decay_lr

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', default="./config/data.yaml", help='data process config')
parser.add_argument('--model_config', default="./config/model.yaml", help='model config')
parser.add_argument('--evogen_config', default="./config/evogen.yaml", help='evogen config')
parser.add_argument('--input_path', help='processed raw feature path')
parser.add_argument('--pdb_path', type=str, help='Location of training pdb file.')
parser.add_argument('--use_pkl', default=False, help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--is_traing', default=False, help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--mixed_precision', default=1, type=int, help='DEVICE_ID')
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--checkpoint_path_assessment', help='assessment model checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--is_training', type=bool, default=False, help='is training or not')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
parser.add_argument('--run_distribute', type=bool, default=False, help='run distribute')
parser.add_argument('--resolution_data', type=str, default=None, help='Location of resolution data file.')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
parser.add_argument('--gradient_clip', type=float, default=0.1, help='gradient clip value')
parser.add_argument('--total_steps', type=int, default=9600000, help='total steps')
parser.add_argument('--decoy_pdb_path', type=str, help='Location of decoy pdb file.')
parser.add_argument('--run_assessment', type=int, default=0, help='Run pdb assessment.')
parser.add_argument('--run_evogen', type=int, default=0, help='Run pdb assessment.')
parser.add_argument('--seq_len', type=int, default=1536, help='Seq len.')
arguments = parser.parse_args()


def fold_infer(args):
    '''mega fold inference'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.eval.crop_size = args.seq_len
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    load_checkpoint(args.checkpoint_path, megafold)
    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megafold, fp32_white_list)
    else:
        megafold.to_float(mstype.float32)

    seq_files = os.listdir(args.input_path)

    if not args.use_pkl:
        feature_generator = RawFeatureGenerator(database_search_config=data_cfg.database_search)
    else:
        feature_generator = None
    for seq_file in seq_files:
        t1 = time.time()
        seq_name = seq_file.split('.')[0]
        raw_feature = get_raw_feature(os.path.join(args.input_path, seq_file), feature_generator, args.use_pkl)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=args.mixed_precision)
        prev_pos = Tensor(prev_pos)
        prev_msa_first_row = Tensor(prev_msa_first_row)
        prev_pair = Tensor(prev_pair)
        t2 = time.time()
        for i in range(data_cfg.common.num_recycle):
            feat_i = [Tensor(x[i]) for x in feat]
            result = megafold(*feat_i,
                              prev_pos,
                              prev_msa_first_row,
                              prev_pair)
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = result
        t3 = time.time()
        final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
        final_atom_mask = feat[16][0][:ori_res_length]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
        confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)

        b_factors = plddt[:, None] * final_atom_mask

        unrelaxed_protein = from_prediction(final_atom_positions,
                                            final_atom_mask,
                                            feat[4][0][:ori_res_length],
                                            feat[17][0][:ori_res_length],
                                            b_factors)
        pdb_file = to_pdb(unrelaxed_protein)
        os.makedirs(f'./result/{seq_name}', exist_ok=True)
        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        pdb_path = f'./result/{seq_name}/unrelaxed_{seq_name}.pdb'
        with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
            fout.write(pdb_file)
        t4 = time.time()
        timings = {"pre_process_time": round(t2 - t1, 2),
                   "predict time ": round(t3 - t2, 2),
                   "pos_process_time": round(t4 - t3, 2),
                   "all_time": round(t4 - t1, 2),
                   "confidence": round(confidence, 2)}

        print(timings)
        with os.fdopen(os.open(f'./result/{seq_name}/timings', os_flags, os_modes), 'w') as fout:
            fout.write(json.dumps(timings))


def fold_train(args):
    """megafold train"""
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.common.max_extra_msa = 1024
    data_cfg.eval.max_msa_clusters = 128
    data_cfg.eval.crop_size = 256
    model_cfg.is_training = True
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megafold, fp32_white_list)
    else:
        megafold.to_float(mstype.float32)

    net_with_criterion = WithLossCell(megafold, model_cfg)
    if args.run_platform == 'GPU':
        lr = cos_decay_lr(start_step=model_cfg.GPU.start_step, lr_init=0.0,
                          lr_min=model_cfg.GPU.lr_min, lr_max=model_cfg.GPU.lr_max,
                          decay_steps=model_cfg.GPU.lr_decay_steps,
                          warmup_steps=model_cfg.GPU.warmup_steps)
    else:
        lr = model_cfg.ascend.lr
    opt = nn.Adam(params=megafold.trainable_params(), learning_rate=lr, eps=1e-6)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args.loss_scale,
                                 gradient_clip_value=args.gradient_clip)

    train_net.set_train(False)
    step = 0
    np.random.seed(1)
    total_time = 0
    backward_compile_time = 0
    forward_compile_time = 0
    backward_run_time = 0
    forward_run_time = 0

    with open("./train_for_test.pkl", "rb") as f:
        d = pickle.load(f)
    for _ in range(10000):
        max_recycle = 2
        inputs_feats = d["target_feat"], d["msa_feat"], d["msa_mask"], d["seq_mask_batch"], d["aatype_batch"], \
                       d["template_aatype"], d["template_all_atom_masks"], d["template_all_atom_positions"], \
                       d["template_mask"], d["template_pseudo_beta_mask"], d["template_pseudo_beta"], \
                       d["extra_msa"], d["extra_has_deletion"], \
                       d["extra_deletion_value"], d["extra_msa_mask"], d["residx_atom37_to_atom14"], \
                       d["atom37_atom_exists_batch"], d["residue_index_batch"]
        prev_pos, prev_msa_first_row, prev_pair = Tensor(d["prev_pos"]), Tensor(d["prev_msa_first_row"]), \
                                                  Tensor(d["prev_pair"])
        ground_truth = d["pseudo_beta_gt"], d["pseudo_beta_mask_gt"], d["all_atom_mask_gt"], \
                       d["true_msa"], d["bert_mask"], d["residx_atom14_to_atom37"], \
                       d["restype_atom14_bond_lower_bound"], d["restype_atom14_bond_upper_bound"], \
                       d["atomtype_radius"], d["backbone_affine_tensor"], d["backbone_affine_mask"], \
                       d["atom14_gt_positions"], d["atom14_alt_gt_positions"], d["atom14_atom_is_ambiguous"], \
                       d["atom14_gt_exists"], d["atom14_atom_exists"], d["atom14_alt_gt_exists"], \
                       d["all_atom_positions"], d["rigidgroups_gt_frames"], d["rigidgroups_gt_exists"], \
                       d["rigidgroups_alt_gt_frames"], d["torsion_angles_sin_cos_gt"], d["use_clamped_fape"], \
                       d["filter_by_solution"], d["chi_mask"]
        # forward recycle 3 steps
        train_net.add_flags_recursive(train_backward=False)
        train_net.phase = 'train_forward'
        ground_truth = [Tensor(gt) for gt in ground_truth]
        for recycle in range(max_recycle):
            inputs_feat = [Tensor(feat[recycle]) for feat in inputs_feats]
            if recycle == max_recycle - 1:
                t1 = time.time()
                prev_pos, prev_msa_first_row, prev_pair = train_net(*inputs_feat, prev_pos, prev_msa_first_row,
                                                                    prev_pair, *ground_truth)
                t2 = time.time()
        forward_time = t2 - t1
        inputs_feat = [Tensor(feat[max_recycle]) for feat in inputs_feats]
        # forward + backward
        train_net.add_flags_recursive(train_backward=True)
        train_net.phase = 'train_backward'
        loss = train_net(*inputs_feat, prev_pos, prev_msa_first_row, prev_pair, *ground_truth)
        t3 = time.time()
        t4 = time.time()
        backward_time = t4 - t3
        step_time = backward_time + forward_time
        loss_info = f"step is: {step}, total_loss: {loss[0]}, fape_sidechain_loss: {loss[1]}," \
                    f" fape_backbone_loss: {loss[2]}, angle_norm_loss: {loss[3]}, distogram_loss: {loss[4]}," \
                    f" masked_loss: {loss[5]}, plddt_loss: {loss[6]}, step_time: {step_time}, " \
                    f"backward_time: {backward_time}, forward_time: {forward_time}"
        print(loss_info, flush=True)
        if step == 0:
            backward_compile_time = backward_time
            forward_compile_time = forward_time
        if step in range(1, 10000):
            total_time += step_time
            backward_run_time += backward_time
            forward_run_time += forward_time
        step += 1
    print("compile_total_time:{ct:.2f}h, "
          "average_run_time:{art:.2f}s".format(ct=(backward_compile_time + forward_compile_time) / 3600,
                                               art=total_time / (step - 1)))
    print(
        "backward_compile_time:{bct:.2f}h, forward_compile_time:{fct:.2f}h, "
        "average_run_time:{at:.2f}s, backward_run_time:{brt:.2f}s, forward_run_time:{frt:.2f}s".format(
            bct=backward_compile_time / 3600, fct=forward_compile_time / 3600, at=total_time / (step - 1),
            brt=backward_run_time / (step - 1), frt=forward_run_time / (step - 1)))


if __name__ == "__main__":
    if arguments.is_traing:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            max_device_memory="29GB",
                            device_id=arguments.device_id)
        fold_train(arguments)
    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            memory_optimize_level="O1",
                            max_call_depth=6000,
                            device_id=arguments.device_id)
        fold_infer(arguments)
