# -*- coding: utf-8 -*-
import argparse
import json
import os

import click
from torch.utils.data import DataLoader

import base_model
import utils
from dataset import Dictionary, VQAFeatureDataset
from train import train
from vqa_debias_loss_functions import *


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=True,
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2", "cpv1"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    # 嵌入损失的种类
    parser.add_argument(
        '--mode', default="updn",
        choices=["updn", "q_debias","v_debias","q_v_debias"],
        help="Kind of ensemble loss to use")

    parser.add_argument(
        '--debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none",'focal'],
        help="Kind of ensemble loss to use")

    # 遮盖前几个单词
    parser.add_argument(
        '--topq', type=int,default=1,
        choices=[1,2,3],
        help="num of words to be masked in questio")

    parser.add_argument(
        '--keep_qtype', default=True,
        help="keep qtype or not")
    # 图像中隐藏几个对象
    parser.add_argument(
        '--topv', type=int,default=1,
        choices=[1,3,5,-1],
        help="num of object bbox to be masked in image")

    # 前几的hint，？？？
    parser.add_argument(
        '--top_hint',type=int, default=9,
        choices=[9,18,27,36],
        help="num of hint")

    # ratio of q_bias and v_bias
    parser.add_argument(
        '--qvp', type=int,default=0,
        choices=[0,1,2,3,4,5,6,7,8,9,10],
        help="ratio of q_bias and v_bias")

    parser.add_argument(
        '--eval_each_epoch', default=True,
        help="Evaluate every epoch, instead of at the end")

    # Arguments from the original model, we leave this default, except we
    # set --epochs to 30 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='logs/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

def get_bias(train_dset,eval_dset):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    # 这里的偏差只是每种答案/问题类型的预期分数
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:   # 从训练集依次获得例子
        ans = ex["answer"]          # 获得答案
        q_type = ans["question_type"]        # 获得答案的问题类型
        question_type_to_count[q_type] += 1     # 每种类型样本数加1（这种问题类型是指比较具体的，不是yesno,other,num等）
        if ans["labels"] is not None:           
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset,eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]


# 添加消除偏见的损失函数方法，使用最初的基线模型去训练
def main():
    args = parse_args()
    dataset=args.dataset     # cpv2
    args.output=os.path.join('logs',args.output)

    # 判断输出的路径是否已经存在
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)
        else:
            os._exit(1)

    # 加载dictionary.pkl
    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    # 建立训练集
    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                   cache_image_features=args.cache_features)
    # 建立测试集
    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)

    get_bias(train_dset, eval_dset)  # 计算并得到偏见

    # Build the model using the original constructor
    # 使用原始构造函数构建模型

    constructor = 'build_%s' % args.model  # args.model=baseline0_newatt
    # 设置模型是 build_baseline0_newatt
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()

    # 选择的单词嵌入
    if dataset=='cpv1':
        model.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
    elif dataset=='cpv2' or dataset=='v2':
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    # Add the loss_fn based our arguments
    # 计算处理偏差的方法是Learned-Mixin +H
    if args.debias == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.debias == "none":
        model.debias_loss_fn = Plain()
    elif args.debias == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.debias == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    elif args.debias=='focal':
        model.debias_loss_fn = Focal()
    else:
        raise RuntimeError(args.mode)

    # 加载问题类型分类（yn，num，other）的json文件
    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    model=model.cuda()
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 加载训练集和验证集
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    # 开始训练
    train(model, train_loader, eval_loader, args, qid2type)

if __name__ == '__main__':
    main()






