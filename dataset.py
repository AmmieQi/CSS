#coding = utf-8

from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import _pickle as cPickle
from collections import Counter

import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from random import choice

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property  # 装饰器，创建只读方法，防止属性被修改
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    # 返回处理后的句子，字符串标记
    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-',
                   ' ').replace('.','').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split() # 以空格为分隔符，分割句子
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    # 将字典写入pkl文件中
    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        # 将word2idx，idx2word写入path中
        print('dictionary dumped to %s' % path)

    @classmethod
    # 加载pkl文件
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    # 添加不在word2idx中的单词
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

# 创建entry用于下面的load_dataset函数
def _create_entry(img_idx, question, answer):
    answer.pop('image_id')  # 为例避免重复，去掉答案中的image_id，question_id
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image_idx'       : img_idx,
        'question'    : question['question'],
        'answer'      : answer
    }
    return entry

# 加载相关数据集的问题与答案，并处理成对应的元组
def _load_dataset(dataroot, name, img_id2val, dataset):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if dataset=='cpv2':
      answer_path = os.path.join(dataroot, 'cp-cache', '%s_target.pkl' % name)
      # 答案train_target.pkl文件路径
      name = "train" if name == "train" else "test"
      question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)   #  加载问题的json文件
    elif dataset=='cpv1':
      answer_path = os.path.join(dataroot, 'cp-v1-cache', '%s_target.pkl' % name)
      name = "train" if name == "train" else "test"
      question_path = os.path.join(dataroot, 'vqacp_v1_%s_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)
    elif dataset=='v2':
      answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
      question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
      with open(question_path) as f:
        questions = json.load(f)["questions"]

    with open(answer_path, 'rb') as f:
      answers = cPickle.load(f)  # cPickle和json加载的区别？？？
                                 # train_target.pkl  这种格式也看不出来啊

    # 按照问题的id排序
    questions.sort(key=lambda x: x['question_id'])
    answers.sort(key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers)) # 判断问题和答案的长度是否相等
    entries = []
    # 打包为元组的列表，使问题与答案一一对应
    for question, answer in zip(questions, answers):
        if answer["labels"] is None:
            raise ValueError()
        # 判断问题与答案是否对应
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        img_idx = None
        if img_id2val:
          img_idx = img_id2val[img_id]

        entries.append(_create_entry(img_idx, question, answer))
        """
            entry = {
            'question_id' : question['question_id'],
            'image_id'    : question['image_id'],
            'image_idx'       : img_idx,
            'question'    : question['question'],
            'answer'      : answer
             }
        """
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', dataset='cpv2',
                 use_hdf5=False, cache_image_features=False):
        super(VQAFeatureDataset, self).__init__()
        self.name=name
        if dataset=='cpv2':
            with open('data/train_cpv2_hintscore.json', 'r') as f:
                self.train_hintscore = json.load(f)
            with open('data/test_cpv2_hintscore.json', 'r') as f:
                self.test_hintsocre = json.load(f)
            with open('util/cpv2_type_mask.json', 'r') as f:
                self.type_mask = json.load(f)
            with open('util/cpv2_notype_mask.json', 'r') as f:
                self.notype_mask = json.load(f)

        elif dataset=='cpv1':
            with open('data/train_cpv1_hintscore.json', 'r') as f:
                self.train_hintscore = json.load(f)
            with open('data/test_cpv1_hintscore.json', 'r') as f:
                self.test_hintsocre = json.load(f)
            with open('util/cpv1_type_mask.json', 'r') as f:
                self.type_mask = json.load(f)
            with open('util/cpv1_notype_mask.json', 'r') as f:
                self.notype_mask = json.load(f)
        elif dataset=='v2':
            with open('data/train_v2_hintscore.json', 'r') as f:
                self.train_hintscore = json.load(f)
            with open('data/test_v2_hintscore.json', 'r') as f:
                self.test_hintsocre = json.load(f)
            with open('util/v2_type_mask.json', 'r') as f:
                self.type_mask = json.load(f)
            with open('util/v2_notype_mask.json', 'r') as f:
                self.notype_mask = json.load(f)

        assert name in ['train', 'val']

        if dataset=='cpv2':
            ans2label_path = os.path.join(dataroot, 'cp-cache', 'trainval_ans2label.pkl')   # 文件路径
            label2ans_path = os.path.join(dataroot, 'cp-cache', 'trainval_label2ans.pkl')
        elif dataset=='cpv1':
            ans2label_path = os.path.join(dataroot, 'cp-v1-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cp-v1-cache', 'trainval_label2ans.pkl')
        elif dataset=='v2':
            ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.use_hdf5 = use_hdf5

        # use_hdf5=false
        if use_hdf5:
            h5_path = os.path.join(dataroot, '%s36.hdf5'%name)
            self.hf = h5py.File(h5_path, 'r')
            self.features = self.hf.get('image_features')

            with open("util/%s36_imgid2img.pkl"%name, "rb") as f:
                imgid2idx = cPickle.load(f)
        else:
            imgid2idx = None

        self.entries = _load_dataset(dataroot, name, imgid2idx, dataset=dataset)
        # 读取数据集，需要返回的数据结构为一个列表
        # { “question-id ”:
        #   "answer":{"question_id": ,"image_id": ,"label":["tokens f"
        #   "score": ["score for every label"]}
        #   "image_id":
        #   "image":idx(of_image_id)
        #    "question"
        #}

        # cache_image_features=false
        if cache_image_features:
            image_to_fe = {}
            for entry in tqdm(self.entries, ncols=100, desc="caching-features"):
                img_id = entry["image_id"]
                if img_id not in image_to_fe:
                    if use_hdf5:
                        fe = np.array(self.features[imgid2idx[img_id]])
                    else:
                        fe=torch.load('data/rcnn_feature/'+str(img_id)+'.pth')['image_feature']
                    image_to_fe[img_id]=fe
            self.image_to_fe = image_to_fe
            if use_hdf5:
                self.hf.close()
        else:
            self.image_to_fe = None

        self.tokenize()
        # 获得序列化的问题，
        self.tensorize()

        self.v_dim = 2048

    def tokenize(self, max_length=14):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        添加问题标记
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in tqdm(self.entries, ncols=100, desc="tokenize"):
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                # 填充前面的句子，使句子长度都为最大长度
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                padding_mask=[self.dictionary.padding_idx-1] * (max_length - len(tokens))
                tokens_mask = padding_mask + tokens
                tokens = padding + tokens

            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            entry['q_token_mask']=tokens_mask

    def tensorize(self):
        # Tqdm 是一个快速，可扩展的Python进度条
        for entry in tqdm(self.entries, ncols=100, desc="tensorize"):
            question = torch.from_numpy(np.array(entry['q_token']))
            question_mask = torch.from_numpy(np.array(entry['q_token_mask']))

            entry['q_token'] = question
            entry['q_token_mask']=question_mask

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    # 获取并处理其中的一项
    def __getitem__(self, index):
        entry = self.entries[index]
        if self.image_to_fe is not None:
            features = self.image_to_fe[entry["image_id"]]
        elif self.use_hdf5:
            features = np.array(self.features[entry['image_idx']])
            features = torch.from_numpy(features).view(36, 2048)
        else:
            # 图像特征
            features = torch.load('data/rcnn_feature/' + str(entry["image_id"]) + '.pth')['image_feature']

        q_id=entry['question_id']
        ques = entry['q_token']
        ques_mask=entry['q_token_mask']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.name=='train':
            train_hint=torch.tensor(self.train_hintscore[str(q_id)])
            type_mask=torch.tensor(self.type_mask[str(q_id)])
            notype_mask=torch.tensor(self.notype_mask[str(q_id)])
            if "bias" in entry:
                # 这种文件格式不对，看不出来有什么
                # hint又是什么？
                return features, ques, target,entry["bias"],train_hint,type_mask,notype_mask,ques_mask
            else:
                return features, ques,target, 0,train_hint
        else:
            test_hint=torch.tensor(self.test_hintsocre[str(q_id)])
            if "bias" in entry:
                return features, ques, target, entry["bias"],q_id,test_hint
            else:
                return features, ques, target, 0,q_id,test_hint

    def __len__(self):
        return len(self.entries)

