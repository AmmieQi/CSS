# -*- coding: utf-8 -*-
import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy

# 计算遮挡部分的贡献分数
def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader,args,qid2type):
    dataset=args.dataset
    num_epochs=args.epochs  # 30
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    if mode=='q_debias':
        topq=args.topq   # 遮住单词的个数
        keep_qtype=args.keep_qtype   #
    elif mode=='v_debias':
        topv=args.topv   # 遮住区域的个数
        top_hint=args.top_hint     # 选择前几的区域作为正确的区域及
    elif mode=='q_v_debias':
        topv=args.topv
        top_hint=args.top_hint
        topq=args.topq
        keep_qtype=args.keep_qtype
        qvp=args.qvp

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        print(len(train_loader.dataset))
        t = time.time()
        # tqdm（）是进度条
        for i, (v, q, a, b, hintscore,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            #########################################
            v = Variable(v).cuda().requires_grad_()  # 512,36,2048
            q = Variable(q).cuda()  # 512,14
            q_mask=Variable(q_mask).cuda()  # 512,14
            a = Variable(a).cuda()  # a是答案标签 512,2274
            b = Variable(b).cuda()  # b是偏差 512,2274
            hintscore = Variable(hintscore).cuda()  # 512,36
            type_mask=Variable(type_mask).float().cuda()  # 512,14
            notype_mask=Variable(notype_mask).float().cuda()  # 512,14

            print('v :', v.size())
            print('q :', q.size())
            print("q_mask:", q_mask.size())
            print('a :', a.size())
            print('b :', b.size())
            print('hintscore :', hintscore.size())
            print('type_mask: ', type_mask.size())
            print('notype_mask: ', notype_mask.size())
            #########################################

            #----------------------------基础的updn-----------------------------------------------------------------------
            if mode=='updn':
                pred, loss,_ = model(v, q, a, b, None) # 
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score
            # --------------------------仅问题Q-CSS-----------------------------------------------------------------------
            elif mode=='q_debias':
                if keep_qtype==True:
                    sen_mask=type_mask
                else:
                    sen_mask=notype_mask
                ## first train
                pred, loss, word_emb = model(v, q, a, b, None)   # VQA（I，Q，a，True）

                word_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), word_emb, create_graph=True)[0]

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

                ## second train

                word_grad_cam = word_grad.sum(2)
                # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
                word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
                word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask

                w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :topq]

                q2 = copy.deepcopy(q_mask)

                m1 = copy.deepcopy(sen_mask)  ##[0,0,0...0,1,1,1,1]
                m1.scatter_(1, w_ind, 0)  ##[0,0,0...0,0,1,1,0]
                m2 = 1 - m1  ##[1,1,1...1,1,0,0,1]
                if dataset=='cpv1':
                    m3=m1*18330
                else:
                    m3 = m1 * 18455  ##[0,0,0...0,0,18455,18455,0]
                q2 = q2 * m2.long() + m3.long()  # q2是Q+

                pred, _, _ = model(v, q2, None, b, None)  # VQA（I，Q+，a，True）

                pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]  # torch.argsort在一维度按从小到大排序，输出的是对应的索引号而不是排序结果
                false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                false_ans.scatter_(1, pred_ind, 0)  # 按某种索引修改放置元素
                a2 = a * false_ans
                q3 = copy.deepcopy(q)
                if dataset=='cpv1':
                    q3.scatter_(1, w_ind, 18330)
                else:
                    q3.scatter_(1, w_ind, 18455)

                ## third train

                # q3是Q-，a2是a-
                pred, loss, _ = model(v, q3, a2, b, None)    # VQA（I，Q-，a-，False）

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
            # -----------------------仅图像 V-CSS-------------------------------------------------------------------
            elif mode=='v_debias':
                ## first train
                pred, loss, _ = model(v, q, a, b, None)     # VQA（I，Q，a，True）
                visual_grad=torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

                ## second train
                v_mask = torch.zeros(v.shape[0], 36).cuda()
                visual_grad_cam = visual_grad.sum(2)
                # 给分数排序
                hint_sort, hint_ind = hintscore.sort(1, descending=True)
                # 选择一小部分作为正确的区域
                v_ind = hint_ind[:, :top_hint]
                v_grad = visual_grad_cam.gather(1, v_ind)

                # 计算贡献
                # topv不等于-1，不看if，直接看else
                if topv==-1:
                    v_grad_score,v_grad_ind=v_grad.sort(1,descending=True)
                    v_grad_score=nn.functional.softmax(v_grad_score*10,dim=1)
                    v_grad_sum=torch.cumsum(v_grad_score,dim=1)
                    v_grad_mask=(v_grad_sum<=0.65).long()
                    v_grad_mask[:,0] = 1
                    v_mask_ind=v_grad_mask*v_ind
                    for x in range(a.shape[0]):
                        num=len(torch.nonzero(v_grad_mask[x]))
                        v_mask[x].scatter_(0,v_mask_ind[x,:num],1)
                else:
                    v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]# 在这一步就得到了关键区域
                    # 或者我们直接找下面的v_mask？
                    v_star = v_ind.gather(1, v_grad_ind)
                    v_mask.scatter_(1, v_star, 1)   # 此时v_mask代表I+


                pred, _, _ = model(v, q, None, b, v_mask)    # VQA（I+，Q，a，True）
                                                            # 此时v_mask代表I+
                pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                false_ans.scatter_(1, pred_ind, 0)
                a2 = a * false_ans
                # v_mask(512行，36列)
                v_mask = 1 - v_mask  # 此时v_mask代表I-

                # 第三次训练
                pred, loss, _ = model(v, q, a2, b, v_mask)    # VQA（I-，Q，a—，False）

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
            # ---------------------------CSS-----------------------------------------------------------------------
            elif mode=='q_v_debias':
                random_num = random.randint(1, 10)
                if keep_qtype == True:
                    sen_mask = type_mask
                else:
                    sen_mask = notype_mask
                if random_num<=qvp:
                    # ---------------------先执行Q-CSS----------------------------------------------------------------
                    ## first train
                    pred, loss, word_emb = model(v, q, a, b, None)   # # VQA（I，Q-，a-，False）
                    word_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), word_emb, create_graph=True)[0]

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
                    batch_score = compute_score_with_logits(pred, a.data).sum()
                    train_score += batch_score

                    ## second train

                    word_grad_cam = word_grad.sum(2)
                    # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
                    word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
                    word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask
                    w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :topq]

                    q2 = copy.deepcopy(q_mask)

                    m1 = copy.deepcopy(sen_mask)  ##[0,0,0...0,1,1,1,1]
                    m1.scatter_(1, w_ind, 0)  ##[0,0,0...0,0,1,1,0]
                    m2 = 1 - m1  ##[1,1,1...1,1,0,0,1]
                    if dataset=='cpv1':
                        m3=m1*18330
                    else:
                        m3 = m1 * 18455  ##[0,0,0...0,0,18455,18455,0]
                    q2 = q2 * m2.long() + m3.long()

                    pred, _, _ = model(v, q2, None, b, None)     # VQA（I，Q+，a+，true）

                    pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                    false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a * false_ans
                    q3 = copy.deepcopy(q)
                    if dataset=='cpv1':
                        q3.scatter_(1, w_ind, 18330)
                    else:
                        q3.scatter_(1, w_ind, 18455)

                    ## third train

                    pred, loss, _ = model(v, q3, a2, b, None)   # VQA（I，Q-，a-，False）

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)

                # --------------------再执行V-CSS-------------------------------------------------------------------
                else:
                    ## first train
                    pred, loss, _ = model(v, q, a, b, None)
                    visual_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
                    batch_score = compute_score_with_logits(pred, a.data).sum()
                    train_score += batch_score

                    #  # second train
                    print("-------------------second train-----------------------------")
                    # v(512, 36, 2048)
                    v_mask = torch.zeros(v.shape[0], 36).cuda()
                    # v_mask(512, 36)
                    print("v_mask: ", v_mask.size())
                    visual_grad_cam = visual_grad.sum(2)
                    # visual_grad(512, 36, 2048)
                    # visual_grad_cam(512,36)
                    hint_sort, hint_ind = hintscore.sort(1, descending=True)
                    # hintscore (512, 36)
                    # hint_ind (512, 36)
                    # hint_sort(512, 36)
                    print("hintscore: ", hintscore.size())
                    v_ind = hint_ind[:, :top_hint]
                    # v_ind (512, 9)
                    print("v_ind: ", v_ind.size())
                    v_grad = visual_grad_cam.gather(1, v_ind)


                    if topv == -1:
                        # 实际上topv == 1才能得到论文里面的结果
                        # 此if语句里面大都是（512， 9）
                        # v_grad_score(512, 9)
                        # v_grad_ind(512, 9)
                        v_grad_score, v_grad_ind = v_grad.sort(1, descending=True)
                        #
                        v_grad_score = nn.functional.softmax(v_grad_score * 10, dim=1)
                        v_grad_sum = torch.cumsum(v_grad_score, dim=1)
                        v_grad_mask = (v_grad_sum <= 0.65).long()
                        v_grad_mask[:,0] = 1
                        # v_grad_mask[:,0] 有512个一
                        v_mask_ind = v_grad_mask * v_ind
                        # v_mask_ind (512, 9)
                        for x in range(a.shape[0]):
                            num = len(torch.nonzero(v_grad_mask[x]))
                            v_mask[x].scatter_(0, v_mask_ind[x,:num], 1)
                    else:
                        print("v_grad: ", v_grad.size())
                        v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]
                        # v_grad_ind(512, 1)
                        print("v_grad_ind: ", v_grad_ind.size())
                        v_star = v_ind.gather(1, v_grad_ind)
                        # v_star(512, 1)
                        print("v_star: ", v_star.size())
                        v_mask.scatter_(1, v_star, 1)
                        # v_mask(512, 36)
                        print("v_mask: ", v_mask.size())

                    pred, _, _ = model(v, q, None, b, v_mask)  # 第二次训练
                    # pred(512, 2274)
                    print("pred: ", pred.size())
                    pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                    # pred_ind (512, 5)
                    print("pred_ind: ", pred_ind.size())
                    false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                    # false_ans(512, 2274)
                    print("false_ans: ", false_ans.size())
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a * false_ans
                    print("-------------------second train-----------------------------")
                    v_mask = 1 - v_mask  # v_mask(512, 36)

                    # v(512, 36, 2048)
                    # q(512, 14)
                    # a2(512, 2247)
                    # b(512, 2247)
                    # v_mask(512, 36)
                    pred, loss, _ = model(v, q, a2, b, v_mask)  # 第三次训练

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
        np_v_mask = v_mask.numpy()
        print(np_v_mask)
        if mode=='updn':
            total_loss /= len(train_loader.dataset)
        else:
            total_loss /= len(train_loader.dataset) * 2
        train_score = 100 * train_score / len(train_loader.dataset)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, b, qids, _ in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred, _,_ = model(v, q, None, None, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')


    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
