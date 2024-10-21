import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
import torch
import numpy as np
import json
import time
import models.rel_model

use_system_io = True
bVis = False


class Framework(object):
    def __init__(self, config):
        self.config = config
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    # define the loss function
    def cal_loss(self, target, predict, mask):
        loss = self.loss_function(predict, target)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def train(self, model_pattern):
        ori_model = model_pattern(self.config)
        ori_model.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr = self.config.learning_rate)

        # path = os.path.join(self.config.checkpoint_dir, "__c_xyarv")
        # if os.path.exists(path):
        #     ori_model.load_state_dict(torch.load(path))

        # path = os.path.join(self.config.checkpoint_dir, "att_final_baseline")
        # if os.path.exists(path):
        #     pretrained_dict = torch.load(path)
        #     model_dict = ori_model.state_dict()
        #     pretrained_dict = {k:v for k, v in pretrained_dict.items() if k and k != "relation_matrix" in model_dict}
        #     model_dict.update(pretrained_dict)
        #     ori_model.load_state_dict(model_dict)

        # whether use multi gpu:
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model


        # check the check_point dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        # training data
        train_data_loader = data_loader.get_loader(self.config, prefix = self.config.train_prefix, num_workers=2)
        # dev data
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True, num_workers=2)

        # other
        model.train()
        global_step = 0
        loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            epoch_start_time = time.time()

            # for batch_ind, data in enumerate(train_data_loader):
            while data is not None:

                pred_triple_matrix = model(data)

                triple_loss = self.cal_loss(data['triple_matrix'], pred_triple_matrix, data['loss_mask'])
                # triple_loss = self.cal_loss(data['negsam_triple_matrix'], pred_triple_matrix, data['loss_mask'])

                optimizer.zero_grad()
                triple_loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += triple_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time

                    if use_system_io:
                        self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.6f}".
                                     format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    else:
                        print("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.6f}".
                                     format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))

                    loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()
            print("total time {}".format(time.time() - epoch_start_time))

            if (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the test function
                precision, recall, f1_score = self.test(test_data_loader, model, current_f1=best_f1_score, output=self.config.result_save_name)

                if use_system_io:
                    self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}'.
                                 format(epoch, time.time() - eval_start_time, f1_score, precision, recall))
                else:
                    print('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}'.
                             format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall

                    if use_system_io:
                        self.logging("saving the model, epoch: {:3d}, precision: {:4.3f}, recall: {:4.3f}, best f1: {:4.3f}".
                                     format(best_epoch, best_precision, best_recall, best_f1_score))
                        # save the best model
                        # path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                        path = os.path.join(self.config.checkpoint_dir, "__" + self.config.config)
                        if not self.config.debug:
                            torch.save(ori_model.state_dict(), path)
                    else:
                        print("saving the model, epoch: {:3d}, precision: {:4.3f}, recall: {:4.3f}, best f1: {:4.3f}".
                                 format(best_epoch, best_precision, best_recall, best_f1_score))


                model.train()

            # manually release the unused cache
            torch.cuda.empty_cache()

        if use_system_io:
            self.logging("finish training")
            self.logging("best epoch: {:3d}, precision: {:4.3f}, recall: {:4.3}, best f1: {:4.3f}, total time: {:5.2f}s".
                         format(best_epoch, best_precision, best_recall, best_f1_score, time.time() - init_time))
        else:
            print("finish training")
            print("best epoch: {:3d}, precision: {:4.3f}, recall: {:4.3}, best f1: {:4.3f}, total time: {:5.2f}s".
                  format(best_epoch, best_precision, best_recall, best_f1_score, time.time() - init_time))


    def test(self, test_data_loader, model, current_f1, output=True):
        
        orders = ['subject', 'relation', 'object']

        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        id2rel = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[0]
        id2tag, tag2id = json.load(open('data/tag2id.json'))
        correct_num, predict_num, gold_num = 0, 0, 0

        results = []
        test_num = 0

        s_time = time.time()

        if bVis:
            model_weights = []
            conv_layers = []

            model_children = list(model.children())
            # print(model_children)
            count = 0

            for i in range(len(model_children)):
                if type(model_children[i]) == models.rel_model.PdcAttBottleneck:
                    # print(list(model_children[i].children()))
                    model_subChild = list(model_children[i].children())
                    for j in range(len(model_subChild)):
                        if type(model_subChild[j]) == models.rel_model.Conv2d:
                            count += 1
                            model_weights.append(model_subChild[j].weight)
                            conv_layers.append(model_subChild[j])
                        elif type(model_subChild[j]) == models.rel_model.PDCBlock:
                            model_subsubChild = list(model_subChild[j].children())
                            for k in range(len(model_subsubChild)):
                                if type(model_subsubChild[k]) == nn.Conv2d:
                                    count += 1
                                    model_weights.append(model_subsubChild[k].weight)
                                    conv_layers.append(model_subsubChild[k])

            print(count)
            print(conv_layers)

            forward_outputs = []
            backward_outputs = []
            names = []

            def forward_hook(module, input, output):
                names.append(module.__class__.__name__)
                # if module.__class__.__name__ == "ChannelAttention" or module.__class__.__name__ == "SpatialAttention":
                #     print(input.shape)
                #     print(output.shape)
                #     res = output * input
                #     forward_outputs.append(res)
                #     print(res.shape)
                # else :
                if module.__class__.__name__ == "PdcAttBottleneck":
                    output = output.permute(0, 3, 1, 2)
                forward_outputs.append(output)
                print("forward_hook")
                print(output.shape)

            def backward_hook(module, grad_input, grad_output):

                if module.__class__.__name__ == "PdcAttBottleneck":
                    grad_output = grad_output.permute(0, 3, 1, 2)
                backward_outputs.append(grad_output)
                print(grad_output.shape)
                print("backward_hook")




            handle_1 = model.pdcAtt.init_block.register_forward_hook(forward_hook)
            handle_2 = model.pdcAtt.block_1.register_forward_hook(forward_hook)
            handle_3 = model.pdcAtt.block_2.register_forward_hook(forward_hook)
            handle_4 = model.pdcAtt.block_3.register_forward_hook(forward_hook)
            handle_5 = model.pdcAtt.register_forward_hook(forward_hook)
            # handle_5 = model.pdcAtt.ca.register_forward_hook(forward_hook)
            # handle_6 = model.pdcAtt.sa.register_forward_hook(forward_hook)

            backhandle_1 = model.pdcAtt.init_block.register_full_backward_hook(backward_hook)
            backhandle_2 = model.pdcAtt.block_1.register_full_backward_hook(backward_hook)
            backhandle_3 = model.pdcAtt.block_2.register_full_backward_hook(backward_hook)
            backhandle_4 = model.pdcAtt.block_3.register_full_backward_hook(backward_hook)
            backhandle_5 = model.pdcAtt.register_full_backward_hook(backward_hook)
            # handle_5 = model.pdcAtt.ca.register_forward_hook(backward_hook)
            # handle_6 = model.pdcAtt.sa.register_forward_hook(backward_hook)

            while data is not None:
                with torch.no_grad():
                    tokens = data['tokens'][0]
                    pred_triple_matrix = model(data, train=False).cpu()[0]
                    rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
                    i = 0
                    with open('normConvWeight.txt', 'a') as f:
                        f.write(str(tokens) + "\n")
                        f.write(' '.join(tokens[1:-1]).replace(' [unused1]', '').replace(' ##', '') + "\n")
                        f.write("sentence length:" + str(seq_lens) + "\n")
                        for feature_map in forward_outputs:
                            # if i < 4:
                            feature_map = feature_map.squeeze(0) # 768 36 36
                            gray_scale = torch.sum(feature_map, 0) # 36 36
                            gray_scale = gray_scale / feature_map.shape[0] #36 36
                            totalSum = torch.sum(gray_scale, (0, 1))
                            # totalSum = torch.sum(feature_map, (0, 1))
                            gray_scale = gray_scale / totalSum
                            # gray_scale = torch.exp(gray_scale)
                            # gray_scale = gray_scale / gray_scale.sum()
                            print(gray_scale.shape)
                            f.write(names[i] + ":" + str(gray_scale.data.cpu().numpy().tolist()) + ";\n")
                            i += 1
                            # elif i == 4:
                            #     print(feature_map.shape)
                            #     feature_map = feature_map.squeeze(0).squeeze(-1).squeeze(-1)
                            #     print(feature_map.shape)
                            # else:
                            #     print(feature_map.shape)
                            #     feature_map = feature_map.squeeze(0).squeeze(1)
                            #     print(feature_map.shape)
                    forward_outputs.clear()
                    data = test_data_prefetcher.next()

            # with torch.no_grad():
            # model.requires_grad_(True)

                # print("----------------------")
                # i = 0

                # f.write(str(tokens) + "\n")
                # f.write(' '.join(tokens[1:-1]).replace(' [unused1]', '').replace(' ##', '') + "\n")
                # f.write("sentence length:" + str(seq_lens) + "\n")
                #
                # centerIndex = 0
                # surroundSize = 5
                # for index, x in np.ndenumerate(tokens):
                #     if x == 'Pradesh':
                #         centerIndex = index[0]
                #
                # print(centerIndex)
                #
                # for feature_map in forward_outputs:
                #     # if i < 4:
                #     # [batch_size , dim_size, seq_size, seq_size]
                #     feature_map = feature_map.squeeze(0)
                #     # [dim_size, seq_size, seq_size]
                #
                #     print(feature_map.shape)
                #     bFirst = True
                #     for k in range(0, surroundSize*2):
                #         for j in range(0, surroundSize*2):
                #             if bFirst:
                #                 gray_scale = feature_map[:, centerIndex - surroundSize,
                #                              centerIndex - surroundSize].unsqueeze(0)
                #                 bFirst = False
                #             else:
                #                 gray_scale = torch.cat((gray_scale, feature_map[:, centerIndex - surroundSize + k,
                #                                                 centerIndex - surroundSize + j].unsqueeze(0)), 0)
                #
                #     print(gray_scale.shape)
                #
                #     gray_scale = gray_scale.t()
                #     totalSum = torch.sum(gray_scale, 0)
                #     gray_scale = gray_scale / totalSum
                #     gray_scale = gray_scale.t()
                #
                #     print(gray_scale.shape)
                #
                #     f.write(names[i] + "_word:" + str(gray_scale.data.cpu().numpy().tolist()) + ";\n")
                #     i += 1
                #
                # f.close()



            handle_1.remove()
            handle_2.remove()
            handle_3.remove()
            handle_4.remove()
            handle_5.remove()
            # handle_6.remove()
            # handle_6.remove()

            backhandle_1.remove()
            backhandle_2.remove()
            backhandle_3.remove()
            backhandle_4.remove()
            backhandle_5.remove()

        if not bVis:
            while data is not None:
            # for i, data in enumerate(test_data_loader):
                with torch.no_grad():
                    print('\r Testing step {} / {}, Please Waiting!'.format(test_num, test_data_loader.dataset.__len__()), end="")

                    token_ids = data['token_ids']
                    tokens = data['tokens'][0]
                    mask = data['mask']
                    # pred_triple_matrix: [1, rel_num, seq_len, seq_len]
                    # pred_triple_matrix: [1, tag_size, rel_num, seq_len, seq_len]
                    pred_triple_matrix = model(data, train = False).cpu()[0]
                    # pred_triple_matrix = pred_triple_matrix.permute(0,2,3,4,1)
                    # pred_triple_matrix_softmax = F.softmax(pred_triple_matrix, dim=1)
                    # print(pred_triple_matrix.shape)
                    # pred_triple_matrix_max , index= torch.max(pred_triple_matrix_softmax, dim=0)
                    # pred_triple_matrix_argmax = pred_triple_matrix_softmax.argmax(dim = 0)
                    # print(pred_triple_matrix.shape)
                    rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
                    # relations_arg, heads_arg, tails_arg = np.where(pred_triple_matrix_argmax > 0)
                    relations, heads, tails = np.where(pred_triple_matrix > 0)

                    triple_list = []

                    pair_numbers = len(relations)
                    # print(pair_numbers)

                    # if pair_numbers > 0 and seq_lens <= 40:
                    #     with open('ori_SOO_triples.txt', 'a') as f:
                    #         f.write(str(tokens) + "\n")
                    #         f.write(' '.join(tokens[1:-1]).replace(' [unused1]', '').replace(' ##', '') + "\n")
                    #         f.write("sentence length:" + str(seq_lens) + "\n")
                    #         r_set = set()
                    #         for i in range(pair_numbers):
                    #             r_index = relations[i]
                    #             r_set.add(r_index)
                    #         for r_index in r_set:
                    #             rel = id2rel[str(int(r_index))]
                    #             f.write(rel + ":" + str(pred_triple_matrix[r_index].tolist()) + ";\n")
                    #         f.close()

                    if pair_numbers > 0:
                        # print('current sentence contains {} triple_pairs'.format(pair_numbers))
                        for i in range(pair_numbers):
                            r_index = relations[i]
                            h_start_index = heads[i]
                            t_start_index = tails[i]
                            # 如果当前第一个标签为B-B
                            if pred_triple_matrix[r_index][h_start_index][t_start_index] == tag2id['B-B'] and i+1 < pair_numbers:
                                # 如果下一个标签为B-E
                                t_end_index = tails[i+1]
                                if pred_triple_matrix[r_index][h_start_index][t_end_index] == tag2id['B-E']:
                                    # 那么就向下找
                                    for h_end_index in range(h_start_index, seq_lens):
                                        # 向下找到了结尾位置
                                        if pred_triple_matrix[r_index][h_end_index][t_end_index] == tag2id['E-E']:

                                            sub_head, sub_tail = h_start_index, h_end_index
                                            obj_head, obj_tail = t_start_index, t_end_index

                                            # sub
                                            sub = tokens[sub_head : sub_tail+1]
                                            sub = ''.join([i.lstrip("##") for i in sub])
                                            sub = ' '.join(sub.split('[unused1]')).strip()
                                            # obj
                                            obj = tokens[obj_head : obj_tail+1]
                                            obj = ''.join([i.lstrip("##") for i in obj])
                                            obj = ' '.join(obj.split('[unused1]')).strip()

                                            rel = id2rel[str(int(r_index))]
                                            if len(sub) > 0 and len(obj) > 0:
                                                triple_list.append((sub, rel, obj))
                                            break


                    triple_set = set()
                    r_set = set()
                    so_set = set()
                    gold_triple_set = set()
                    gold_r_set = set()
                    gold_so_set = set()

                    def partial_match(pred_set, gold_set):
                        pred = {(i[0].split(' ')[-1] if len(i[0].split(' ')) > 0 else i[0], i[1],
                                 i[2].split(' ')[-1] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
                        gold = {(i[0].split(' ')[-1] if len(i[0].split(' ')) > 0 else i[0], i[1],
                                 i[2].split(' ')[-1] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
                        return pred, gold

                    # triple_list, gold_triple = partial_match(triple_list, data['triples'][0])

                    for s, r, o in triple_list:
                        triple_set.add((s, r, o))
                        r_set.add((r))
                        so_set.add((s, o))

                    pred_triple_list = list(triple_set)
                    pred_r_list = list(r_set)
                    pred_so_list = list(so_set)

                    gold_triple = data['triples'][0]

                    def to_tup(triple_list):
                        ret = []
                        for triple in triple_list:
                            ret.append(tuple(triple))
                        return ret

                    pred_triples = set(pred_triple_list)
                    pred_r = set(pred_r_list)
                    pred_so = set(pred_so_list)

                    gold_triples_list = to_tup(gold_triple)
                    for s, r, o in gold_triples_list:
                        gold_triple_set.add((s, r, o))
                        gold_r_set.add((r))
                        gold_so_set.add((s, o))

                    gold_triples = set(gold_triples_list)
                    gold_r = set(list(gold_r_set))
                    gold_so = set(list(gold_so_set))

                    pred_triples = pred_triples
                    gold_triples = gold_triples

                    correct_num += len(pred_triples & gold_triples)
                    predict_num += len(pred_triples)
                    gold_num += len(gold_triples)

                    if output:
                        results.append({
                            'text': ' '.join(tokens[1:-1]).replace(' [unused1]', '').replace(' ##', ''),
                            'triple_list_gold': [
                                dict(zip(orders, triple)) for triple in gold_triples
                            ],
                            'triple_list_pred': [
                                dict(zip(orders, triple)) for triple in pred_triples
                            ],
                            'new': [
                                dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                            ],
                            'lack': [
                                dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                            ]
                        })

                    data = test_data_prefetcher.next()

                test_num += 1

        print('\n' + self.config.model_save_name )
        print("\n correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        if use_system_io:
            if output and f1_score > current_f1:
                if not os.path.exists(self.config.result_dir):
                    os.mkdir(self.config.result_dir)

                path = os.path.join(self.config.result_dir, self.config.result_save_name)

                fw = open(path, 'w')

                for line in results:
                    fw.write(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
                fw.close()

        return precision, recall, f1_score

    def testall(self, model_pattern, model_name):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, model_name)
        model.load_state_dict(torch.load(path))

        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        precision, recall, f1_score = self.test(test_data_loader, model, current_f1=0, output=True)
        print("f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}".format(f1_score, precision, recall))

