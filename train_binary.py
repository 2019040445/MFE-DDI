import copy
from sklearn import datasets
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from models import Molormer, MultiLevelDDI
from collator import *
torch.manual_seed(2)
np.random.seed(3)
from configs import Model_config
from dataset import Dataset
import os
import argparse
from train_logging import LOG,LOSS_FUNCTIONS
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:2" if use_cuda else "cpu")
# device = 'cpu'



# def test(data_set, model):
#     y_pred = []
#     y_label = []
#     model.eval()
#     loss_accumulate = 0.0
#     count = 0.0

#     for _, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
#             p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
#             label,(adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2) in enumerate(tqdm(data_set)):

#         score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(),
#                       d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),p_node.cuda(),
#                       p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(),
#                       p_out_degree.cuda(), p_edge_input.cuda(),
#                       adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
#                       adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
#                       d1.cuda(), d2.cuda(), mask_1.cuda(), mask_2.cuda()
#                       )

#         label = Variable(torch.from_numpy(np.array(label-1)).long()).cuda()
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(score, label)
#         loss_accumulate += loss
#         count += 1

#         outputs = score.argmax(dim=1).detach().cpu().numpy() + 1
#         label_ids = label.to('cpu').numpy() + 1

#         y_label = y_label + label_ids.flatten().tolist()
#         y_pred = y_pred + outputs.flatten().tolist()

#     loss = loss_accumulate / count

#     accuracy = accuracy_score(y_label, y_pred)
#     micro_precision = precision_score(y_label, y_pred, average='micro')
#     micro_recall = recall_score(y_label, y_pred, average='micro')
#     micro_f1 = f1_score(y_label, y_pred, average='micro')

#     macro_precision = precision_score(y_label, y_pred, average='macro')
#     macro_recall = recall_score(y_label, y_pred, average='macro')
#     macro_f1 = f1_score(y_label, y_pred, average='macro')
#     return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss.item()

common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
common_args_parser.add_argument('--loss', type=str, default='CrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='All', help='roc-auc or MSE or All')
common_args_parser.add_argument('--savemodel', action='store_true', default=True, help='Saves model with highest validation score')
common_args_parser.add_argument('--logging', type=str, default='less')

# main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = None

def main():
    config = Model_config()
    print(config)
    global args
    args = common_args_parser.parse_args()
    args_dict = vars(args)

    loss_history = []
    
    # model = torch.load('./save_model/best_model.pth')
    model = MultiLevelDDI(**config)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)

    params = {'batch_size': config['batch_size'],
              'shuffle': True,
              'num_workers': config['num_workers'],
              'drop_last': True,
              'collate_fn': collator}


    train_data = pd.read_csv('dataset/train.csv')
    val_data = pd.read_csv('dataset/valid.csv')
    test_data = pd.read_csv('dataset/test.csv')

    # start1 = time.time()
    training_set = Dataset(train_data.index.values, train_data.label.values, train_data)
    # end1 = time.time()
    # print('Running time of training_set processing1: %s Seconds'%(end1 - start1))
    # start2 = time.time()
    training_generator = data.DataLoader(training_set, **params)
    # end2 = time.time()
    # print('Running time of training_set processing2: %s Seconds'%(end2 - start2))

    # start1 = time.time()
    validation_set = Dataset(val_data.index.values, val_data.label.values, val_data)
    # end1 = time.time()
    # print('Running time of validation_set processing1: %s Seconds'%(end1 - start1))
    
    start2 = time.time()
    validation_generator = data.DataLoader(validation_set, **params)
    end2 = time.time()
    # print('Running time of validation_set processing2: %s Seconds'%(end2 - start2))

    testing_set = Dataset(test_data.index.values, test_data.label.values, test_data)
    testing_generator = data.DataLoader(testing_set, **params)

    max_auc = 0
    model_max = copy.deepcopy(model)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = LOSS_FUNCTIONS[args.loss].cuda()
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'], eta_min=args.min_lr)


    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(config['epochs']):
        model.train()
        for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
                p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
                label, (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2) in enumerate(training_generator):

            # print(d_node.shape, d_attn_bias.shape, d_spatial_pos.shape, d_in_degree.shape, d_out_degree.shape, d_edge_input.shape,
            #               p_node.shape, p_attn_bias.shape, p_spatial_pos.shape, p_in_degree.shape, p_out_degree.shape, p_edge_input.shape,
            #               adj_1.shape, nd_1.shape, ed_1.shape,
            #               adj_2.shape, nd_2.shape, ed_2.shape,
            #               d1.shape,d2.shape,mask_1.shape,mask_2.shape)
            # assert False
            opt.zero_grad()
            score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(), d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),
                          p_node.cuda(), p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(), p_out_degree.cuda(), p_edge_input.cuda(),
                          adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
                          adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
                          d1.cuda(),d2.cuda(),mask_1.cuda(),mask_2.cuda()) # torch tensor
            # print(score.shape,score)

            label = torch.from_numpy(np.array(label)).long().cuda() # torch tensor
            # loss_fct = torch.nn.CrossEntropyLoss().cuda()
            loss= criterion(score, label)
            # print(label.shape,label)
            # print(loss)
            # assert False
            # loss = loss_fct(score, label)
            loss_history.append(loss)

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            # 用于裁剪梯度，梯度裁剪是一种正则化技术，用于防止在训练深度学习模型时发生梯度爆炸
            opt.step()
            # scheduler.step()

            if (i % 300 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        with torch.set_grad_enabled(False):
            model.eval()
            LOG[args.logging](
                model, training_generator, validation_generator, testing_generator, criterion, epo, args)


            # accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss = test(validation_generator, model)
            # print("[Validation metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
            #     loss, accuracy, macro_precision, macro_recall, macro_f1))
            # if accuracy > max_auc:
            #    # torch.save(model, 'save_model/' + str(accuracy) + '_model.pth')
            #     torch.save(model, 'save_model/best_model.pth')
            #     model_max = copy.deepcopy(model)
            #     max_auc = accuracy
            #     print("*" * 30 + " save best model " + "*" * 30)

        torch.cuda.empty_cache()

    # print('\n--- Go for Testing ---')
    # try:
    #     with torch.set_grad_enabled(False):
    #         accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss  = test(testing_generator, model_max)
    #         print("[Testing metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
    #             loss, accuracy, macro_precision, macro_recall, macro_f1))
    # except:
    #     print('testing failed')
    return model_max, loss_history


'''
nohup python -u train_binary.py > new3_adataset_lr-5_1.log 2>&1 &
new1: AMDE + Molormer
new2: AMDE + Molormer + decoder-->BilinearDecoder  ×
new3: AMDE + Molormer + concat-->乘以权重
'''




if __name__=='__main__':
    main()
