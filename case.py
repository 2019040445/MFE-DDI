from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from collator import *
from argparse import ArgumentParser
torch.manual_seed(2)
np.random.seed(3)
from dataset import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='Molormer Prediction.')
parser.add_argument('-b', '--batch-size', default=8, type=int,metavar='N')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')

class CrossEntropy(nn.Module):#loss= criterion(score, label)
    def forward(self, input, target):
        scores = torch.sigmoid(input)
        # print(scores)
        target_active = (target == 1).float()  # from -1/1 to 0/1
        # print(target_active)
        loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
        # print(loss_terms.sum())
        # print(len(loss_terms))
        b=loss_terms.sum()/len(loss_terms)
        return b

def test(data_generator, model):
    scores = []
    predictions = [] 

    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
                p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
                label, (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2) in enumerate(tqdm(data_generator)):

        score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(), d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),
                    p_node.cuda(), p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(), p_out_degree.cuda(), p_edge_input.cuda(),
                    adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
                    adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
                    d1.cuda(),d2.cuda(),mask_1.cuda(),mask_2.cuda())
       
        
        # label = Variable(torch.from_numpy(np.array(label-1)).long()).cuda()
        label = torch.from_numpy(np.array(label)).long().cuda()
        print(label.shape)
        print(score.shape)
        criterion = CrossEntropy().cuda()
        loss= criterion(score, label)
        print('Training loss: ' + str(loss.cpu().detach().numpy()))
        # 
        score = torch.sigmoid(score)
        print(score)
        print(label)
        scores.extend(score.tolist())
        # assert False


    # 获取得分最高的前 20 个分数及其索引
    top_20_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:15]

    # 获取对应的测试数据
    # df_test = pd.read_csv('dataset/search_newddis.csv')
    df_test = pd.read_csv('dataset/search_DB00852.csv')
    top_20_data = df_test.iloc[top_20_indices]

    return top_20_data, scores

'''
    预测新的DDIs
'''
def main():
    args = parser.parse_args()

    # model = torch.load('save_model/0.9693686184621796_model.pth')

    model = torch.load('savedmodels_zhang/MultiLevelDDI')
    # model = torch.load('savedmodels/MultiLevelDDI')

    model = model.to(device)


    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': False,
              'collate_fn': collator}

    # df_test = pd.read_csv('dataset/selected_newdrugs.csv')
    df_test = pd.read_csv('dataset/search_DB00852.csv')


    testing_set = Dataset(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        # 调用 test 函数
        top_20_data, scores = test(testing_generator, model)
        print("Top 20 scores and corresponding data:")
        for i, (score, data1) in enumerate(zip(scores, top_20_data.values)):
            print(f"Score {i+1}: {score}, Data: {data1}")

        # test(testing_generator, model)

    torch.cuda.empty_cache()

main()
print("Done!")

# 在 main 函数中调用 test 函数
# def main():
#     # 其他部分的代码...

#     # 调用 test 函数
#     top_20_data, scores = test(testing_generator, model)
#     print("Top 20 scores and corresponding data:")
#     for i, (score, data1) in enumerate(zip(scores, top_20_data.values)):
#         print(f"Score {i+1}: {score}, Data: {data1}")

#     # 其他部分的代码...
