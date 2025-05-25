
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import random

import numpy as np
import os
import pandas as pd
import h5py
import torch
import difflib

def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()
def make_graph(wordlist,words):
    if words == None:
        return
    wordmake = []
    for i in wordlist:
        lemma = words[f'{int(i)}'].split("/")[1]
        wordmake.append(lemma)
    wordmake = np.stack(wordmake)
    n = len(wordmake)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            dist = similar(wordmake[i], wordmake[j])
            matrix[i][j] = dist
            matrix[j][i] = dist  # 对称性

    return np.stack(matrix)

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    # df = np.stack(df, axis=0)
    x, y = [], []
    random.shuffle(df)
    i= 0
    for dict in df:
        if i%100 == 0:
            print(f"{i}/{len(df)}")
        i = i + 1
        data1 =[]

        t = dict.shape[0]
        min_t = abs(min(x_offsets))
        max_t = abs(t - abs(max(y_offsets))+1)  # Exclusive
        for t in range(min_t,max_t):
            x_t = dict[t+x_offsets]
            x_t[-1][1] = 0
            y_t = dict[t][1]
            x.append(x_t)
            y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    y = y.reshape(-1,1)

    # x.reshape(x.shape[1],x.shape[0],x.shape[2],x.shape[3])
    return x, y

def word_relation(args):
    wordfile = '../data/duolingguo/all_data/word.npy'
    with np.load('all_wordsrelationship.npz',allow_pickle=True) as data:
        # 假设 'data_array' 是文件中的一个数组
        words = data['arr_0']
    wordmake = []
    for i in words:
        lemma = words[f'{int(i)}'].split("/")[1]
        wordmake.append(lemma)
    wordmake = np.stack(wordmake)
    n = len(wordmake)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        if i % 100 == 0:
            print(f"{i}/{n}")
        for j in range(i, n):
            dist = similar(wordmake[i], wordmake[j])
            matrix[i][j] = dist
            matrix[j][i] = dist  # 对称性
    matrix=np.stack(matrix)
    np.savez("all_wordsrelationship.npz",matrix)

def generate_train_val_test(args):
    # wordfile = '../data/duolingguo/all_data/word.npy'
    # with np.load('../data/duolingguo/all_data/wordsreverse.npz',allow_pickle=True) as data:
    #     # 假设 'data_array' 是文件中的一个数组
    #     words = data['arr_0'].item()
    filename = args.kt_df_filename + '/en_to_es/en_to_es.txt'
    df = []
    file = open(filename,'r')
    content = file.read()
    lines = content.splitlines()
    i=0
    delte=[]
    idx=[]
    p =[]
    detle_mean = 0
    detle_std = 0
    idx_mean=0
    idx_std=0
    for line in lines:
        if(i%12==0):
            i=i+1
            data=[]
            continue

        data1 = np.array(line.split(','))
        # if(i%11==1):
        #     p.append(data1.astype('float32'))
        # if(i%11==2):
        #     idx.append(data1.astype('float32'))
        # if(i%11==3):
        #     delte.append(data1.astype('float32'))
        data.append(data1.astype('float32'))
        if (i % 12 == 11):
            data = np.stack(data,axis=1)
            df.append(data)
        i=i+1

    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-15, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 2, 1))

    x, y= generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False
        # words = words
    )
    # y[y<1] = 0
    # z = x[:,:,1]
    # z[z<1] = 0
    # x[:,:,1]=z
    # x[:,:,0]=(x[:,:,0] - p_mean) / p_std
    # y = (y - p_mean) / p_std
    # x[:,:,1]=(x[:,:,1] - idx_mean) / idx_std
    # x[:, :, 2] = (x[:, :, 2] - detle_mean) / detle_std

    # rand_set
    # x0 = x[:,:,0]
    # x4 = x[:, :, 4]
    # x = np.array(torch.randn(x.shape)*3+6)
    # x[x<0] = 0.001
    # x[:,:,0] = x0
    # x[:, :, 4] = x4


    # y = np.array(torch.randint(1, 11, y.shape))
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples *0.7)
    num_val = num_samples - num_test - num_train

    # train
    # graph_train = data1[:num_train]
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    cat = "graph_train"
    # print(cat,"shape",graph_train.shape)
    # np.savez_compressed(
    #     os.path.join(args.output_dir, "%s.npz" % cat),
    #     x=graph_train,
    # )
    for cat in ["train", "val", "test"]:
        _x, _y= locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
        )





def duolingguo(args):
    filelist = os.listdir(args.kt_df_filename)
    words = {}
    wordsreverse = {}
    users={}
    cnts = 0
    cntspeople=0
    flag = 0
    for filename in filelist:
        if flag == 0:
            flag +=1
        else:
            break
        df = pd.read_csv(args.kt_df_filename + '/' + filename, sep=',')
        # df = pd.read_csv('../'+filename, sep=',')
        dict = {}
        last_time = {}
        word_last_time = {}
        for index, row in df.iterrows():
            if dict.get(f"{row['user_id']}") == None:
                dict[f"{row['user_id']}"] = {'user_id':[],'p_recall': [], 'timestamp':[], 'delta': [], 'lexeme_id': [], 'history_seen': [],'history_correct': [],
                                             'session_seen': [], 'session_correct': [],'delta_t':[],'delta_s':[],'wordsize':[]}
                delta_t = 0
            else:
                delta_t = row['timestamp'] - last_time[f"{row['user_id']}"]
            dict[f"{row['user_id']}"]['timestamp'].append(float(row['timestamp']))

            p = int(int(row['p_recall']) == 1 )
            dict[f"{row['user_id']}"]['p_recall'].append(p)


            if users.get(f"{row['user_id']}") == None:
                users[f"{row['user_id']}"] = cntspeople
                cntspeople = cntspeople +1
            if words.get(f"{row['lexeme_id']}") == None:
                words[f"{row['lexeme_id']}"] = cnts
                wordsreverse[f"{cnts}"] = row['lexeme_string']
                cnts = cnts +1
            dict[f"{row['user_id']}"]['user_id'].append(users[f"{row['user_id']}"])
            dict[f"{row['user_id']}"]['lexeme_id'].append(words[f"{row['lexeme_id']}"])
            dict[f"{row['user_id']}"]['history_seen'].append(np.log(int(row['history_seen'])))
            dict[f"{row['user_id']}"]['history_correct'].append(np.log(int(row['history_correct'])))
            dict[f"{row['user_id']}"]['session_seen'].append(row['session_seen'])
            last_time[f"{row['user_id']}"] = float(row['timestamp'])
            word, remainder = row['lexeme_string'].split("/", 1)
            if word == "<*sf>":
                word = remainder.split("<", 1)[0]

            lemma = row['lexeme_string'].split("/")[1]
            timestamp = float(row['timestamp'])
            if str(row['user_id']) + word in word_last_time:
                delta_s = timestamp - word_last_time[str(row['user_id']) + word]
            else:
                delta_s = 0
            delta_t = round(delta_t / 86400, 3)
            delta_s = round(delta_s / 86400, 3)
            delta = round(int(row['delta'])/ 86400, 3)
            wordsize = len(word)
            word_last_time[f"{row['user_id']}"+ word] = float(row['timestamp'])
            dict[f"{row['user_id']}"]['delta_t'].append(delta_t)
            dict[f"{row['user_id']}"]['delta_s'].append(delta_s)
            dict[f"{row['user_id']}"]['wordsize'].append(wordsize)
            dict[f"{row['user_id']}"]['delta'].append(delta)

        print("begin" + f'{filename[:-4]}')
        file = open(f"../{filename[:-4]}.txt", "w")
        for id, id_dict in dict.items():
            file.write(str(id) + '\n')
            for _, list in id_dict.items():
                num = len(list)
                cnt = 0;
                for l in list:
                    cnt = cnt + 1;
                    l = str(l)
                    file.write(str(l))
                    if cnt == num:
                        file.write('\n')
                    else:
                        file.write(',')
        file.close()
        print("done!" + f'{filename}')
    np.savez("../data/duolingguo/wordsreverse.npz", wordsreverse)
    np.savez("../data/duolingguo/word.npz", words)
    np.savez("../data/duolingguo/user.npz", users)

def hello(args):
    folder_path = args.kt_df_filename
    file_list = os.listdir(folder_path)

    with open(args.kt_df_filename+'/total.txt', 'w') as merged_file:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path) as file:
                line = file.read()
                merged_file.write(line)

def do_my_data(args):

    file = open(args.kt_df_filename+'/opensource_dataset_forgetting_curve.tsv', "r")

    # 步骤2：读取文件内容
    content = file.read()

    # 步骤3：逐行处理文件内容
    lines = content.splitlines()

    i =0;
    data = []
    for line in lines:
        line = line.split()
        if(i%3==2):

            data.append(line)
        i +=1
        # 在此处进行每行的处理操作
    with open(data.txt,'w') as file:
        file.write(data)
    # 步骤4：关闭文件
    file.close()

# def duolingopro(args):
#     filename = '../data/learning_traces.13m/learning_traces.13m.csv'
#     df = pd.read_csv( filename, sep=',')

def main(args):
    print("Generating training data")
    # hello(args)
    # do_my_data(args)
    # duolingguo(args)
    # momo(args)
    generate_train_val_test(args)
    # word_relation(args)

def read_large_file(file_path):
    with open(file_path,  encoding='utf-8') as file:
        for line in file:
            yield line.strip()
def momo(args):

    file_path = args.kt_df_filename + '/opensource_dataset.tsv'
    file_generator = read_large_file(file_path)
    j = 0
    dict = {}
    user = {}
    word = {}
    user_word={}
    dfall=[]
    for line in file_generator:
        df = []
        j = j + 1
        if j % 100000 == 0:
            print("已经处理",j,"行；","有效数据为：",len(dfall))
        if j == 1:
            continue
        parts = line.split('\t')
        if int(parts[2]) < 5:
            continue
        if user.get(f"{parts[0]}") == None:
            user[f"{parts[0]}"] = len(user) + 1
        if word.get(f"{parts[1]}") == None:
            word[f"{parts[1]}"] = len(user) + 1
        if user_word.get(f"{parts[1]}{parts[0]}") == None:
            user_word[f"{parts[1]}{parts[0]}"] = 1
        else:
            continue
        last_time = []
        word_last_time = []
        df.append(np.full(int(parts[2]), user[f"{parts[0]}"]))
        df.append(np.full(int(parts[2]), word[f"{parts[1]}"]))
        time = [float(num) for num in parts[3].split(',')]+[float(parts[5])]
        for i in time:
            if i == time[0]:
                word_last_time.append(i)
            else:
                word_last_time.append(i+word_last_time[-1])
        df.append(time)
        df.append(word_last_time)
        exer =  [int(num) for num in parts[4].split(',')] + [int(parts[6])]
        history_seen=[0]
        history_right =[0]
        for i in exer[1:]:
            history_right.append(i+history_right[-1])
            history_seen.append(1+history_seen[-1])
        df.append(history_right)
        df.append(history_seen)
        df.append(exer)
        data = np.stack(df, axis=1)
        dfall.append(data)
    print("Finish!")
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-4, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 2, 1))

    x, y = generate_graph_seq2seq_io_data(
        dfall,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )

    # y = np.array(torch.randint(1, 11, y.shape))
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.1)
    num_train = round(num_samples*0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ["train", "val", "test"]:
        _x, _y= locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="../data/duolingguo", help="Output directory."
    )
    parser.add_argument(
        "--kt_df_filename",
        type=str,
        default="../data/duolingguo",
        help="data readings",
    )
    #/duolingguo/en_to_de
    args = parser.parse_args()
    main(args)
