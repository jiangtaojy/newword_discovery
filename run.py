# -*- coding=utf8 -*-


from __future__ import division
import time

import re
from math import log

# hanzi_re = re.compile(u"[\u4E00-\u9FD5]+", re.U)
hanzi_re = re.compile(u"[\w]+", re.U)
PHRASE_MAX_LENGTH = 5


def extract_hanzi(sentence):
    """提取汉字"""
    return hanzi_re.findall(sentence)


def cut_sentence(sentence):
    """把句子按照前后关系切分"""
    result = {}
    sentence_length = len(sentence)
    for i in range(sentence_length):
        for j in range(1, min(sentence_length - i + 1, PHRASE_MAX_LENGTH + 1)):
            tmp = sentence[i: j + i]
            result[tmp] = result.get(tmp, 0) + 1
    return result


def gen_word_dict(path):
    """统计文档所有候选词，词频（包括单字）"""
    word_dict = {}
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            hanzi_rdd = extract_hanzi(line)  # list
            for words in hanzi_rdd:
                raw_phrase_rdd = cut_sentence(words)  # dict
                for word in raw_phrase_rdd:
                    if word in word_dict:
                        word_dict[word] += raw_phrase_rdd[word]
                    else:
                        word_dict[word] = raw_phrase_rdd[word]
    return word_dict


def gen_pmi_dict(word_dict, counts, thr_fq, thr_mtro):
    """
    计算互信息，并进行了频数和互信息筛选。
    返回互信息字典
    """

    pmi_dict = {}
    for word in word_dict:
        if len(word) == 1 or word_dict.get(word) < thr_fq:
            continue
        p_x_y = max([word_dict.get(word[:i]) * word_dict.get(word[i:]) for i in range(1, len(word))])
        pmi = log(word_dict.get(word) * counts / p_x_y, 2)
        if pmi > thr_mtro:
            pmi_dict[word] = pmi
    return pmi_dict


def cal_entro(word_dict, thr_fq, thr_entro):
    """计算左右熵，并取左右熵的较小值，返回熵字典"""
    entro_l_dict = {}
    entro_r_dict = {}
    word_l_dict = {}
    word_r_dict = {}
    for word in word_dict:
        if len(word) < 3 or word_dict.get(word) < thr_fq:
            continue
        word_l_l = word[:-1]
        word_l_r = word[-1]
        word_r_l = word[0]
        word_r_r = word[1:]
        if word_l_l not in word_l_dict:
            word_l_dict[word_l_l] = [word_dict[word]]
        else:
            word_l_dict[word_l_l].append(word_dict[word])

        if word_r_r not in word_r_dict:
            word_r_dict[word_r_r] = [word_dict[word]]
        else:
            word_r_dict[word_r_r].append(word_dict[word])

    for word_l in word_l_dict:
        word_l_list = word_l_dict[word_l]
        entro_l = 0
        all_l_num = sum(word_l_list)
        for l_num in word_l_list:
            entro_l -= l_num / all_l_num * log(l_num / all_l_num, 2)
        entro_l_dict[word_l] = entro_l

    for word_r in word_r_dict:
        word_r_list = word_r_dict[word_r]
        entro_r = 0
        all_r_num = sum(word_r_list)
        for r_num in word_r_list:
            entro_r -= r_num / all_r_num * log(r_num / all_r_num, 2)
        entro_r_dict[word_r] = entro_r

    entro_dict = {}
    for word in entro_l_dict.keys() & entro_r_dict.keys():
        entro = min(entro_l_dict[word], entro_r_dict[word])
        if entro > thr_entro:
            entro_dict[word] = entro

    # for word in entro_l_dict.keys() - entro_r_dict.keys():
    #     entro = entro_l_dict[word]
    #     if entro > thr_entro:
    #         entro_dict[word] = entro
    #
    # for word in entro_r_dict.keys() - entro_l_dict.keys():
    #     entro = entro_r_dict[word]
    #     if entro > thr_entro:
    #         entro_dict[word] = entro

    return entro_dict


def final_filter(pmi_dict, entro_dict, word_dict, thr_final):
    """互信息+熵过滤筛选"""
    final_dict = {}
    for word in pmi_dict.keys() & entro_dict.keys():
        if pmi_dict[word] + entro_dict[word] > thr_final:
            print(word, pmi_dict[word], entro_dict[word])
            final_dict[word] = word_dict[word]

    print('（最终筛选后）词数量：', len(final_dict))

    return final_dict


def train_corpus_words(path):
    """读取语料文件，根据互信息、左右信息熵训练出语料词库"""
    thr_fq = 10  # 词频筛选阈值
    thr_mtro = 7  # 互信息筛选阈值
    thr_entro = 3  # 信息熵筛选阈值
    thr_final = 11
    # 步骤1：统计文档所有候选词，词频（包括单字）
    st = time.time()
    word_dict = gen_word_dict(path)
    et = time.time()
    print('读数耗时：', et - st)
    counts = sum(word_dict.values())  # 总词频数
    print('总词频数：', counts, '候选词总数：', len(word_dict))
    # print('dict内存:', sys.getsizeof(word_dict))

    # 步骤2：统计长度>1的词的左右字出现的频数，并进行了频数和互信息筛选，得到互信息词典。
    print('rl_dict is starting...')
    st = time.time()
    pmi_dict = gen_pmi_dict(word_dict, counts, thr_fq,
                            thr_mtro) 
    et = time.time()
    print('互信息筛选耗时：', et - st)


    # 步骤3： 计算左右熵，并取较小值，得到熵词典

    entro_dict = cal_entro(word_dict, thr_fq, thr_entro)
    et1 = time.time()
    print('左右熵筛选耗时：', et1 - et)


    # 步骤5： 信息熵筛选
    final_dict = final_filter(pmi_dict, entro_dict, word_dict, thr_final)
    del pmi_dict, entro_dict, word_dict

    # 步骤6：输出最终满足的词，并按词频排序
    result = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)

    with open('userdict1.txt', 'w', encoding='utf-8') as kf:
        for w, m in result:
            # print w, m
            kf.write(w + ' %d\n' % m)

    print('\n词库训练完成！总耗时：')


if __name__ == "__main__":
    path = 'mlm_train_clean'
    print('训练开始...')
    train_corpus_words(path)
    print('training is ok !')
