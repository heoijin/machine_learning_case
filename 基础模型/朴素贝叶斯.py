import jieba
import numpy as np


def load():
    '''
    读取训练数据集，利用jieba拆分句为单词
    :return: 每句对应的情感色彩，0为负面，1为正面
    '''
    arr = [
        '不知道该说什么，这么烂的抄袭片也能上映，我感到很尴尬',
        '天呐。一个大写的滑稽',
        '剧情太狗血，演技太浮夸，结局太无语。总体太差了。这一个半小时废了。',
        '画面很美，音乐很好听，主角演得很到位，很值得一看得电影，男主角很帅很帅，赞赞赞',
        '超级喜欢得一部爱情电影',
        '故事情节吸引人，演员演得也很好，电影里得歌也很好听，总之值得一看，看了之后也会很感动',
    ]
    ret = []
    for i in arr:
        words = jieba.lcut(i)  # 将句子切分成词
        ret.append(words)
    emotional = [0, 0, 0, 1, 1, 1]
    return ret, emotional


def create_vocab(data):
    '''
    创建词汇表：包含训练集中出现得所有词汇
    :param data:
    :return:
    '''
    vocab_set = set([])
    for document in data:
        vocab_set = vocab_set | set(document)  # 取并集
    return list(vocab_set)


def words_to_vec(vocab_list, vocab_set):
    '''
    将句之转换成词表格式，类似于OneHot转换
    :param vocab_list:
    :param vocab_set:
    :return:
    '''
    ret = np.zeros(len(vocab_list))  # 创建数据表中中得一行，并设置初值为0
    for word in vocab_set:
        if word in vocab_list:
            ret[vocab_list.index(word)] = 1  # 若该词在本句中出现，则设置为1
    return ret


def train(X, y):
    '''
    1. 根据公式计算出每个词在正/反例中出现的概率
    2. 整体实例中，正例所占比例
    :param X:
    :param y:
    :return:
    '''
    rows = X.shape[0]
    cols = X.shape[1]
    percent = sum(y) / float(rows)
    p0_arr = np.ones(cols)
    p1_arr = np.ones(cols)
    p0_count = 2.0  # 设置初始值为2，后作分母
    p1_count = 2.0
    for i in range(rows):  # 遍历每一句
        if y[i] == 1:
            p1_arr += X[i]
            p1_count += sum(X[i])
        else:
            p0_arr += X[i]
            p0_count += sum(X[i])
    p1_vec = np.log(p1_arr / p1_count)  # 当为正例时，每个词出现的概率
    p0_vec = np.log(p0_arr / p0_count)
    return p0_vec, p1_vec, percent


def predict(X, p0_vec, p1_vec, percent):
    p1 = sum(X * p1_vec) + np.log(percent)  # 分类为1的概率
    p0 = sum(X * p0_vec) + np.log(1.0 - percent)  # 分类为0的概率
    if p1 > p0:
        return 1
    else:
        return 0


def main():
    '''
    朴素贝叶斯算法的具体实现
    :return:
    '''
    sentences, y = load()
    vocab_list = create_vocab(sentences)
    X = []
    for sentence in sentences:
        X.append(words_to_vec(vocab_list, sentence))

    p0_vec, p1_vec, percent = train(np.array(X), np.array(y))
    test = jieba.lcut('抄袭得这么明显也是醉了！')
    test_X = np.array(words_to_vec(vocab_list, test))
    print(test, f'分类{predict(test_X, p0_vec, p1_vec, percent)}')  # ['抄袭', '得', '这么', '明显', '也', '是', '醉', '了', '！'] 分类0


if __name__ == '__main__':
    main()
