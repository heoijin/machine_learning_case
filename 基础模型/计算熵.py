import math


def entropy(*c):
    '''
    计算熵，*c为各类别发生概率
    :param c:
    :return:
    '''
    if (len(c)) <= 0:
        return -1
    result = 0
    for x in c:
        result += (-x) * math.log(x, 2)
    return result
