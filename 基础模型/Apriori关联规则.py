def load():
    '''
    生成简单购物数据
    :return:
    '''
    data = [
        ['香蕉', '苹果', '梨', '葡萄', '樱桃', '西瓜', '芒果', '枇杷'],
        ['苹果', '菠萝', '梨', '香蕉', '荔枝', '芒果', '橙子'],
        ['菠萝', '香蕉', '橘子', '橙子'],
        ['菠萝', '梨', '枇杷'],
        ['香蕉', '苹果', '梨', '荔枝', '枇杷', '芒果', '香瓜']
    ]
    return data


def create_collection_1(data):
    '''
    创建所有物品的集合
    :param data:
    :return:
    '''
    c = []
    for item in data:
        for g in item:
            if not [g] in c:
                c.append([g])
    c.sort()
    return list(map(frozenset, c)) # frozenset函数用于冻结列表，使得列表无法增加或删除元素


def check_support(d_list, c_list, min_support):
    '''
    计算每组组合的支持度，并判断是否大于设置的最下支持度
    - 如果满足条件，加入列表，用字典的方式返回各种组合的支持度
    :param d_list: 购物数据
    :param c_list: 物品集合
    :param min_support: 支持度
    :return ret: 满足支持率的组合
    :return support_dic: 支持度字典
    '''
    c_dic = {}  # 组合计数
    for d in d_list:  # 每次购物
        for c in c_list:  # 每个组合
            if c.issubset(d): # 判断集合x所有元素是否都包含在指定集合y中，x.issubset(y)
                if c in c_dic:
                    c_dic[c] += 1  # 组合计数+1
                else:
                    c_dic[c] = 1  # 将组合加入字典
    d_count = float(len(d_list))  # 购物次数
    ret = []
    support_dic = {}
    for key in c_dic:
        support = c_dic[key] / d_count
        if support >= min_support:  # 判断支持度
            ret.append(key)
        support_dic[key] = support  # 记录支持度
    return ret, support_dic  # 返回满足支持率的组合和支持度字典


def create_collection_n(lk, k):
    '''
    建立多个商品组合，只创建组合，不做任何筛选和判断
    :param lk:
    :param k: 用于设置组合中商品的个数
    :return:
    '''
    ret = []
    for i in range(len(lk)):
        for j in range(i + 1, len(lk)):
            l1 = list(lk[i])[:k - 2]
            l1.sort()
            l2 = list(lk[j])[:k - 2]
            l2.sort()
            if l1 == l2:
                ret.append(lk[i] | lk[j])
    return ret


def apripri(data, min_support=0.5):
    '''
    - 依次创建从一个商品到多个商品的组合
    - 判断各个组合是否满足支持度
        - 如果满足，加入返回列表

    :param data:
    :param min_support:
    :return:
    '''
    c1 = create_collection_1(data) # 创建一个包含所有物品的集合，此集合无法修改
    d_list = list(map(set, data))  # 将购物列表转换成集合列表
    l1, support_dic = check_support(d_list, c1, min_support)
    l = [l1]
    k = 2
    while len(l[k - 2]) > 0:
        ck = create_collection_n(l[k - 2], k)  # 建立新组合
        lk, support = check_support(d_list, ck, min_support)  # 判断新组合是否满足支持率
        support_dic.update(support)
        l.append(lk)
        k += 1
    return l, support_dic


def main():
    data = load()
    l, support_dic = apripri(data)
    print(l)
    print(support_dic)


if __name__ == '__main__':
    main()
