import math
import time
import random
import numpy as np
from tqdm import tqdm, trange              # 显示进度条


# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('func %s, run time: %s' % (func.__name__, stop_time-start_time))
        return res
    return wrapper


# load data    split data
class Dataset():
    def __init__(self, fp):
        # fp: data file path
        self.data = self.loadData(fp)

    @timmer
    def loadData(self, fp):
        data = []
        for f in open(fp).readlines()[1:]:
            data.append(f.strip().split('\t')[:3])

        new_data = {}
        for user, item, tag in data:
            if user not in new_data:
                new_data[user] = {}
            if item not in new_data[user]:
                new_data[user][item] = set()
            new_data[user][item].add(tag)

        ret = []
        for user in new_data:
            for item in new_data[user]:
                ret.append((user, item, list(new_data[user][item])))

        return ret

    @timmer
    def splitData(self, M, k, seed=1):
        '''
        # :param: self.data 加载所有的（user， item）数据条目
        # :param: M, 划分的数目，最后需要取M折平均
        # :param: k， 本次是第几轮划分
        # :param: seed, random的随机种子数，对于不同的k应设置为一样的
        # :return:  train， test
        '''
        train, test = [], []
        random.seed(seed)
        for user, item, tags in self.data:
            # randint左右都覆盖
            if random.randint(0, M-1) == k:
                test.append((user, item, tags))
            else:
                train.append((user, item, tags))
        # 处理为字典的形式
        def convert_dict(data):
            data_dict = {}
            for user, item, tags in data:
                if user not in data_dict:
                    data_dict[user] = {}
                data_dict[user][item] = tags
            return data_dict

        return convert_dict(train), convert_dict(test)

# 评价指标 Precision Recall Coverage Diversity Popularity(Novelty)
class Metric():
    def __init__(self, train, test, GetRecommendation):
        '''
        :param train:
        :param test:
        :param GetRecommendation:  为某个用户获得推荐的函数
        '''
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    def getRec(self):
        recs = {}
        for user in self.test:
            recs[user] = {}
            for item in self.test[user]:
                rank = self.GetRecommendation(user, item)
                recs[user][item] = rank
        return recs

    def recall(self):
        hit = 0
        all = 0
        for user in self.test:
            for item in self.test[user]:
                test_tags = set(self.test[user][item])
                rank = self.recs[user][item]
                for tag, score in rank:
                    if tag in test_tags:
                        hit += 1
                all += len(test_tags)
        return round(hit / all * 100, 2)

    def precision(self):
        hit = 0
        all = 0
        for user in self.test:
            for item in self.test[user]:
                test_tags = set(self.test[user][item])
                rank = self.recs[user][item]
                for tag, score in rank:
                    if tag in test_tags:
                        hit += 1
                all += len(rank)
        return round(hit / all * 100, 2)

    def eval(self):
        metric = {
            'Precision': self.precision(),
            'Recall': self.recall()
        }
        print('Metric: ', metric)
        return metric

# 算法实现
# 1.推荐热门标签
def PopularTags(train, N):
    tags = {}
    for user in train:
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in tags:
                    tags[tag] = 0
                tags[tag] += 1
    tags = list(sorted(tags.items(), key=lambda x: x[1], reverse=True))[:N]

    def GetRecommendation(user, item):
        return tags

    return GetRecommendation

# 2.推荐用户最热门标签
def UserPopularTags(train, N):
    user_tags = {}
    for user in train:
        if user not in user_tags:
            user_tags[user] = {}
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
    user_tags = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in user_tags.items()}

    def GetRecommendation(user, item):
        if user in user_tags:
            return user_tags[user][:N]
        else:
            return []

    return GetRecommendation

# 3.推荐最热门标签
def ItemPopularTags(train, N):
    item_tags = {}
    for user in train:
        for item in train[user]:
            if item not in item_tags:
                item_tags[item] = {}
            for tag in train[user][item]:
                if tag not in item_tags[item]:
                    item_tags[item][tag] = 0
                item_tags[item][tag] += 1

    item_tags = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in item_tags.items()}

    def GetRecommendation(user, item):
        if item in item_tags:
            return item_tags[item][:N]
        else:
            return []

    return GetRecommendation

# 4.联合用户和物品进行推荐
def HybridPopularTags(train, N, alpha):
    user_tags = {}
    for user in train:
        if user not in user_tags:
            user_tags[user] = {}
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1

    item_tags = {}
    for user in train:
        for item in train[user]:
            if item not in item_tags:
                item_tags[item] = {}
            for tag in train[user][item]:
                if tag not in item_tags[item]:
                    item_tags[item][tag] = 0
                item_tags[item][tag] += 1

    def GetRecommendation(user, item):
        tag_score = {}
        if user in user_tags:
            max_user_tag = max(user_tags[user].values())
            for tag in user_tags[user]:
                if tag not in tag_score:
                    tag_score[tag] = 0
                tag_score[tag] += (1 - alpha) * user_tags[user][tag] / max_user_tag

        if item in item_tags:
            max_item_tag = max(item_tags[item].values())
            for tag in item_tags[item]:
                if tag not in tag_score:
                    tag_score[tag] = 0
                tag_score[tag] += alpha * item_tags[item][tag] / max_item_tag

        return list(sorted(tag_score.items(), key=lambda x: x[1], reverse=True))[:N]

    return GetRecommendation

# 实验
class Experiment():
    def __init__(self, M, N, fp='./dataset/delicious-2k/user_taggedbookmarks.dat', rt='PopularTags'):
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'PopularTags': PopularTags, 'UserPopularTags': UserPopularTags,
                    'ItemPopularTags': ItemPopularTags, 'HybridPopularTags': HybridPopularTags}

    @timmer
    def worker(self, train, test, **kwargs):
        getRecommendation = self.alg[self.rt](train, self.N, **kwargs)
        metric = Metric(train, test, getRecommendation)

        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self, **kwargs):
        metrics = {'Precision': 0, 'Recall': 0}
        dataset = Dataset(self.fp)
        for i in range(self.M):
            train, test = dataset.splitData(self.M, i)
            print('Experiment {}'.format(i))
            metric = self.worker(train, test, **kwargs)
            metrics = {k: metrics[k]+metric[k] for k in metrics}
        metrics = {k: metrics[k]/self.M for k in metrics}
        print('Average Result (M={}, N={}): {}'.format(self.M, self.N, metrics))


# 1. PopularTags实验
# M, N = 10, 10
# exp = Experiment(M, N, rt='PopularTags')
# exp.run()
#
# Average Result (M=10, N=10): {'Precision': 2.7589999999999995, 'Recall': 6.606999999999999}

# 2. UserPopularTags实验
# M, N = 10, 10
# exp = Experiment(M, N, rt='UserPopularTags')
# exp.run()
#
# Average Result (M=10, N=10): {'Precision': 10.518, 'Recall': 24.963}

# 2. ItemPopularTags实验
M, N = 10, 10
exp = Experiment(M, N, rt='ItemPopularTags')
exp.run()

# Average Result (M=10, N=10): {'Precision': 12.052, 'Recall': 8.575}


# 4.HybridPopularTags实验
# M, N = 10, 10
# for alpha in range(0, 11):
#     alpha /= 10
#     print('alpha =', alpha)
#     exp = Experiment(M, N, rt='HybridPopularTags')
#     exp.run(alpha=alpha)

# alpha = 0.0
# Average Result (M=10, N=10): {'Precision': 10.482, 'Recall': 24.978}
# alpha = 0.1
# Average Result (M=10, N=10): {'Precision': 10.794, 'Recall': 25.720999999999997}
# alpha = 0.2
# Average Result (M=10, N=10): {'Precision': 11.138, 'Recall': 26.538}
# alpha = 0.3
# Average Result (M=10, N=10): {'Precision': 11.396999999999998, 'Recall': 27.158000000000005}
# alpha = 0.4
# Average Result (M=10, N=10): {'Precision': 11.539, 'Recall': 27.496}
# alpha = 0.5
# Average Result (M=10, N=10): {'Precision': 11.591000000000001, 'Recall': 27.619000000000007}
# alpha = 0.6
# Average Result (M=10, N=10): {'Precision': 11.586, 'Recall': 27.606}
# alpha = 0.7
# Average Result (M=10, N=10): {'Precision': 11.56, 'Recall': 27.542}
# alpha = 0.8
# Average Result (M=10, N=10): {'Precision': 11.524000000000001, 'Recall': 27.458}
# alpha = 0.9
# Average Result (M=10, N=10): {'Precision': 11.504000000000001, 'Recall': 27.409999999999997}
# alpha = 1.0
# Average Result (M=10, N=10): {'Precision': 7.502000000000001, 'Recall': 17.878999999999998}



