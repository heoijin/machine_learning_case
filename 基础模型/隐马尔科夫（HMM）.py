import numpy as np
from hmmlearn import hmm

# 定义隐式状态
states = ['A', 'B', 'C']
n_states = len(states)

# 定义观察值
observations = ['down', 'up']
n_observations = len(observations)

# 设置初始值该路为Pi
p=np.array([0.7,0.2,0.1])

# 设置状态转移矩阵A
a=np.array([
    [0.5,0.2,0.3],
    [0.3,0.5,0.2],
    [0.2,0.3,0.5]
])

# 设置状态对观测的生成矩阵B
b=np.array([
    [0.6,0.2],
    [0.3,0.3],
    [0.1,0.5]
])

# 设置观察状态
o=np.array([[1,0,1,1,1]]).T

# 初始化模型参数
model=hmm.MultinomialHMM(n_components=n_states)
model.startprob_=p # 初始值Pi
model.transmat_=a # 转移矩阵
model.emissionprob_=b # 状态对观察生成矩阵

logprob,h=model.decode(o,algorithm='viterbi')
print('The hidden h',','.join(map(lambda x : states[x],h)))




