import torch
import torch.nn as nn
from torch.distributions import Categorical

from game import LunarLanderGame


class model_v1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(model_v1, self).__init__()

        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 动作决策层
        self.actor = nn.Linear(128, action_dim)

        # 状态价值评估层
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        # 共享层提取特征
        shared_out = self.shared(state)
        # 分别计算动作 logits 和状态价值
        action_logits = self.actor(shared_out)
        state_value = self.critic(shared_out)

        return action_logits, state_value

    def act(self, state):
        # 获取动作 logits
        action_logits, state_value = self.forward(state)

        dist = Categorical(logits=action_logits)  # 获取动作分布
        action = dist.sample()  # 采样动作
        log_prob = dist.log_prob(action)  # 计算动作的 log 概率

        # 使用 detach() 将返回值从计算图中分离，避免梯度传播
        return action.item(), log_prob.detach(), state_value.detach()

    def evaluate(self, states, action):
        '''
        暂时无法理解，先放在这里
        '''
        # 获取动作 logits 和状态价值
        action_logits, state_value = self.forward(states)

        dist = Categorical(logits=action_logits)  # 获取动作分布
        log_prob = dist.log_prob(action)  # 计算动作的 log 概率
        entropy = dist.entropy()  # 计算动作分布的熵

        return log_prob, state_value.squeeze(-1), entropy


class buffer:
    '''经验放回缓冲区，用于存储一个 episode 中的状态、动作、奖励等信息，供后续训练使用'''

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.done = False

    def store(self, state, action, reward, log_prob, state_value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.done = done

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.done = False

# ==============================


# 环境初始化
game = LunarLanderGame(
    env_name="LunarLander-v3",
    seed=42,
    render_mode="human",
    log_dir="./logs",
    save_step_trace=False
)

# 模型和缓冲区初始化
model = model_v1(game.get_state_dim(), game.get_action_dim())
buffer = buffer()

# 超参数设置
episodes = 1000

for epoch in range(episodes):
    state = game.reset()
    done = False  # episode 是否结束

    while not done:
        # 采样动作
        action, log_prob, state_value = model.act(torch.FloatTensor(state))
        # 环境执行
        next_state, reward, done, _ = game.step(action)
        # 存储经验
        buffer.store(state, action, reward, log_prob, state_value, done)
        # 更新状态
        state = next_state

    # ==============================
    # 此部分为训练逻辑，暂未实现
    # ==============================

    # 清空缓冲区
    buffer.clear()
