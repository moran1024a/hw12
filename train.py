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
        return action.item(), log_prob.item(), state_value.item()

    def evaluate(self, states, actions):
        '''
        暂时无法理解，先放在这里
        '''
        # 获取动作 logits 和状态价值
        action_logits, state_value = self.forward(states)

        dist = Categorical(logits=action_logits)  # 获取动作分布
        log_prob = dist.log_prob(actions)  # 计算动作的 log 概率
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
        self.dones = []

    def store(self, state, action, reward, log_prob, state_value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.dones = []

# ==============================


# 环境初始化
game = LunarLanderGame(
    env_name="LunarLander-v3",
    seed=42,
    #render_mode="human",
    log_dir="./logs",
    save_step_trace=False
)

# 模型和缓冲区初始化
model = model_v1(game.get_state_dim(), game.get_action_dim())
buffer = buffer()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# 超参数
gamma = 0.99         # 折扣因子
eps_clip = 0.2       # PPO裁剪系数
value_coef = 0.5     # critic损失权重
entropy_coef = 0.01  # 熵奖励权重
episodes = 1000      # 训练 episode 数

for epoch in range(episodes):
    state = game.reset()
    done = False  # episode 是否结束

    while not done:
        # 采样动作
        action, log_prob, state_value = model.act(torch.FloatTensor(state))
        # 环境执行
        result = game.step(action)
        next_state = result.state
        reward = result.reward
        done = result.done
        # 存储经验
        buffer.store(state, action, reward, log_prob, state_value, done)
        # 更新状态
        state = next_state

    # ============================================================
    # 以下PPO核心训练内容理解不够准确，部分来自AI
    # ============================================================
    # 从缓冲区中提取数据，准备训练
    states = torch.FloatTensor(buffer.states)  # 环境状态
    actions = torch.LongTensor(buffer.actions)  # 执行的动作
    rewards = buffer.rewards  # 奖励反馈
    old_log_probs = torch.FloatTensor(buffer.log_probs)  # 旧动作概率
    old_state_values = torch.FloatTensor(buffer.state_values)  # 旧状态价值评估
    dones = buffer.dones  # episode 是否结束

    returns = []
    discounted_return = 0

    for reward, done_flag in zip(reversed(rewards), reversed(dones)):
        # 如果 episode 结束，重置折扣回报为0，否则继续累积折扣回报
        if done_flag:
            discounted_return = 0
        discounted_return = reward + gamma * discounted_return
        returns.insert(0, discounted_return)  # 将当前折扣回报插入列表开头

    returns = torch.FloatTensor(returns)

    # 优势函数
    advantages = returns - old_state_values
    # 标准化
    advantages = (advantages-advantages.mean()) / \
        (advantages.std(unbiased=False)+1e-8)

    # 评估当前策略
    new_log_probs, new_state_values, entropy = model.evaluate(states, actions)

    # 计算 PPO 损失
    ratio = torch.exp(new_log_probs - old_log_probs)  # 计算概率比

    # 计算裁剪的损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()  # PPO actor 损失

    # 计算 critic 损失
    critic_loss = nn.MSELoss()(new_state_values, returns)

    # 熵奖励损失
    entropy_loss = entropy.mean()

    # 总损失
    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

    # ============================================================

    # 反向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 清空缓冲区
    buffer.clear()
