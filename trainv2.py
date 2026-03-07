import torch
import torch.nn as nn
from torch.distributions import Categorical

import os
import random
import matplotlib.pyplot as plt

from game import LunarLanderGame


class model_v1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(model_v1, self).__init__()

        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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

    def get_value(self, state):
        '''下一步状态价值计算，供 GAE 使用'''
        _, state_value = self.forward(state)
        return state_value.item()

    def evaluate(self, states, actions):
        '''
        输入一批 states 和 actions，
        返回这些动作在当前策略下的 log_prob、状态价值和熵
        '''
        # 获取动作 logits 和状态价值
        action_logits, state_value = self.forward(states)

        dist = Categorical(logits=action_logits)  # 获取动作分布
        log_prob = dist.log_prob(actions)  # 计算动作的 log 概率
        entropy = dist.entropy()  # 计算动作分布的熵

        return log_prob, state_value.squeeze(-1), entropy


class buffer:
    '''经验缓冲区'''

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.dones = []
        # GAE 计算条件 next_state_values
        self.next_state_values = []

    def store(self, state, action, reward, log_prob, state_value, next_state_value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.next_state_values.append(next_state_value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.dones = []
        self.next_state_values = []


# >>>>>>> [ADD] 新增：滑动平均函数
def moving_average(data, window=50):
    result = []
    for i in range(len(data)):
        left = max(0, i - window + 1)
        result.append(sum(data[left:i + 1]) / (i - left + 1))
    return result
# <<<<<<< [ADD]


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    '''计算 GAE 优势函数和 returns'''
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)

    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns
# <<<<<<< [ADD]

# ==============================


# 环境初始化
game = LunarLanderGame(
    env_name="LunarLander-v3",
    seed=42,
    # render_mode="human",
    log_dir="./logs",
    save_step_trace=False
)

# 超参数
LR = 3e-4            # 学习率
gamma = 0.99         # 折扣因子
eps_clip = 0.2       # PPO裁剪系数
value_coef = 0.5     # critic损失权重
entropy_coef = 0.01  # 熵奖励权重(鼓励探索)
updates = 1000       # PPO 更新次数
# PPO稳定参数
gae_lambda = 0.95            # GAE lambda 参数（取值范围 [0, 1]，控制 bias-variance 权衡）
rollout_steps = 2048         # 每次更新前收集的步数
ppo_update_epochs = 10       # 同一批数据重复训练次数
mini_batch_size = 256        # mini-batch 大小
max_grad_norm = 0.5          # 梯度裁剪

# 模型和缓冲区初始化
model = model_v1(game.get_state_dim(), game.get_action_dim())
buffer = buffer()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# 保存路径
save_dir = "./ppo_outputs"
os.makedirs(save_dir, exist_ok=True)

# 数据记录
reward_history = []          # 每个完整 episode 的 reward
actor_loss_history = []      # 训练过程中 actor loss 的历史记录
critic_loss_history = []     # 训练过程中 critic loss 的历史记录
entropy_history = []         # 训练过程中 policy entropy 的历史记录
total_loss_history = []      # 训练过程中总 loss 的历史记录

best_avg_reward = -float("inf")  # 目前最优平均奖励
save_window = 50  # 计算平均奖励的窗口大小

state = game.reset()
current_episode_reward = 0.0  # 统计当前 episode 的奖励总和
current_episode_steps = 0  # 统计当前 episode 的步数

for update_idx in range(updates):
    '''此部分改为按照PPO更新次数循环，每次更新需要收集 rollout_steps 条数据'''
    collected_steps = 0

    # 数据采集
    while collected_steps < rollout_steps:
        # 采样动作
        action, log_prob, state_value = model.act(
            torch.FloatTensor(state).to(device))

        # 环境执行
        result = game.step(action)
        next_state = result.state
        reward = result.reward
        done = result.done

        # next_state_value 计算，供 GAE 使用
        if done:
            next_state_value = 0.0
        else:
            next_state_value = model.get_value(
                torch.FloatTensor(next_state).to(device))

        # buffer保存轨迹数据
        buffer.store(state, action, reward, log_prob,
                     state_value, next_state_value, done)

        # 更新 rollout 状态
        state = next_state
        collected_steps += 1

        # 统计完整 episode 奖励（用于保存模型和可视化训练过程）
        current_episode_reward += reward
        current_episode_steps += 1

        # episode 结束后记录奖励并重置环境
        if done:
            reward_history.append(current_episode_reward)
            print(
                f"[Episode Finished] reward={current_episode_reward:.2f} | steps={current_episode_steps}")

            state = game.reset()
            current_episode_reward = 0.0
            current_episode_steps = 0

    # ============================================================
    # PPO 核心训练
    # ============================================================
    # 读取 buffer 中轨迹数据并转换为张量
    states = torch.FloatTensor(buffer.states).to(device)
    actions = torch.LongTensor(buffer.actions).to(device)
    old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
    old_state_values = torch.FloatTensor(buffer.state_values).to(device)
    next_state_values = torch.FloatTensor(buffer.next_state_values).to(device)
    rewards = buffer.rewards
    dones = buffer.dones

    # 优势函数修改为 GAE：
    # 1. 原始 returns - V(s) 方差较大，训练容易崩
    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_state_values.tolist(),
        next_values=next_state_values.tolist(),
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda
    )

    advantages = torch.FloatTensor(advantages).to(device)
    returns = torch.FloatTensor(returns).to(device)

    # 标准化
    advantages = (advantages - advantages.mean()) / \
        (advantages.std(unbiased=False) + 1e-8)

    # PPO更新修改为mini-batch
    data_size = states.size(0)

    for _ in range(ppo_update_epochs):
        indices = list(range(data_size))
        random.shuffle(indices)

        for start in range(0, data_size, mini_batch_size):
            end = start + mini_batch_size
            batch_idx = indices[start:end]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            new_log_probs, new_state_values, entropy = model.evaluate(
                batch_states, batch_actions)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 +
                                eps_clip) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(new_state_values, batch_returns)

            entropy_loss = entropy.mean()

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    # 记录训练指标
    actor_loss_history.append(actor_loss.item())
    critic_loss_history.append(critic_loss.item())
    entropy_history.append(entropy_loss.item())
    total_loss_history.append(loss.item())

    # 根据最近 save_window 个 episode 的平均奖励保存最优模型
    if len(reward_history) > 0:
        avg_reward = sum(reward_history[-save_window:]) / \
            len(reward_history[-save_window:])
    else:
        avg_reward = 0.0

    # 保存最新模型
    torch.save(
        {
            "update_idx": update_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_reward": avg_reward,
        },
        os.path.join(save_dir, "latest_model.pt")
    )

    if len(reward_history) >= 10 and avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(
            {
                "update_idx": update_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_avg_reward": best_avg_reward,
            },
            os.path.join(save_dir, "best_model.pt")
        )
        print(
            f"[SAVE] best model updated at update {update_idx + 1}, avg_reward={best_avg_reward:.3f}")

    # 训练信息输出
    print(
        f"Update {update_idx + 1}/{updates} | "
        f"CollectedSteps: {collected_steps} | "
        f"Episodes: {len(reward_history)} | "
        f"AvgReward({save_window}): {avg_reward:.2f} | "
        f"ActorLoss: {actor_loss.item():.4f} | "
        f"CriticLoss: {critic_loss.item():.4f} | "
        f"Entropy: {entropy_loss.item():.4f}"
    )

    # 清空 buffer
    buffer.clear()

# ============================================================
# 训练可视化
# ============================================================
ma_reward = moving_average(reward_history, window=save_window)

# reward 曲线
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Episode Reward")
plt.plot(ma_reward, label=f"Moving Avg Reward ({save_window})")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Training Reward Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "reward_curve.png"))
plt.close()

# loss 曲线
plt.figure(figsize=(10, 5))
plt.plot(actor_loss_history, label="Actor Loss")
plt.plot(critic_loss_history, label="Critic Loss")
plt.plot(total_loss_history, label="Total Loss")
plt.xlabel("Update")
plt.ylabel("Loss")
plt.title("PPO Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()

# entropy 曲线
plt.figure(figsize=(10, 5))
plt.plot(entropy_history, label="Entropy")
plt.xlabel("Update")
plt.ylabel("Entropy")
plt.title("Policy Entropy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "entropy_curve.png"))
plt.close()
