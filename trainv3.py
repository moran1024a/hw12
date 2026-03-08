import torch
import torch.nn as nn
from torch.distributions import Categorical

import os
import random
import matplotlib.pyplot as plt
import numpy as np

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
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # 状态价值评估层
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

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

    def act_batch(self, states):
        '''多样本action采样，供并行环境使用'''
        action_logits, state_values = self.forward(states)

        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return (
            actions.detach().cpu().numpy(),
            log_probs.detach().cpu().numpy(),
            state_values.squeeze(-1).detach().cpu().numpy()
        )

    def get_value(self, state):
        '''下一步状态价值计算，供 GAE 使用'''
        _, state_value = self.forward(state)
        return state_value.item()

    def get_values_batch(self, states):
        '''批量状态价值计算，供并行环境使用'''
        _, state_values = self.forward(states)
        return state_values.squeeze(-1).detach().cpu().numpy()

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


class RolloutBuffer:
    '''并行经验缓冲区'''

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.dones = []
        # GAE 计算条件 next_state_values
        # self.next_state_values = []

    def store(self, states, actions, rewards, log_probs, state_values, dones):
        self.states.append(np.array(states, dtype=np.float32))
        self.actions.append(np.array(actions, dtype=np.int64))
        self.rewards.append(np.array(rewards, dtype=np.float32))
        self.log_probs.append(np.array(log_probs, dtype=np.float32))
        self.state_values.append(np.array(state_values, dtype=np.float32))
        self.dones.append(np.array(dones, dtype=np.float32))

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.dones = []
        # self.next_state_values = []


def moving_average(data, window=50):
    result = []
    for i in range(len(data)):
        left = max(0, i - window + 1)
        result.append(sum(data[left:i + 1]) / (i - left + 1))
    return result


def compute_gae_parallel(rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95):
    '''
    并行环境下的 GAE 计算
    rewards:     [T, N]
    values:      [T, N]
    dones:       [T, N]
    last_values: [N]   rollout 结束时每个环境最后状态的 V(s_T)，若该环境最后一步 done，则应为 0
    '''
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]

        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

# ==============================


# 并行环境初始化
num_envs = 8

games = [
    LunarLanderGame(
        env_name="LunarLander-v3",
        seed=42 + i,
        # render_mode="human",
        log_dir=f"./logs/env_{i}",
        save_step_trace=False
    ) for i in range(num_envs)
]

# 超参数
actor_lr = 3e-4             # actor学习率
critic_lr = 2e-4            # critic学习率
gamma = 0.99                # 折扣因子
eps_clip = 0.2              # PPO裁剪系数
value_coef = 0.5            # critic损失权重
entropy_coef_start = 0.01   # 初始熵奖励权重
entropy_coef_end = 0.001    # 最终熵奖励权重（训练后期减少探索鼓励）
updates = 1000              # PPO 更新次数
# PPO稳定参数
gae_lambda = 0.95           # GAE lambda 参数（取值范围 [0, 1]，控制 bias-variance 权衡）
rollout_steps = 2048        # 每次更新前收集的步数
ppo_update_epochs = 8       # 同一批数据重复训练次数
mini_batch_size = 256       # mini-batch 大小
max_grad_norm = 0.5         # 梯度裁剪
# 并行环境
rollout_steps_per_env = rollout_steps // num_envs  # 每个环境需要收集的步数

# 多环境模型和缓冲区初始化
model = model_v1(games[0].get_state_dim(), games[0].get_action_dim())
buffer = RolloutBuffer()
optimizer = torch.optim.Adam([
    {"params": model.shared.parameters(), "lr": actor_lr},
    {"params": model.actor.parameters(), "lr": actor_lr},
    {"params": model.critic.parameters(), "lr": critic_lr},
])

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
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

states = [game.reset() for game in games]
current_episode_rewards = [0.0 for _ in range(num_envs)]
current_episode_steps = [0 for _ in range(num_envs)]

for update_idx in range(updates):
    '''此部分改为按照PPO更新次数循环，每次更新需要收集 rollout_steps 条数据'''
    collected_steps = 0

    # 并行数据采集（同时推进 num_envs 个环境）
    for _ in range(rollout_steps_per_env):
        states_tensor = torch.FloatTensor(np.array(states)).to(device)

        # 批量采样动作
        actions, log_probs, state_values = model.act_batch(states_tensor)

        next_states = []
        rewards = []
        dones = []

        for env_i, game in enumerate(games):
            result = game.step(int(actions[env_i]))
            next_state = result.state
            reward = result.reward
            done = result.done

            rewards.append(reward)
            dones.append(done)

            current_episode_rewards[env_i] += reward
            current_episode_steps[env_i] += 1

            if done:
                reward_history.append(current_episode_rewards[env_i])
                print(
                    f"[Episode Finished][env={env_i}] "
                    f"reward={current_episode_rewards[env_i]:.2f} | "
                    f"steps={current_episode_steps[env_i]}"
                )
                next_state = game.reset()
                current_episode_rewards[env_i] = 0.0
                current_episode_steps[env_i] = 0

            next_states.append(next_state)

        # 存储到 buffer
        buffer.store(
            states=states,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            state_values=state_values,
            dones=dones
        )

        states = next_states
        collected_steps += num_envs

    # ============================================================
    # PPO 核心训练
    # ============================================================
    # 读取 buffer 中轨迹数据并转换为张量
    # [T, N, state_dim]
    states_arr = np.array(buffer.states, dtype=np.float32)
    actions_arr = np.array(buffer.actions, dtype=np.int64)
    rewards_arr = np.array(buffer.rewards, dtype=np.float32)
    log_probs_arr = np.array(buffer.log_probs, dtype=np.float32)
    values_arr = np.array(buffer.state_values, dtype=np.float32)
    dones_arr = np.array(buffer.dones, dtype=np.float32)

    # rollout 结束后，对最后一个 next_state 做 bootstrap value 计算，供 GAE 使用
    last_state_tensor = torch.FloatTensor(np.array(states)).to(device)
    last_values = model.get_values_batch(last_state_tensor)

    # 对于最后一步已经 done 的环境，bootstrap 置 0
    last_dones = dones_arr[-1]
    last_values = last_values * (1.0 - last_dones)

    # GAE 计算优势函数和 returns
    advantages_arr, returns_arr = compute_gae_parallel(
        rewards=rewards_arr,
        values=values_arr,
        dones=dones_arr,
        last_values=last_values,
        gamma=gamma,
        gae_lambda=gae_lambda
    )

    # 展平为 PPO 训练用的一维 batch
    T, N = actions_arr.shape
    train_states = torch.FloatTensor(states_arr.reshape(T * N, -1)).to(device)
    train_actions = torch.LongTensor(actions_arr.reshape(T * N)).to(device)
    train_old_log_probs = torch.FloatTensor(
        log_probs_arr.reshape(T * N)).to(device)
    train_advantages = torch.FloatTensor(
        advantages_arr.reshape(T * N)).to(device)
    train_returns = torch.FloatTensor(returns_arr.reshape(T * N)).to(device)

    # 标准化
    train_advantages = (train_advantages - train_advantages.mean()) / \
        (train_advantages.std(unbiased=False) + 1e-8)

    # PPO更新修改为mini-batch
    data_size = train_states.size(0)

    for _ in range(ppo_update_epochs):
        indices = list(range(data_size))
        random.shuffle(indices)

        for start in range(0, data_size, mini_batch_size):
            end = start + mini_batch_size
            batch_idx = indices[start:end]

            batch_states = train_states[batch_idx]
            batch_actions = train_actions[batch_idx]
            batch_old_log_probs = train_old_log_probs[batch_idx]
            batch_advantages = train_advantages[batch_idx]
            batch_returns = train_returns[batch_idx]

            new_log_probs, new_state_values, entropy = model.evaluate(
                batch_states, batch_actions)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 +
                                eps_clip) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss_fn = nn.SmoothL1Loss()
            critic_loss = critic_loss_fn(new_state_values, batch_returns)

            entropy_loss = entropy.mean()
            progress = update_idx / updates
            current_entropy_coef = entropy_coef_start * \
                (1 - progress) + entropy_coef_end * progress

            loss = actor_loss + value_coef * critic_loss - \
                current_entropy_coef * entropy_loss

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
        f"CollectedSteps(total): {collected_steps} | "
        f"NumEnvs: {num_envs} | "
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
