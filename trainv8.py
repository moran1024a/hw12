# ==============================================================
# PPO训练脚本 - 版本8
# 说明：
#    - 8帧叠帧（与每帧8个特征组成8*8二维矩阵）
#    - CNN网络结构（适配8*8输入）
#    - 轻量注意力机制（CNN后接Self-Attention）
# ==============================================================

import torch
import torch.nn as nn
from torch.distributions import Categorical

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import deque

from game import LunarLanderGame

# 叠帧参数（不要修改）
STACK_FRAMES = 8

def init_frame_stack(first_obs, stack_frames=STACK_FRAMES):
    '''
    用初始观测构造帧栈。
    初始时还没有历史帧，因此将第一帧复制 stack_frames 次，
    得到 (8, 8) 的二维状态矩阵。

    参数：
        first_obs: shape=(8,) 的单帧状态
    返回：
        frame_stack: deque, 长度=8
    '''
    first_obs = np.asarray(first_obs, dtype=np.float32)
    frame_stack = deque(maxlen=stack_frames)

    for _ in range(stack_frames):
        frame_stack.append(first_obs.copy())

    return frame_stack


def update_frame_stack(frame_stack, new_obs):
    '''
    将新的一帧状态压入帧栈尾部，自动挤掉最旧的一帧。
    '''
    frame_stack.append(np.asarray(new_obs, dtype=np.float32))
    return frame_stack


def get_stacked_state(frame_stack):
    '''
    将 deque 中的 8 帧拼成 shape=(8, 8) 的二维矩阵。
    第0维表示时间帧，第1维表示单帧8个特征。
    '''
    return np.stack(frame_stack, axis=0).astype(np.float32)


class model_v1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(model_v1, self).__init__()

        # CNN特征提取层，适配输入状态 shape=(8, 8)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 轻量自注意力层，适配 CNN 输出的特征维度
        self.attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        self.shared = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _encode(self, state):
        '''
        批样本状态编码器
        输入：
            单样本: (8, 8)
            批样本: (B, 8, 8)

        输出：
            shared_out: (B, 128)
        '''
        # 单样本扩成 batch 形式
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, 8, 8)

        # 增加通道维，给 CNN 使用
        # (B, 8, 8) -> (B, 1, 8, 8)
        x = state.unsqueeze(1)

        # CNN特征提取
        # (B, 1, 8, 8) -> (B, 64, 8, 8)
        x = self.cnn(x)

        # 展开成 token 序列
        # (B, 64, 8, 8) -> (B, 64, 64)
        # 解释：
        #   - 8x8 共64个空间位置，每个位置一个 token
        #   - 每个 token 的 embedding dim = 64
        x = x.flatten(2).transpose(1, 2)

        # Self-Attention
        attn_out, _ = self.attn(x, x, x)   # (B, 64, 64)

        # 全局平均池化，将 64 个 token 聚合成一个全局特征
        x = attn_out.mean(dim=1)           # (B, 64)

        # shared 共享层
        shared_out = self.shared(x)        # (B, 128)
        return shared_out

    def forward(self, state):
        # 共享层提取特征
        shared_out = self._encode(state)

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


def moving_average(data, window=50):
    result = []
    for i in range(len(data)):
        left = max(0, i - window + 1)
        result.append(sum(data[left:i + 1]) / (i - left + 1))
    return result


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


class RunningMeanStd:
    '''
    运行时均值方差统计器
    - 给状态归一化提供 mean / std
    - 给奖励缩放提供 return 的方差统计
    '''

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # 避免初始除零

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)

        # 如果是一条状态，例如 shape=(8,)
        # 则扩展成 (1, 8) 方便统一处理
        if x.ndim == 1:
            x = x[None, :]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * \
            batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


def normalize_obs(obs, obs_rms, clip_range=10.0):
    '''
    环境归一化函数
    - 对环境状态做标准化
    - 再做clip，避免极端值干扰训练
    '''
    obs = np.asarray(obs, dtype=np.float32)
    obs_norm = (obs - obs_rms.mean) / obs_rms.std
    obs_norm = np.clip(obs_norm, -clip_range, clip_range)
    return obs_norm.astype(np.float32)


class RewardScaler:
    '''
    奖励缩放器
    - 对 discounted return 的波动范围做统计，再用它缩放 reward
    '''

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.running_return = 0.0
        self.return_rms = RunningMeanStd(shape=())

    def reset(self):
        # 每个 episode 结束都要重置
        self.running_return = 0.0

    def scale(self, reward):
        # 维护折扣累计回报
        self.running_return = self.gamma * self.running_return + reward

        # 注意：RunningMeanStd.update 对标量要传 shape=(1,) 数组
        self.return_rms.update(
            np.array([[self.running_return]], dtype=np.float64))

        # 用 return 的标准差缩放即时 reward
        scaled_reward = reward / (np.sqrt(self.return_rms.var) + 1e-8)

        # clip 防止极端异常值
        scaled_reward = np.clip(scaled_reward, -10.0, 10.0)
        return float(scaled_reward)


def linear_lr_decay(initial_lr, final_lr, progress):
    '''
    线性学习率衰减
    参数：
        initial_lr: 初始学习率
        final_lr:   最终学习率下限
        progress:   当前训练进度，范围 [0, 1]
    返回：
        current_lr: 当前应使用的学习率
    '''
    progress = min(max(progress, 0.0), 1.0)
    current_lr = initial_lr + (final_lr - initial_lr) * progress
    return current_lr


# BadSeed 课程采样辅助函数
def load_badseed_csv(csv_path):
    '''
    从 CSV 中读取 badseed。
    CSV结构：
        case,badseed
    其中：
        case = 0 -> 坠毁
        case = 1 -> 超时

    返回：
        all_badseeds      : 所有 badseed 列表
        crash_badseeds    : 坠毁 badseed 列表
        timeout_badseeds  : 超时 badseed 列表
    '''
    all_badseeds = []
    crash_badseeds = []
    timeout_badseeds = []

    if (csv_path is None) or (not os.path.exists(csv_path)):
        print(f"[WARN] badseed CSV 不存在，将退化为普通随机采样: {csv_path}")
        return all_badseeds, crash_badseeds, timeout_badseeds

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # 简单校验列名
        fieldnames = reader.fieldnames if reader.fieldnames is not None else []
        if ("case" not in fieldnames) or ("badseed" not in fieldnames):
            raise ValueError(
                f"badseed CSV 列名错误，期望包含 ['case', 'badseed']，实际为: {fieldnames}"
            )

        for row in reader:
            case_value = str(row["case"]).strip()
            seed_value = str(row["badseed"]).strip()

            if seed_value == "":
                continue

            # 尽量兼容整数或浮点形式的 seed
            seed_int = int(float(seed_value))
            all_badseeds.append(seed_int)

            if case_value == "0":
                crash_badseeds.append(seed_int)
            elif case_value == "1":
                timeout_badseeds.append(seed_int)
            else:
                # 允许未知 case，但仍保留到 all_badseeds
                pass

    print(
        f"[INFO] 读取 badseed 完成 | all={len(all_badseeds)} | "
        f"crash={len(crash_badseeds)} | timeout={len(timeout_badseeds)}"
    )
    return all_badseeds, crash_badseeds, timeout_badseeds


def get_badseed_ratio(progress,
                      warmup_end=0.30,
                      ramp_end=0.70,
                      ratio_start=0.10,
                      ratio_mid=0.40,
                      ratio_end=0.60):
    '''
    按训练进度返回当前 badseed 采样率。

    设计为三段式课程采样：
    1) 前 30% 训练：固定低比例 badseed，避免过早扰乱基础策略
    2) 中间 40%：线性从 ratio_start 增长到 ratio_mid
    3) 最后 30%：线性从 ratio_mid 增长到 ratio_end
    '''
    progress = min(max(progress, 0.0), 1.0)

    if progress <= warmup_end:
        return ratio_start

    if progress <= ramp_end:
        inner_progress = (progress - warmup_end) / \
            max(1e-8, (ramp_end - warmup_end))
        return ratio_start + (ratio_mid - ratio_start) * inner_progress

    inner_progress = (progress - ramp_end) / max(1e-8, (1.0 - ramp_end))
    return ratio_mid + (ratio_end - ratio_mid) * inner_progress


def reset_game_with_seed(game, seed=None):
    '''
    带 seed 的 reset 封装。
    之所以单独封装，是为了最小程度修改主训练循环。

    优先尝试：
        game.reset(seed=seed)
    如果你的 LunarLanderGame.reset() 不支持 seed 参数，
    则会抛 TypeError，此时退化为普通 reset，并给出提示。
    '''
    if seed is None:
        return game.reset()

    try:
        return game.reset(seed=seed)
    except TypeError:
        print(
            "[WARN] 当前 LunarLanderGame.reset() 可能不支持 seed 参数，"
            "本次将退化为普通 reset。请在 game 封装中确认 reset(seed=...) 接口。"
        )
        return game.reset()


def sample_reset_state(game, progress, use_badseed_curriculum, badseed_pool):
    '''
    按课程采样策略决定本 episode 是否使用 badseed。

    返回：
        state                : reset 后的原始状态
        current_seed         : 本 episode 使用的 seed；若为普通随机则返回 None
        current_seed_source  : "badseed" 或 "normal"
        current_badseed_ratio: 当前训练进度下的 badseed 采样率
    '''
    # 默认不使用 badseed
    current_seed = None
    current_seed_source = "normal"
    current_badseed_ratio = 0.0

    if use_badseed_curriculum and len(badseed_pool) > 0:
        current_badseed_ratio = get_badseed_ratio(
            progress=progress,
            warmup_end=badseed_warmup_end,
            ramp_end=badseed_ramp_end,
            ratio_start=badseed_ratio_start,
            ratio_mid=badseed_ratio_mid,
            ratio_end=badseed_ratio_end
        )

        # 以当前 badseed 采样率决定是否从 badseed 池抽样
        if random.random() < current_badseed_ratio:
            current_seed = random.choice(badseed_pool)
            current_seed_source = "badseed"

    # 使用选中的 seed 做 reset
    state = reset_game_with_seed(game, seed=current_seed)
    return state, current_seed, current_seed_source, current_badseed_ratio


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
actor_lr = 3e-4             # 初始acto/shared学习率
critic_lr = 2e-4            # 初始critic学习率
actor_lr_end = 3e-5         # 最终actor/shared 学习率【原 3e-5】
critic_lr_end = 5e-5        # 最终critic 学习率【原 5e-5】
gamma = 0.99                # 折扣因子
eps_clip = 0.2              # PPO裁剪系数
value_coef = 0.7            # critic损失权重【原 0.5】
entropy_coef_start = 0.01   # 初始熵奖励权重
entropy_coef_end = 0.0015   # 最终熵奖励权重（训练后期减少探索鼓励）【原 0.0005】
updates = 1500              # PPO 更新次数

# PPO稳定参数
gae_lambda = 0.95           # GAE lambda 参数（取值范围 [0, 1]，控制 bias-variance 权衡）
rollout_steps = 4096        # 每次更新前收集的步数
ppo_update_epochs = 6       # 同一批数据重复训练次数【原 8】
mini_batch_size = 512       # mini-batch 大小【原 256】

max_grad_norm = 0.5         # 梯度裁剪

target_kl = 0.02           # 当近似 KL 超过该阈值时，提前停止当前 update 的 PPO epoch

use_obs_norm = True         # 状态归一化
use_reward_scaling = True   # 奖励缩放

badseed_csv_path = "./badseedlist.csv"   # badseed CSV 文件路径
use_badseed_curriculum = True            # 是否启用 badseed 课程采样

# 三段式课程采样参数：
# 前 30% 训练：badseed 比例固定为 10%
# 中间 40% 训练：从 10% 线性升到 40%
# 最后 30% 训练：从 40% 线性升到 60%
badseed_warmup_end = 0.30               # 阶段1结束进度
badseed_ramp_end = 0.70                 # 阶段2结束进度
badseed_ratio_start = 0.05              # 前期 badseed 采样率
badseed_ratio_mid = 0.30                # 中期 badseed 采样率
badseed_ratio_end = 0.50                # 后期 badseed 采样率

# 读取 badseed 列表
all_badseeds, crash_badseeds, timeout_badseeds = load_badseed_csv(
    badseed_csv_path)
# ==============================================================

# 初始化运行时均值方差统计器和奖励缩放器
obs_rms = RunningMeanStd(shape=(game.get_state_dim(),))
reward_scaler = RewardScaler(gamma=0.99)

model = model_v1(game.get_state_dim(), game.get_action_dim())
buffer = buffer()

# 优化器初始化，注意共享层和 actor 同学习率，critic 独立学习率
optimizer = torch.optim.Adam([
    {"params": model.shared.parameters(), "lr": actor_lr},   # 参数组0：shared
    {"params": model.actor.parameters(), "lr": actor_lr},    # 参数组1：actor
    {"params": model.critic.parameters(), "lr": critic_lr},  # 参数组2：critic
])

# cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# 保存路径
save_dir = "./ppo_outputs"
os.makedirs(save_dir, exist_ok=True)

# 数据记录
reward_history = []          # 每个完整 episode 的原始环境 reward（不是缩放后的reward）
actor_loss_history = []      # 训练过程中 actor loss 的历史记录
critic_loss_history = []     # 训练过程中 critic loss 的历史记录
entropy_history = []         # 训练过程中 policy entropy 的历史记录
total_loss_history = []      # 训练过程中总 loss 的历史记录
actor_lr_history = []        # 记录每次 update 的 actor/shared 学习率
critic_lr_history = []       # 记录每次 update 的 critic 学习率

badseed_ratio_history = []       # 每次 update 的目标 badseed 采样率
episode_seed_source_history = []  # 每个 episode 是否来自 badseed

best_avg_reward = -float("inf")  # 目前最优平均奖励
save_window = 50  # 计算平均奖励的窗口大小

# 环境初始化
# 初始叠帧流程：单帧归一化 -> 初始化8帧栈 -> 得到(8,8)状态
raw_state, current_episode_seed, current_episode_seed_source, current_badseed_ratio = sample_reset_state(
    game=game,
    progress=0.0,
    use_badseed_curriculum=use_badseed_curriculum,
    badseed_pool=all_badseeds
)

# 先用初始“单帧状态”更新统计量，再做归一化
if use_obs_norm:
    obs_rms.update(raw_state)
    raw_state = normalize_obs(raw_state, obs_rms)

# 初始化8帧栈，并构造叠帧状态 (8, 8)
frame_stack = init_frame_stack(raw_state, stack_frames=STACK_FRAMES)
state = get_stacked_state(frame_stack)

# 每个episode开始前重置 reward scaler
reward_scaler.reset()

current_episode_reward = 0.0  # 统计当前 episode 的原始奖励总和
current_episode_steps = 0     # 统计当前 episode 的步数

for update_idx in range(updates):
    '''
    此部分改为按照PPO更新次数循环，每次更新需要收集 rollout_steps 条数据
    '''

    # 训练进度
    progress = update_idx / max(1, updates - 1)

    # 记录当前 update 的 badseed 采样率（即使本 update 没有 badseed 采样，也记录当前课程采样目标比例，方便分析训练过程）
    current_badseed_ratio_for_log = get_badseed_ratio(
        progress=progress,
        warmup_end=badseed_warmup_end,
        ramp_end=badseed_ramp_end,
        ratio_start=badseed_ratio_start,
        ratio_mid=badseed_ratio_mid,
        ratio_end=badseed_ratio_end
    ) if (use_badseed_curriculum and len(all_badseeds) > 0) else 0.0
    badseed_ratio_history.append(current_badseed_ratio_for_log)
    # ==============================

    # 决策层学习率更新
    current_actor_lr = linear_lr_decay(
        initial_lr=actor_lr,
        final_lr=actor_lr_end,
        progress=progress
    )
    # 价值层学习率更新
    current_critic_lr = linear_lr_decay(
        initial_lr=critic_lr,
        final_lr=critic_lr_end,
        progress=progress
    )
    # 更新优化器中对应参数组的学习率
    optimizer.param_groups[0]["lr"] = current_actor_lr   # shared 跟随 actor 学习率
    optimizer.param_groups[1]["lr"] = current_actor_lr   # actor 学习率
    optimizer.param_groups[2]["lr"] = current_critic_lr  # critic 学习率

    actor_lr_history.append(current_actor_lr)
    critic_lr_history.append(current_critic_lr)

    collected_steps = 0

    # 数据采集
    while collected_steps < rollout_steps:
        # 采样动作
        action, log_prob, state_value = model.act(
            torch.FloatTensor(state).to(device)
        )

        # 环境执行
        # 注意：game.step(action) 返回的 reward 是原始环境奖励，不是缩放后的 reward
        result = game.step(action)
        raw_next_state = result.state      # 单帧原始状态 shape=(8,)
        raw_reward = result.reward
        done = result.done

        # next_state 处理逻辑：
        # 1. 对单帧状态做归一化
        # 2. 更新到帧栈
        # 3. 得到新的叠帧状态 shape=(8,8)
        if use_obs_norm:
            # 先更新统计量，再归一化
            obs_rms.update(raw_next_state)
            norm_next_frame = normalize_obs(raw_next_state, obs_rms)
        else:
            norm_next_frame = np.asarray(raw_next_state, dtype=np.float32)

        # 将新单帧压入帧栈，生成新的 8帧叠帧状态
        frame_stack = update_frame_stack(frame_stack, norm_next_frame)
        next_state = get_stacked_state(frame_stack)

        # 对 reward 做缩放
        if use_reward_scaling:
            reward = reward_scaler.scale(raw_reward)
        else:
            reward = raw_reward

        # next_state_value 计算，供 GAE 使用
        if done:
            next_state_value = 0.0
        else:
            next_state_value = model.get_value(
                torch.FloatTensor(next_state).to(device)
            )

        # buffer 保存轨迹数据
        # 注意：
        # - state / next_state 存的是叠帧后的状态 shape=(8,8)
        # - reward 存的是缩放后的 reward（若开启）
        buffer.store(
            state, action, reward, log_prob,
            state_value, next_state_value, done
        )

        # 更新 rollout 状态
        state = next_state
        collected_steps += 1

        # 统计原始奖励总和和 episode 步数
        current_episode_reward += raw_reward
        current_episode_steps += 1

        # episode 结束后记录奖励并重置环境
        if done:
            reward_history.append(current_episode_reward)

            # 记录当前 episode 是否来自 badseed
            episode_seed_source_history.append(current_episode_seed_source)

            print(
                f"[Episode Finished] reward={current_episode_reward:.2f} | "
                f"steps={current_episode_steps} | "
                f"seed_source={current_episode_seed_source} | "
                f"seed={current_episode_seed} | "
                f"badseed_ratio={current_badseed_ratio:.3f}"
            )

            # reset 环境并初始化新 episode 的状态
            raw_state, current_episode_seed, current_episode_seed_source, current_badseed_ratio = sample_reset_state(
                game=game,
                progress=progress,
                use_badseed_curriculum=use_badseed_curriculum,
                badseed_pool=all_badseeds
            )

            # 注意：这里的 raw_state 是原始环境单帧状态
            if use_obs_norm:
                obs_rms.update(raw_state)
                raw_state = normalize_obs(raw_state, obs_rms)

            frame_stack = init_frame_stack(raw_state, stack_frames=STACK_FRAMES)
            state = get_stacked_state(frame_stack)

            # 每个 episode 结束都要重置 reward scaler 的累计回报
            reward_scaler.reset()

            current_episode_reward = 0.0
            current_episode_steps = 0

    # ============================================================
    # PPO 核心训练
    # ============================================================
    # 读取 buffer 中轨迹数据并转换为张量
    # states 现在形状为 (B, 8, 8)
    states = torch.FloatTensor(
        np.array(buffer.states, dtype=np.float32)).to(device)
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

    # 标准化 advantage，降低训练方差
    advantages = (advantages - advantages.mean()) / \
        (advantages.std(unbiased=False) + 1e-8)

    # 裁剪 advantage，避免极端样本导致更新过猛
    advantages = torch.clamp(advantages, -10, 10)

    # PPO更新修改为mini-batch
    data_size = states.size(0)

    # KL early stop 标志位
    early_stop = False

    for epoch_idx in range(ppo_update_epochs):
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
                batch_states, batch_actions
            )

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 +
                                eps_clip) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss_fn = nn.SmoothL1Loss()
            critic_loss = critic_loss_fn(new_state_values, batch_returns)

            entropy_loss = entropy.mean()
            progress_entropy = update_idx / max(1, updates - 1)
            current_entropy_coef = entropy_coef_start * \
                (1 - progress_entropy) + entropy_coef_end * progress_entropy

            loss = actor_loss + value_coef * critic_loss - \
                current_entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # 计算近似 KL 散度，用于 early stop
            approx_kl = (batch_old_log_probs - new_log_probs).mean().item()

            # 如果策略变化过大，则提前停止当前 update 的剩余 PPO epoch
            if approx_kl > target_kl:
                print(
                    f"[KL Early Stop] update={update_idx + 1}, "
                    f"epoch={epoch_idx + 1}, approx_kl={approx_kl:.6f}"
                )
                early_stop = True
                break

        # 跳出外层 epoch 循环
        if early_stop:
            break

    # 记录训练指标
    actor_loss_history.append(actor_loss.item())
    critic_loss_history.append(critic_loss.item())
    entropy_history.append(entropy_loss.item())
    total_loss_history.append(loss.item())

    # 根据最近 save_window 个 episode 的平均奖励保存最优模型
    # 注意：这里的 reward_history 仍然是原始环境得分，便于真实评估训练效果
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
            "current_actor_lr": current_actor_lr,
            "current_critic_lr": current_critic_lr,
            "obs_rms_mean": obs_rms.mean,
            "obs_rms_var": obs_rms.var,
            "obs_rms_count": obs_rms.count,
            "use_badseed_curriculum": use_badseed_curriculum,
            "badseed_csv_path": badseed_csv_path,
            "badseed_ratio_start": badseed_ratio_start,
            "badseed_ratio_mid": badseed_ratio_mid,
            "badseed_ratio_end": badseed_ratio_end,
            "badseed_warmup_end": badseed_warmup_end,
            "badseed_ramp_end": badseed_ramp_end,
            "stack_frames": STACK_FRAMES,
            "model_arch": "cnn_attention"
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
                "current_actor_lr": current_actor_lr,
                "current_critic_lr": current_critic_lr,
                "obs_rms_mean": obs_rms.mean,
                "obs_rms_var": obs_rms.var,
                "obs_rms_count": obs_rms.count,
                "use_badseed_curriculum": use_badseed_curriculum,
                "badseed_csv_path": badseed_csv_path,
                "badseed_ratio_start": badseed_ratio_start,
                "badseed_ratio_mid": badseed_ratio_mid,
                "badseed_ratio_end": badseed_ratio_end,
                "badseed_warmup_end": badseed_warmup_end,
                "badseed_ramp_end": badseed_ramp_end,
                "stack_frames": STACK_FRAMES,
                "model_arch": "cnn_attention"
            },
            os.path.join(save_dir, "best_model.pt")
        )
        print(
            f"[SAVE] best model updated at update {update_idx + 1}, avg_reward={best_avg_reward:.3f}"
        )

    # 训练信息输出
    print(
        f"Update {update_idx + 1}/{updates} | "
        f"CollectedSteps: {collected_steps} | "
        f"Episodes: {len(reward_history)} | "
        f"AvgReward({save_window}): {avg_reward:.2f} | "
        f"ActorLoss: {actor_loss.item():.4f} | "
        f"CriticLoss: {critic_loss.item():.4f} | "
        f"Entropy: {entropy_loss.item():.4f} | "
        f"ActorLR: {current_actor_lr:.8f} | "
        f"CriticLR: {current_critic_lr:.8f} | "
        f"BadSeedRatio: {current_badseed_ratio_for_log:.3f}"
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

# 学习率曲线
plt.figure(figsize=(10, 5))
plt.plot(actor_lr_history, label="Actor/Shared LR")
plt.plot(critic_lr_history, label="Critic LR")
plt.xlabel("Update")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lr_curve.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(badseed_ratio_history, label="BadSeed Sampling Ratio")
plt.xlabel("Update")
plt.ylabel("BadSeed Ratio")
plt.title("BadSeed Curriculum Schedule Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "badseed_ratio_curve.png"))
plt.close()


# 训练结束后统计 badseed 相关信息，分析课程采样对训练的影响
if len(episode_seed_source_history) > 0:
    badseed_episode_count = sum(
        [1 for x in episode_seed_source_history if x == "badseed"])
    normal_episode_count = sum(
        [1 for x in episode_seed_source_history if x == "normal"])
    total_episode_count = len(episode_seed_source_history)

    print(
        f"[Seed Source Summary] total_episodes={total_episode_count} | "
        f"badseed_episodes={badseed_episode_count} | "
        f"normal_episodes={normal_episode_count} | "
        f"badseed_episode_ratio={badseed_episode_count / max(1, total_episode_count):.4f}"
    )