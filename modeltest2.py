import os
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from game import LunarLanderGame


class model_v1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(model_v1, self).__init__()

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
        shared_out = self.shared(state)
        action_logits = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_logits, state_value

    def act_deterministic(self, state):
        action_logits, state_value = self.forward(state)
        action = torch.argmax(action_logits, dim=-1)
        return action.item(), state_value.item()


def safe_get_attr(obj, name, default=None):
    return getattr(obj, name, default)


def moving_average(data, window=5):
    if len(data) == 0:
        return []
    ma = []
    for i in range(len(data)):
        left = max(0, i - window + 1)
        ma.append(sum(data[left:i + 1]) / (i - left + 1))
    return ma


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


def normalize_obs(obs, obs_rms, clip_range=10.0):
    """
    与训练代码保持一致的状态归一化函数
    """
    obs = np.asarray(obs, dtype=np.float32)
    obs_norm = (obs - obs_rms.mean) / obs_rms.std
    obs_norm = np.clip(obs_norm, -clip_range, clip_range)
    return obs_norm.astype(np.float32)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_episode(model, obs_rms, seed, render=False, log_dir="./test_logs"):
    """
    每个 episode 单独创建一个环境，确保 seed 真正生效。
    """
    set_global_seed(seed)

    game = LunarLanderGame(
        env_name="LunarLander-v3",
        seed=seed,
        render_mode="human" if render else None,
        log_dir=log_dir,
        save_step_trace=False
    )

    state = game.reset()
    state = normalize_obs(state, obs_rms)
    done = False

    episode_reward = 0.0
    step_count = 0

    last_success = 0
    last_crash = 0
    last_timeout = 0

    with torch.no_grad():
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, _ = model.act_deterministic(state_tensor)

            result = game.step(action)

            raw_next_state = result.state
            reward = result.reward
            done = result.done
            state = normalize_obs(raw_next_state, obs_rms)
            episode_reward += reward
            step_count += 1

            last_success = safe_get_attr(result, "success", last_success)
            last_crash = safe_get_attr(result, "crash", last_crash)
            last_timeout = safe_get_attr(result, "timeout", last_timeout)

    # 如果你的 LunarLanderGame 有 close()，最好关掉
    if hasattr(game, "close") and callable(getattr(game, "close")):
        game.close()

    return {
        "seed": seed,
        "reward": float(episode_reward),
        "steps": int(step_count),
        "success": int(bool(last_success)),
        "crash": int(bool(last_crash)),
        "timeout": int(bool(last_timeout)),
    }


def save_results_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = ["episode", "seed", "reward",
                  "steps", "success", "crash", "timeout"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(results, start=1):
            row = {
                "episode": idx,
                "seed": item["seed"],
                "reward": item["reward"],
                "steps": item["steps"],
                "success": item["success"],
                "crash": item["crash"],
                "timeout": item["timeout"],
            }
            writer.writerow(row)


def plot_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    episodes = list(range(1, len(results) + 1))
    rewards = [x["reward"] for x in results]
    steps = [x["steps"] for x in results]
    success = sum(x["success"] for x in results)
    crash = sum(x["crash"] for x in results)
    timeout = sum(x["timeout"] for x in results)
    others = len(results) - success - crash - timeout

    reward_ma = moving_average(rewards, window=5)

    plt.figure(figsize=(16, 10))

    # 1) 每局 reward 曲线
    plt.subplot(2, 3, 1)
    plt.plot(episodes, rewards, marker="o")
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # 2) reward 移动平均
    plt.subplot(2, 3, 2)
    plt.plot(episodes, reward_ma, marker="o")
    plt.title("Reward Moving Average (window=5)")
    plt.xlabel("Episode")
    plt.ylabel("MA Reward")
    plt.grid(True, alpha=0.3)

    # 3) reward 分布
    plt.subplot(2, 3, 3)
    plt.hist(rewards, bins=min(10, max(5, len(results) // 2)))
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    # 4) 结果统计
    plt.subplot(2, 3, 4)
    labels = ["Success", "Crash", "Timeout", "Other"]
    values = [success, crash, timeout, max(others, 0)]
    plt.bar(labels, values)
    plt.title("Outcome Counts")
    plt.ylabel("Count")

    # 5) Steps vs Reward
    plt.subplot(2, 3, 5)
    plt.scatter(steps, rewards)
    plt.title("Steps vs Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # 6) 累积平均 reward
    plt.subplot(2, 3, 6)
    cumulative_avg = [sum(rewards[:i]) / i for i in range(1, len(rewards) + 1)]
    plt.plot(episodes, cumulative_avg, marker="o")
    plt.title("Cumulative Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_summary(results):
    rewards = [x["reward"] for x in results]
    steps = [x["steps"] for x in results]

    success_count = sum(x["success"] for x in results)
    crash_count = sum(x["crash"] for x in results)
    timeout_count = sum(x["timeout"] for x in results)

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    max_reward = float(np.max(rewards)) if rewards else 0.0
    min_reward = float(np.min(rewards)) if rewards else 0.0
    avg_steps = float(np.mean(steps)) if steps else 0.0

    print("\n" + "=" * 70)
    print("回测结束，统计结果如下：")
    print(f"测试局数          : {len(results)}")
    print(f"平均奖励          : {avg_reward:.2f}")
    print(f"奖励标准差        : {std_reward:.2f}")
    print(f"最高奖励          : {max_reward:.2f}")
    print(f"最低奖励          : {min_reward:.2f}")
    print(f"平均步数          : {avg_steps:.2f}")
    print(f"成功次数          : {success_count}")
    print(f"坠毁次数          : {crash_count}")
    print(f"超时次数          : {timeout_count}")
    print(f"成功率            : {success_count / len(results) * 100:.2f}%")
    print(f"坠毁率            : {crash_count / len(results) * 100:.2f}%")
    print(f"超时率            : {timeout_count / len(results) * 100:.2f}%")
    print("=" * 70)


def main():
    # =========================
    # 可调参数
    # =========================
    model_path = "./ppo_outputs/best_model.pt"
    test_episodes = 1000            # 建议至少 30；更稳一点可以 50 / 100
    render = False                  # 批量评估建议 False；想肉眼看时再开 True
    log_dir = "./test_logs"
    csv_path = os.path.join(log_dir, "evaluation_results.csv")
    fig_path = os.path.join(log_dir, "evaluation_summary.png")

    # 是否使用固定种子列表，便于复现
    use_fixed_seed_list = True
    base_seed = 20260307
    # =========================

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    os.makedirs(log_dir, exist_ok=True)

    # 先用一个临时环境拿 state_dim / action_dim
    temp_game = LunarLanderGame(
        env_name="LunarLander-v3",
        seed=base_seed,
        render_mode=None,
        log_dir=log_dir,
        save_step_trace=False
    )

    state_dim = temp_game.get_state_dim()
    action_dim = temp_game.get_action_dim()

    if hasattr(temp_game, "close") and callable(getattr(temp_game, "close")):
        temp_game.close()

    model = model_v1(state_dim, action_dim)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    obs_rms = RunningMeanStd(shape=(state_dim,))
    obs_rms.mean = checkpoint["obs_rms_mean"]
    obs_rms.var = checkpoint["obs_rms_var"]
    obs_rms.count = checkpoint["obs_rms_count"]

    print("=" * 70)
    print(f"已加载模型: {model_path}")
    if "best_avg_reward" in checkpoint:
        print(f"训练阶段 best_avg_reward: {checkpoint['best_avg_reward']:.3f}")
    if "avg_reward" in checkpoint:
        print(f"训练阶段 avg_reward: {checkpoint['avg_reward']:.3f}")
    if "update_idx" in checkpoint:
        print(f"保存于 update_idx: {checkpoint['update_idx']}")
    if "epoch" in checkpoint:
        print(f"保存于 epoch: {checkpoint['epoch']}")
    print("=" * 70)

    # 生成每局不同 seed
    if use_fixed_seed_list:
        # 固定种子列表：可复现，每局不同
        episode_seeds = [base_seed + i for i in range(test_episodes)]
    else:
        # 完全随机：每次运行都不同
        rng = random.Random()
        episode_seeds = [rng.randint(0, 10**9) for _ in range(test_episodes)]

    results = []

    for episode_idx, seed in enumerate(episode_seeds, start=1):
        result = run_one_episode(
            model=model,
            obs_rms=obs_rms,
            seed=seed,
            render=render,
            log_dir=log_dir
        )
        results.append(result)

        print(
            f"[Test Episode {episode_idx:02d}/{test_episodes}] "
            f"Seed: {seed:<10d} | "
            f"Reward: {result['reward']:8.2f} | "
            f"Steps: {result['steps']:4d} | "
            f"Success: {result['success']} | "
            f"Crash: {result['crash']} | "
            f"Timeout: {result['timeout']}"
        )

    save_results_csv(results, csv_path)
    plot_results(results, fig_path)
    print_summary(results)

    print(f"\n结果明细已保存到: {csv_path}")
    print(f"可视化图片已保存到: {fig_path}")


if __name__ == "__main__":
    main()
