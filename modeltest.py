import os
import torch
import torch.nn as nn
from torch.distributions import Categorical

import random

from game import LunarLanderGame


class model_v1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(model_v1, self).__init__()

        # 与训练时保持一致
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared_out = self.shared(state)
        action_logits = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_logits, state_value

    def act_deterministic(self, state):
        """
        确定性动作选择：
        直接选择 action_logits 中概率最高的动作，不进行采样
        """
        action_logits, state_value = self.forward(state)
        action = torch.argmax(action_logits, dim=-1)
        return action.item(), state_value.item()


def safe_get_attr(obj, name, default=None):
    """
    安全读取属性：
    兼容你的 game.step(action) 返回对象中
    可能有 success / crash / timeout，也可能没有
    """
    return getattr(obj, name, default)


def main():
    # =========================
    # 可调参数
    # =========================
    model_path = "./ppo_outputs/best_model.pt"
    test_episodes = 20
    render = True
    env_seed = random.randint(0, 10000)  # 每次测试使用不同的随机种子
    # =========================

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    # 初始化环境
    game = LunarLanderGame(
        env_name="LunarLander-v3",
        seed=env_seed,
        render_mode="human" if render else None,
        log_dir="./test_logs",
        save_step_trace=False
    )

    # 初始化模型
    model = model_v1(game.get_state_dim(), game.get_action_dim())

    # 加载权重
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 60)
    print(f"已加载模型: {model_path}")
    if "best_avg_reward" in checkpoint:
        print(f"训练阶段 best_avg_reward: {checkpoint['best_avg_reward']:.3f}")
    if "update_idx" in checkpoint:
        print(f"保存于 update_idx: {checkpoint['update_idx']}")
    if "epoch" in checkpoint:
        print(f"保存于 epoch: {checkpoint['epoch']}")
    print("=" * 60)

    all_rewards = []
    success_count = 0
    crash_count = 0
    timeout_count = 0

    with torch.no_grad():
        for episode in range(test_episodes):
            state = game.reset()
            done = False

            episode_reward = 0.0
            step_count = 0

            last_success = 0
            last_crash = 0
            last_timeout = 0

            while not done:
                state_tensor = torch.FloatTensor(state)
                action, _ = model.act_deterministic(state_tensor)

                result = game.step(action)

                next_state = result.state
                reward = result.reward
                done = result.done

                episode_reward += reward
                step_count += 1
                state = next_state

                last_success = safe_get_attr(result, "success", last_success)
                last_crash = safe_get_attr(result, "crash", last_crash)
                last_timeout = safe_get_attr(result, "timeout", last_timeout)

            all_rewards.append(episode_reward)
            success_count += int(bool(last_success))
            crash_count += int(bool(last_crash))
            timeout_count += int(bool(last_timeout))

            print(
                f"[Test Episode {episode + 1:02d}/{test_episodes}] "
                f"Reward: {episode_reward:8.2f} | "
                f"Steps: {step_count:4d} | "
                f"Success: {int(bool(last_success))} | "
                f"Crash: {int(bool(last_crash))} | "
                f"Timeout: {int(bool(last_timeout))}"
            )

    # 汇总统计
    avg_reward = sum(all_rewards) / len(all_rewards)
    max_reward = max(all_rewards)
    min_reward = min(all_rewards)

    print("\n" + "=" * 60)
    print("回测结束，统计结果如下：")
    print(f"测试局数      : {test_episodes}")
    print(f"平均奖励      : {avg_reward:.2f}")
    print(f"最高奖励      : {max_reward:.2f}")
    print(f"最低奖励      : {min_reward:.2f}")
    print(f"成功次数      : {success_count}")
    print(f"坠毁次数      : {crash_count}")
    print(f"超时次数      : {timeout_count}")
    print(f"成功率        : {success_count / test_episodes * 100:.2f}%")
    print(f"坠毁率        : {crash_count / test_episodes * 100:.2f}%")
    print(f"超时率        : {timeout_count / test_episodes * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
