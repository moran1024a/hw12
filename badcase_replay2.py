# badcase_replay.py
# ==============================================================
# 用途：
# 1. 对指定 seed 的 LunarLander bad case 进行单独回放
# 2. 支持 render 可视化开关
# 3. 支持保存逐步轨迹 CSV
# 4. 支持打印最后 N 步状态
# 5. 支持画出该局的轨迹诊断图
#
# 使用方式：
# 直接修改 main() 里的参数即可
# ==============================================================

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from game import LunarLanderGame


# ==============================================================
# 模型结构：必须与训练时保持一致
# ==============================================================
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

    def forward(self, state):
        shared_out = self.shared(state)
        action_logits = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_logits, state_value

    def act_deterministic(self, state):
        """
        回放时使用确定性动作：
        直接选 logits 最大的动作，便于复现和分析。
        """
        action_logits, state_value = self.forward(state)
        action = torch.argmax(action_logits, dim=-1)
        return action.item(), state_value.item()


# ==============================================================
# 与训练保持一致的 obs normalization
# ==============================================================
class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


def normalize_obs(obs, obs_rms, clip_range=10.0):
    obs = np.asarray(obs, dtype=np.float32)
    obs_norm = (obs - obs_rms.mean) / obs_rms.std
    obs_norm = np.clip(obs_norm, -clip_range, clip_range)
    return obs_norm.astype(np.float32)


# ==============================================================
# 工具函数
# ==============================================================
def safe_get_attr(obj, name, default=None):
    return getattr(obj, name, default)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_to_text(action):
    """
    LunarLander 离散动作的常见含义：
    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine
    """
    mapping = {
        0: "noop",
        1: "left_engine",
        2: "main_engine",
        3: "right_engine",
    }
    return mapping.get(int(action), f"action_{action}")


# ==============================================================
# 单个 seed 回放
# ==============================================================
def replay_one_seed(
    model,
    obs_rms,
    seed,
    render=False,
    log_dir="./badcase_replay_logs",
):
    """
    运行单个 seed，并返回完整轨迹信息。
    """
    set_global_seed(seed)

    game = LunarLanderGame(
        env_name="LunarLander-v3",
        seed=seed,
        render_mode="human" if render else None,
        log_dir=log_dir,
        save_step_trace=False
    )

    raw_state = game.reset(seed=seed)
    norm_state = normalize_obs(raw_state, obs_rms)

    done = False
    step_idx = 0
    total_reward = 0.0

    last_success = 0
    last_crash = 0
    last_timeout = 0

    # 逐步轨迹记录
    trajectory = []

    with torch.no_grad():
        while not done:
            state_tensor = torch.tensor(norm_state, dtype=torch.float32)
            action, value = model.act_deterministic(state_tensor)

            result = game.step(action)

            raw_next_state = np.asarray(result.state, dtype=np.float32)
            reward = float(result.reward)
            done = bool(result.done)

            success = safe_get_attr(result, "success", last_success)
            crash = safe_get_attr(result, "crash", last_crash)
            timeout = safe_get_attr(result, "timeout", last_timeout)

            # 当前 step 的原始状态拆分
            # 默认 LunarLander 状态：
            # [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
            x = float(raw_state[0]) if len(raw_state) > 0 else 0.0
            y = float(raw_state[1]) if len(raw_state) > 1 else 0.0
            vx = float(raw_state[2]) if len(raw_state) > 2 else 0.0
            vy = float(raw_state[3]) if len(raw_state) > 3 else 0.0
            angle = float(raw_state[4]) if len(raw_state) > 4 else 0.0
            ang_vel = float(raw_state[5]) if len(raw_state) > 5 else 0.0
            left_leg = float(raw_state[6]) if len(raw_state) > 6 else 0.0
            right_leg = float(raw_state[7]) if len(raw_state) > 7 else 0.0

            trajectory.append({
                "step": step_idx,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "angle": angle,
                "angular_velocity": ang_vel,
                "left_leg": left_leg,
                "right_leg": right_leg,
                "action": int(action),
                "action_text": action_to_text(action),
                "value": float(value),
                "reward": reward,
                "success": int(bool(success)),
                "crash": int(bool(crash)),
                "timeout": int(bool(timeout)),
            })

            total_reward += reward
            step_idx += 1

            last_success = success
            last_crash = crash
            last_timeout = timeout

            raw_state = raw_next_state
            norm_state = normalize_obs(raw_state, obs_rms)

    if hasattr(game, "close") and callable(getattr(game, "close")):
        game.close()

    summary = {
        "seed": int(seed),
        "reward": float(total_reward),
        "steps": int(step_idx),
        "success": int(bool(last_success)),
        "crash": int(bool(last_crash)),
        "timeout": int(bool(last_timeout)),
    }

    return summary, trajectory


# ==============================================================
# 保存逐步轨迹到 CSV
# ==============================================================
def save_trajectory_csv(trajectory, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "step",
        "x", "y",
        "vx", "vy",
        "angle", "angular_velocity",
        "left_leg", "right_leg",
        "action", "action_text",
        "value",
        "reward",
        "success", "crash", "timeout",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory)


# ==============================================================
# 打印最后 N 步
# ==============================================================
def print_last_steps(trajectory, last_n=30):
    if len(trajectory) == 0:
        print("轨迹为空")
        return

    print("\n" + "=" * 120)
    print(f"最后 {min(last_n, len(trajectory))} 步详细信息：")
    print("=" * 120)

    selected = trajectory[-last_n:]

    header = (
        f"{'step':>6} | {'x':>9} | {'y':>9} | {'vx':>9} | {'vy':>9} | "
        f"{'angle':>9} | {'ang_vel':>9} | {'L':>3} | {'R':>3} | "
        f"{'action':>12} | {'value':>10} | {'reward':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in selected:
        print(
            f"{row['step']:6d} | "
            f"{row['x']:9.4f} | "
            f"{row['y']:9.4f} | "
            f"{row['vx']:9.4f} | "
            f"{row['vy']:9.4f} | "
            f"{row['angle']:9.4f} | "
            f"{row['angular_velocity']:9.4f} | "
            f"{int(row['left_leg'] > 0.5):3d} | "
            f"{int(row['right_leg'] > 0.5):3d} | "
            f"{row['action_text']:>12s} | "
            f"{row['value']:10.4f} | "
            f"{row['reward']:10.4f}"
        )

    print("=" * 120)


# ==============================================================
# 绘图：单局诊断图
# ==============================================================
def plot_trajectory(trajectory, summary, save_path):
    """
    画该 seed 的单局轨迹诊断图。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(trajectory) == 0:
        return

    steps = [row["step"] for row in trajectory]
    xs = [row["x"] for row in trajectory]
    ys = [row["y"] for row in trajectory]
    vxs = [row["vx"] for row in trajectory]
    vys = [row["vy"] for row in trajectory]
    angles = [row["angle"] for row in trajectory]
    ang_vels = [row["angular_velocity"] for row in trajectory]
    actions = [row["action"] for row in trajectory]
    values = [row["value"] for row in trajectory]
    rewards = [row["reward"] for row in trajectory]

    plt.figure(figsize=(16, 12))

    # 1. x / y
    plt.subplot(3, 2, 1)
    plt.plot(steps, xs, label="x")
    plt.plot(steps, ys, label="y")
    plt.title("Position")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. vx / vy
    plt.subplot(3, 2, 2)
    plt.plot(steps, vxs, label="vx")
    plt.plot(steps, vys, label="vy")
    plt.title("Velocity")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. angle / angular_velocity
    plt.subplot(3, 2, 3)
    plt.plot(steps, angles, label="angle")
    plt.plot(steps, ang_vels, label="angular_velocity")
    plt.title("Attitude")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. action
    plt.subplot(3, 2, 4)
    plt.plot(steps, actions, marker="o", markersize=2, linewidth=1)
    plt.title("Action")
    plt.xlabel("Step")
    plt.ylabel("Discrete Action")
    plt.grid(True, alpha=0.3)

    # 5. reward
    plt.subplot(3, 2, 5)
    plt.plot(steps, rewards)
    plt.title("Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # 6. value
    plt.subplot(3, 2, 6)
    plt.plot(steps, values)
    plt.title("Value Estimate")
    plt.xlabel("Step")
    plt.ylabel("V(s)")
    plt.grid(True, alpha=0.3)

    plt.suptitle(
        f"Seed={summary['seed']} | Reward={summary['reward']:.2f} | "
        f"Steps={summary['steps']} | Success={summary['success']} | "
        f"Crash={summary['crash']} | Timeout={summary['timeout']}",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=220)
    plt.close()


# ==============================================================
# 打印单局摘要
# ==============================================================
def print_summary(summary, trajectory):
    print("\n" + "=" * 80)
    print("单 seed 回放结果")
    print(f"Seed               : {summary['seed']}")
    print(f"Reward             : {summary['reward']:.2f}")
    print(f"Steps              : {summary['steps']}")
    print(f"Success            : {summary['success']}")
    print(f"Crash              : {summary['crash']}")
    print(f"Timeout            : {summary['timeout']}")

    if len(trajectory) > 0:
        main_engine_ratio = sum(
            1 for x in trajectory if x["action"] == 2) / len(trajectory)
        side_engine_ratio = sum(1 for x in trajectory if x["action"] in [
                                1, 3]) / len(trajectory)

        if len(trajectory) >= 2:
            switches = sum(
                1 for i in range(1, len(trajectory))
                if trajectory[i]["action"] != trajectory[i - 1]["action"]
            )
            switch_rate = switches / (len(trajectory) - 1)
        else:
            switch_rate = 0.0

        max_abs_x = max(abs(x["x"]) for x in trajectory)
        max_abs_vx = max(abs(x["vx"]) for x in trajectory)
        max_descend_speed = abs(min(x["vy"] for x in trajectory))
        max_abs_angle = max(abs(x["angle"]) for x in trajectory)
        max_abs_ang_vel = max(abs(x["angular_velocity"]) for x in trajectory)

        print("-" * 80)
        print(f"Main Engine Ratio  : {main_engine_ratio:.4f}")
        print(f"Side Engine Ratio  : {side_engine_ratio:.4f}")
        print(f"Action Switch Rate : {switch_rate:.4f}")
        print(f"Max |x|            : {max_abs_x:.4f}")
        print(f"Max |vx|           : {max_abs_vx:.4f}")
        print(f"Max Descend Speed  : {max_descend_speed:.4f}")
        print(f"Max |angle|        : {max_abs_angle:.4f}")
        print(f"Max |ang_vel|      : {max_abs_ang_vel:.4f}")
    print("=" * 80)


# ==============================================================
# 加载模型
# ==============================================================
def load_model_and_obs_rms(model_path, log_dir, seed_for_temp_env=0):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    temp_game = LunarLanderGame(
        env_name="LunarLander-v3",
        seed=seed_for_temp_env,
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

    print("=" * 80)
    print(f"已加载模型: {model_path}")
    if "best_avg_reward" in checkpoint:
        print(f"训练阶段 best_avg_reward: {checkpoint['best_avg_reward']:.3f}")
    if "avg_reward" in checkpoint:
        print(f"训练阶段 avg_reward: {checkpoint['avg_reward']:.3f}")
    if "update_idx" in checkpoint:
        print(f"保存于 update_idx: {checkpoint['update_idx']}")
    print("=" * 80)

    return model, obs_rms


# ==============================================================
# 主函数
# ==============================================================
def main():
    # ==========================================================
    # 可调参数区
    # ==========================================================
    model_path = "./ppo_outputs/best_model.pt"

    replay_seeds = [10281799]

    # 是否开启环境画面可视化
    # True: 会弹出环境窗口
    # False: 只跑逻辑，不显示环境
    render = True

    # 是否保存每个 seed 的逐步轨迹 CSV
    save_csv = True

    # 是否保存每个 seed 的轨迹图
    save_figure = True

    # 是否打印最后 N 步详细信息
    print_last_n_steps = 50

    # 输出目录
    log_dir = "./badcase_replay_logs"
    # ==========================================================

    os.makedirs(log_dir, exist_ok=True)

    model, obs_rms = load_model_and_obs_rms(
        model_path=model_path,
        log_dir=log_dir,
        seed_for_temp_env=0
    )

    for seed in replay_seeds:
        print(f"\n开始回放 seed = {seed}")

        summary, trajectory = replay_one_seed(
            model=model,
            obs_rms=obs_rms,
            seed=seed,
            render=render,
            log_dir=log_dir,
        )

        print_summary(summary, trajectory)

        if print_last_n_steps is not None and print_last_n_steps > 0:
            print_last_steps(trajectory, last_n=print_last_n_steps)

        seed_dir = os.path.join(log_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        if save_csv:
            csv_path = os.path.join(seed_dir, f"seed_{seed}_trajectory.csv")
            save_trajectory_csv(trajectory, csv_path)
            print(f"轨迹 CSV 已保存: {csv_path}")

        if save_figure:
            fig_path = os.path.join(seed_dir, f"seed_{seed}_trajectory.png")
            plot_trajectory(trajectory, summary, fig_path)
            print(f"轨迹图已保存: {fig_path}")


if __name__ == "__main__":
    main()
