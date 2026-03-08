# badcase2.py
# ==============================================================
# PPO / LunarLander bad case 定位脚本
# --------------------------------------------------------------
# 功能说明：
# 1. 加载训练好的 PPO 模型（best_model.pt 或 latest_model.pt）
# 2. 按固定种子列表或随机种子列表批量回测
# 3. 对每个 episode 记录完整统计特征
# 4. 基于启发式规则自动标注 bad case 类型
# 5. 输出：
#    - 每局详细结果 CSV
#    - bad case 分类汇总 CSV
#    - 各类 bad case 的种子池 CSV
#    - 可视化图像 PNG
#
# 说明：
# - 本脚本只负责“定位 bad case”，不负责“根据 bad case 做训练修复”
# - 代码尽量与现有 evaluate / test 脚本保持风格一致
# - 默认假设 LunarLander 状态为 8 维：
#     [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
# - 若你自己的 LunarLanderGame 对 success / crash / timeout 有额外字段，
#   本脚本会优先读取；否则会根据 reward / 状态做一定兜底判断
# ==============================================================

import os
import csv
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from game import LunarLanderGame


# ==============================================================
# 1) 模型定义
#    必须与训练时结构保持一致，否则无法正确加载 checkpoint
# ==============================================================
class model_v1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(model_v1, self).__init__()

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor：输出动作 logits
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Critic：输出状态价值
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
        确定性动作：
        评估 / 回测时通常不再采样，而是直接选 argmax 动作，
        这样结果更稳定，更容易复现 bad case。
        """
        action_logits, state_value = self.forward(state)
        action = torch.argmax(action_logits, dim=-1)
        return action.item(), state_value.item()


# ==============================================================
# 2) 与训练代码保持一致的状态归一化工具
# ==============================================================
class RunningMeanStd:
    """
    用于加载训练阶段保存的 obs_rms 均值 / 方差 / count，
    这里不再在线更新，只做推理归一化。
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


def normalize_obs(obs, obs_rms, clip_range=10.0):
    """
    与训练代码保持一致的状态归一化逻辑
    """
    obs = np.asarray(obs, dtype=np.float32)
    obs_norm = (obs - obs_rms.mean) / obs_rms.std
    obs_norm = np.clip(obs_norm, -clip_range, clip_range)
    return obs_norm.astype(np.float32)


# ==============================================================
# 3) 基础工具函数
# ==============================================================
def safe_get_attr(obj, name, default=None):
    """
    安全读取对象字段，防止某些 result 不含指定属性时报错
    """
    return getattr(obj, name, default)


def set_global_seed(seed):
    """
    固定 python / numpy / torch 的随机种子，保证 episode 可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(data, window=10):
    """
    简单移动平均，用于画图平滑
    """
    if len(data) == 0:
        return []
    ma = []
    for i in range(len(data)):
        left = max(0, i - window + 1)
        ma.append(sum(data[left:i + 1]) / (i - left + 1))
    return ma


# ==============================================================
# 4) bad case 分类规则
# --------------------------------------------------------------
# 这里只做“定位”，所以使用启发式规则即可。
#
# LunarLander 离散动作通常是：
#   0: do nothing
#   1: fire left orientation engine
#   2: fire main engine
#   3: fire right orientation engine
#
# 状态默认：
#   state[0] = x
#   state[1] = y
#   state[2] = vx
#   state[3] = vy
#   state[4] = angle
#   state[5] = angular_velocity
#   state[6] = left_leg_contact
#   state[7] = right_leg_contact
#
# 注：
# - 阈值不是“理论唯一正确值”，而是为了工程定位 bad case 的实用值
# - 后续如果你观察到分类不准，可以继续调这些阈值
# ==============================================================

def analyze_episode_badcases(episode_trace, final_info):
    """
    根据一整个 episode 的轨迹，对 bad case 做规则标注。

    参数：
        episode_trace: dict
            包含本局所有状态 / 动作 / reward / value 等轨迹信息
        final_info: dict
            包含 success / crash / timeout / reward / steps 等收尾信息

    返回：
        analysis: dict
            统计特征 + bad case 标签
    """
    states = episode_trace["raw_states"]         # 每一步执行动作前的原始状态
    next_states = episode_trace["raw_next_states"]
    actions = episode_trace["actions"]
    rewards = episode_trace["rewards"]
    values = episode_trace["values"]

    # 防御性处理：避免空轨迹
    if len(states) == 0:
        return {
            "category_main": "empty_episode",
            "category_list": ["empty_episode"],
            "max_abs_x": 0.0,
            "final_abs_x": 0.0,
            "max_abs_vx": 0.0,
            "max_descend_speed": 0.0,
            "max_abs_angle": 0.0,
            "max_abs_ang_vel": 0.0,
            "main_engine_ratio": 0.0,
            "side_engine_ratio": 0.0,
            "noop_ratio": 0.0,
            "action_switch_rate": 0.0,
            "near_ground_steps": 0,
            "leg_touch_steps": 0,
            "late_stage_fail": 0,
            "hard_landing": 0,
            "large_drift": 0,
            "unstable_attitude": 0,
            "oscillating_control": 0,
            "fuel_waste": 0,
            "early_crash": 0,
        }

    # --------------------------
    # 把状态拆开，便于统计
    # --------------------------
    xs = [float(s[0]) for s in states]
    ys = [float(s[1]) for s in states]
    vxs = [float(s[2]) for s in states]
    vys = [float(s[3]) for s in states]
    angles = [float(s[4]) for s in states]
    ang_vels = [float(s[5]) for s in states]

    left_legs = [float(s[6]) for s in states] if len(
        states[0]) > 6 else [0.0] * len(states)
    right_legs = [float(s[7]) for s in states] if len(
        states[0]) > 7 else [0.0] * len(states)

    final_state = next_states[-1] if len(next_states) > 0 else states[-1]
    final_x = float(final_state[0])
    final_y = float(final_state[1])
    final_vx = float(final_state[2])
    final_vy = float(final_state[3])
    final_angle = float(final_state[4])
    final_ang_vel = float(final_state[5])
    final_left_leg = float(final_state[6]) if len(final_state) > 6 else 0.0
    final_right_leg = float(final_state[7]) if len(final_state) > 7 else 0.0

    # --------------------------
    # 轨迹统计特征
    # --------------------------
    max_abs_x = max(abs(x) for x in xs)
    final_abs_x = abs(final_x)

    max_abs_vx = max(abs(vx) for vx in vxs)
    max_abs_vy = max(abs(vy) for vy in vys)

    # 下降速度：vy 往往向下为负，这里取“最危险的下降速度大小”
    # 即 min(vy) 越小，下降越猛，所以取 abs(min(vy))
    max_descend_speed = abs(min(vys))

    max_abs_angle = max(abs(a) for a in angles)
    max_abs_ang_vel = max(abs(w) for w in ang_vels)

    # 动作统计
    total_steps = max(1, len(actions))
    main_engine_steps = sum(1 for a in actions if a == 2)
    side_engine_steps = sum(1 for a in actions if a in [1, 3])
    noop_steps = sum(1 for a in actions if a == 0)

    main_engine_ratio = main_engine_steps / total_steps
    side_engine_ratio = side_engine_steps / total_steps
    noop_ratio = noop_steps / total_steps

    # 动作切换率：切换过于频繁，通常说明控制抖动 / 震荡
    if len(actions) >= 2:
        action_switches = sum(1 for i in range(
            1, len(actions)) if actions[i] != actions[i - 1])
        action_switch_rate = action_switches / (len(actions) - 1)
    else:
        action_switch_rate = 0.0

    # 近地阶段统计：
    # y 较低时视为近地阶段，用于判断“最后阶段崩掉”
    near_ground_threshold = 0.35
    near_ground_indices = [i for i, y in enumerate(
        ys) if y <= near_ground_threshold]
    near_ground_steps = len(near_ground_indices)

    # 腿接触次数
    leg_touch_steps = sum(
        1 for l, r in zip(left_legs, right_legs) if (l > 0.5 or r > 0.5)
    )

    # 是否发生“接近成功后仍失败”
    # 经验判断：
    # - 出现过腿接触 或 曾明显接近地面
    # - 最终不是 success
    late_stage_fail = int(
        (leg_touch_steps > 0 or near_ground_steps >= 10) and (
            final_info["success"] == 0)
    )

    # --------------------------
    # 启发式 bad case 规则
    # --------------------------
    # 下面的阈值可根据你的具体环境继续微调
    # 这里只求“便于定位问题”，不追求完美分类
    hard_landing = int(
        (final_info["crash"] == 1) and (
            final_vy < -0.80 or
            abs(final_vx) > 0.90
        )
    )

    large_drift = int(
        max_abs_x > 1.00 or
        final_abs_x > 0.60 or
        max_abs_vx > 1.20
    )

    unstable_attitude = int(
        max_abs_angle > 0.45 or
        abs(final_angle) > 0.35 or
        max_abs_ang_vel > 2.50
    )

    oscillating_control = int(
        action_switch_rate > 0.55 and side_engine_ratio > 0.18
    )

    fuel_waste = int(
        main_engine_ratio > 0.42 and final_info["success"] == 0
    )

    early_crash = int(
        final_info["crash"] == 1 and final_info["steps"] < 120
    )

    timeout_hovering = int(
        final_info["timeout"] == 1 and main_engine_ratio > 0.20
    )

    poor_terminal_control = int(
        late_stage_fail == 1 and (
            abs(final_vy) > 0.60 or
            abs(final_angle) > 0.25 or
            abs(final_vx) > 0.60
        )
    )

    # --------------------------
    # 分类标签集合
    # --------------------------
    category_list = []

    # 先根据最终结果粗分
    if final_info["success"] == 1:
        category_list.append("success")
    elif final_info["crash"] == 1:
        category_list.append("crash")
    elif final_info["timeout"] == 1:
        category_list.append("timeout")
    else:
        category_list.append("other_fail")

    # 再叠加具体 failure mode
    if hard_landing:
        category_list.append("hard_landing")

    if large_drift:
        category_list.append("large_drift")

    if unstable_attitude:
        category_list.append("unstable_attitude")

    if oscillating_control:
        category_list.append("oscillating_control")

    if fuel_waste:
        category_list.append("fuel_waste")

    if early_crash:
        category_list.append("early_crash")

    if timeout_hovering:
        category_list.append("timeout_hovering")

    if poor_terminal_control:
        category_list.append("poor_terminal_control")

    if late_stage_fail:
        category_list.append("late_stage_fail")

    # --------------------------
    # 主类别：
    # 便于后续做“每个 episode 只有一个主 bad case”的统计
    # 优先级：越能直接解释失败原因的标签优先级越高
    # --------------------------
    if final_info["success"] == 1:
        category_main = "success"
    else:
        priority_order = [
            "poor_terminal_control",
            "hard_landing",
            "large_drift",
            "unstable_attitude",
            "oscillating_control",
            "timeout_hovering",
            "fuel_waste",
            "early_crash",
            "late_stage_fail",
            "crash",
            "timeout",
            "other_fail",
        ]

        category_main = "other_fail"
        for tag in priority_order:
            if tag in category_list:
                category_main = tag
                break

    analysis = {
        "category_main": category_main,
        "category_list": category_list,

        # 轨迹统计特征
        "max_abs_x": float(max_abs_x),
        "final_abs_x": float(final_abs_x),
        "max_abs_vx": float(max_abs_vx),
        "max_abs_vy": float(max_abs_vy),
        "max_descend_speed": float(max_descend_speed),
        "max_abs_angle": float(max_abs_angle),
        "max_abs_ang_vel": float(max_abs_ang_vel),
        "main_engine_ratio": float(main_engine_ratio),
        "side_engine_ratio": float(side_engine_ratio),
        "noop_ratio": float(noop_ratio),
        "action_switch_rate": float(action_switch_rate),
        "near_ground_steps": int(near_ground_steps),
        "leg_touch_steps": int(leg_touch_steps),

        # 终局状态特征
        "final_x": float(final_x),
        "final_y": float(final_y),
        "final_vx": float(final_vx),
        "final_vy": float(final_vy),
        "final_angle": float(final_angle),
        "final_ang_vel": float(final_ang_vel),
        "final_left_leg": int(final_left_leg > 0.5),
        "final_right_leg": int(final_right_leg > 0.5),

        # bad case 二值标签
        "late_stage_fail": int(late_stage_fail),
        "hard_landing": int(hard_landing),
        "large_drift": int(large_drift),
        "unstable_attitude": int(unstable_attitude),
        "oscillating_control": int(oscillating_control),
        "fuel_waste": int(fuel_waste),
        "early_crash": int(early_crash),
        "timeout_hovering": int(timeout_hovering),
        "poor_terminal_control": int(poor_terminal_control),
    }

    return analysis


# ==============================================================
# 5) 单局运行并采集轨迹
# --------------------------------------------------------------
# 重点：
# - 每个 episode 单独新建环境，确保 seed 真正生效
# - 记录 raw state / normalized state / action / value / reward
# - episode 结束后做 bad case 分析
# ==============================================================

def run_one_episode_with_trace(model, obs_rms, seed, render=False, log_dir="./badcase_logs"):
    """
    运行一局，并返回：
    1. 本局基础结果
    2. 本局详细统计
    3. 本局 bad case 标签
    4. 必要轨迹（仅保存在内存中，不默认全量落盘）
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

    episode_reward = 0.0
    step_count = 0

    # 终局标志优先从环境结果字段读取
    last_success = 0
    last_crash = 0
    last_timeout = 0

    # 轨迹缓存
    episode_trace = {
        "raw_states": [],
        "norm_states": [],
        "raw_next_states": [],
        "actions": [],
        "rewards": [],
        "values": [],
    }

    with torch.no_grad():
        while not done:
            state_tensor = torch.tensor(norm_state, dtype=torch.float32)
            action, state_value = model.act_deterministic(state_tensor)

            result = game.step(action)

            raw_next_state = result.state
            reward = float(result.reward)
            done = bool(result.done)

            # 保存轨迹
            episode_trace["raw_states"].append(
                np.asarray(raw_state, dtype=np.float32).tolist())
            episode_trace["norm_states"].append(
                np.asarray(norm_state, dtype=np.float32).tolist())
            episode_trace["raw_next_states"].append(
                np.asarray(raw_next_state, dtype=np.float32).tolist())
            episode_trace["actions"].append(int(action))
            episode_trace["rewards"].append(float(reward))
            episode_trace["values"].append(float(state_value))

            # 更新计数
            episode_reward += reward
            step_count += 1

            # 更新结果字段
            last_success = safe_get_attr(result, "success", last_success)
            last_crash = safe_get_attr(result, "crash", last_crash)
            last_timeout = safe_get_attr(result, "timeout", last_timeout)

            # 进入下一步
            raw_state = raw_next_state
            norm_state = normalize_obs(raw_state, obs_rms)

    # 如果环境支持 close，则关闭
    if hasattr(game, "close") and callable(getattr(game, "close")):
        game.close()

    final_info = {
        "seed": int(seed),
        "reward": float(episode_reward),
        "steps": int(step_count),
        "success": int(bool(last_success)),
        "crash": int(bool(last_crash)),
        "timeout": int(bool(last_timeout)),
    }

    # 调用规则分析器，对本局做 bad case 标注
    analysis = analyze_episode_badcases(
        episode_trace=episode_trace,
        final_info=final_info
    )

    # 整合结果，形成“每局一条记录”
    episode_result = {
        "seed": final_info["seed"],
        "reward": final_info["reward"],
        "steps": final_info["steps"],
        "success": final_info["success"],
        "crash": final_info["crash"],
        "timeout": final_info["timeout"],

        "category_main": analysis["category_main"],
        "category_list": "|".join(analysis["category_list"]),

        "max_abs_x": analysis["max_abs_x"],
        "final_abs_x": analysis["final_abs_x"],
        "max_abs_vx": analysis["max_abs_vx"],
        "max_abs_vy": analysis["max_abs_vy"],
        "max_descend_speed": analysis["max_descend_speed"],
        "max_abs_angle": analysis["max_abs_angle"],
        "max_abs_ang_vel": analysis["max_abs_ang_vel"],
        "main_engine_ratio": analysis["main_engine_ratio"],
        "side_engine_ratio": analysis["side_engine_ratio"],
        "noop_ratio": analysis["noop_ratio"],
        "action_switch_rate": analysis["action_switch_rate"],
        "near_ground_steps": analysis["near_ground_steps"],
        "leg_touch_steps": analysis["leg_touch_steps"],

        "final_x": analysis["final_x"],
        "final_y": analysis["final_y"],
        "final_vx": analysis["final_vx"],
        "final_vy": analysis["final_vy"],
        "final_angle": analysis["final_angle"],
        "final_ang_vel": analysis["final_ang_vel"],
        "final_left_leg": analysis["final_left_leg"],
        "final_right_leg": analysis["final_right_leg"],

        "late_stage_fail": analysis["late_stage_fail"],
        "hard_landing": analysis["hard_landing"],
        "large_drift": analysis["large_drift"],
        "unstable_attitude": analysis["unstable_attitude"],
        "oscillating_control": analysis["oscillating_control"],
        "fuel_waste": analysis["fuel_waste"],
        "early_crash": analysis["early_crash"],
        "timeout_hovering": analysis["timeout_hovering"],
        "poor_terminal_control": analysis["poor_terminal_control"],
    }

    return episode_result, episode_trace


# ==============================================================
# 6) 保存详细 CSV
# ==============================================================

def save_episode_details_csv(results, save_path):
    """
    保存每个 episode 的详细结果表
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "episode", "seed", "reward", "steps", "success", "crash", "timeout",
        "category_main", "category_list",

        "max_abs_x", "final_abs_x",
        "max_abs_vx", "max_abs_vy", "max_descend_speed",
        "max_abs_angle", "max_abs_ang_vel",
        "main_engine_ratio", "side_engine_ratio", "noop_ratio",
        "action_switch_rate", "near_ground_steps", "leg_touch_steps",

        "final_x", "final_y", "final_vx", "final_vy",
        "final_angle", "final_ang_vel", "final_left_leg", "final_right_leg",

        "late_stage_fail",
        "hard_landing",
        "large_drift",
        "unstable_attitude",
        "oscillating_control",
        "fuel_waste",
        "early_crash",
        "timeout_hovering",
        "poor_terminal_control",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(results, start=1):
            row = {"episode": idx}
            for key in fieldnames:
                if key == "episode":
                    continue
                row[key] = item.get(key, "")
            writer.writerow(row)


def save_category_summary_csv(results, save_path):
    """
    保存 bad case 主类别统计汇总

    输出字段：
    - category_main
    - count
    - ratio
    - avg_reward
    - avg_steps
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    total = max(1, len(results))
    by_category = defaultdict(list)

    for item in results:
        by_category[item["category_main"]].append(item)

    rows = []
    for cat, items in by_category.items():
        avg_reward = float(np.mean([x["reward"]
                           for x in items])) if items else 0.0
        avg_steps = float(np.mean([x["steps"]
                          for x in items])) if items else 0.0
        rows.append({
            "category_main": cat,
            "count": len(items),
            "ratio": len(items) / total,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        })

    rows.sort(key=lambda x: x["count"], reverse=True)

    fieldnames = ["category_main", "count", "ratio", "avg_reward", "avg_steps"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_badcase_seed_pool_csv(results, save_path):
    """
    为 bad case 定位准备“种子池”。

    含义：
    - 把所有非 success 的 episode 按主类别保存
    - 后续你可以直接抽这些 seed 做专项复现 / 可视化 / 调试

    输出字段：
    - seed
    - category_main
    - reward
    - steps
    - crash
    - timeout
    - category_list
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = ["seed", "category_main", "reward",
                  "steps", "crash", "timeout", "category_list"]
    rows = []

    for item in results:
        if item["category_main"] != "success":
            rows.append({
                "seed": item["seed"],
                "category_main": item["category_main"],
                "reward": item["reward"],
                "steps": item["steps"],
                "crash": item["crash"],
                "timeout": item["timeout"],
                "category_list": item["category_list"],
            })

    rows.sort(key=lambda x: (x["category_main"], x["reward"]))

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_top_badcase_examples_json(results, save_path, top_k_per_type=20):
    """
    保存各类 bad case 的“代表性 seed”
    规则：
    - 每个主类别按 reward 从低到高排序
    - 取最差 top-k

    这样后续做人工复现会比较方便。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grouped = defaultdict(list)
    for item in results:
        grouped[item["category_main"]].append(item)

    output = {}
    for cat, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x["reward"])
        output[cat] = [
            {
                "seed": int(x["seed"]),
                "reward": float(x["reward"]),
                "steps": int(x["steps"]),
                "crash": int(x["crash"]),
                "timeout": int(x["timeout"]),
                "category_list": x["category_list"],
            }
            for x in items_sorted[:top_k_per_type]
        ]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ==============================================================
# 7) 可视化
# ==============================================================

def plot_badcase_summary(results, save_path):
    """
    画一张总览图，帮助快速看出：
    - reward 分布
    - 主类别分布
    - 失败样本中的常见问题
    - steps vs reward
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rewards = [x["reward"] for x in results]
    steps = [x["steps"] for x in results]
    reward_ma = moving_average(rewards, window=10)

    # 主类别统计
    category_counter = Counter([x["category_main"] for x in results])
    category_names = [k for k, _ in category_counter.most_common()]
    category_counts = [category_counter[k] for k in category_names]

    # 二值 bad case 标签统计
    tag_names = [
        "hard_landing",
        "large_drift",
        "unstable_attitude",
        "oscillating_control",
        "fuel_waste",
        "early_crash",
        "timeout_hovering",
        "poor_terminal_control",
        "late_stage_fail",
    ]
    tag_counts = [sum(int(x[tag]) for x in results) for tag in tag_names]

    plt.figure(figsize=(18, 12))

    # 1) reward 曲线
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(results) + 1), rewards, marker="o", linewidth=1)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # 2) reward 移动平均
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(results) + 1), reward_ma, marker="o", linewidth=1)
    plt.title("Reward Moving Average (window=10)")
    plt.xlabel("Episode")
    plt.ylabel("MA Reward")
    plt.grid(True, alpha=0.3)

    # 3) reward 直方图
    plt.subplot(2, 3, 3)
    bins = min(30, max(10, len(results) // 10))
    plt.hist(rewards, bins=bins)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    # 4) 主类别柱状图
    plt.subplot(2, 3, 4)
    plt.bar(category_names, category_counts)
    plt.title("Main Bad Case Category Counts")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")

    # 5) bad case 标签统计
    plt.subplot(2, 3, 5)
    plt.bar(tag_names, tag_counts)
    plt.title("Bad Case Tag Counts")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")

    # 6) steps vs reward
    plt.subplot(2, 3, 6)
    plt.scatter(steps, rewards)
    plt.title("Steps vs Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# ==============================================================
# 8) 控制台总结输出
# ==============================================================

def print_summary(results):
    """
    在控制台打印整体统计结果，便于快速判断：
    - 平均 reward
    - success / crash / timeout
    - 主 bad case 分布
    """
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

    print("\n" + "=" * 80)
    print("bad case 定位完成，统计结果如下：")
    print(f"测试局数              : {len(results)}")
    print(f"平均奖励              : {avg_reward:.2f}")
    print(f"奖励标准差            : {std_reward:.2f}")
    print(f"最高奖励              : {max_reward:.2f}")
    print(f"最低奖励              : {min_reward:.2f}")
    print(f"平均步数              : {avg_steps:.2f}")
    print(f"成功次数              : {success_count}")
    print(f"坠毁次数              : {crash_count}")
    print(f"超时次数              : {timeout_count}")
    print(
        f"成功率                : {success_count / max(1, len(results)) * 100:.2f}%")
    print(
        f"坠毁率                : {crash_count / max(1, len(results)) * 100:.2f}%")
    print(
        f"超时率                : {timeout_count / max(1, len(results)) * 100:.2f}%")
    print("-" * 80)

    # 主类别统计
    category_counter = Counter([x["category_main"] for x in results])
    print("主 bad case 类别分布：")
    for cat, cnt in category_counter.most_common():
        print(f"  {cat:<24s}: {cnt:5d} ({cnt / max(1, len(results)) * 100:.2f}%)")

    print("-" * 80)

    # 二值标签统计
    tag_names = [
        "hard_landing",
        "large_drift",
        "unstable_attitude",
        "oscillating_control",
        "fuel_waste",
        "early_crash",
        "timeout_hovering",
        "poor_terminal_control",
        "late_stage_fail",
    ]
    print("bad case 标签命中次数：")
    for tag in tag_names:
        cnt = sum(int(x[tag]) for x in results)
        print(f"  {tag:<24s}: {cnt:5d}")

    print("=" * 80)


# ==============================================================
# 9) 主函数
# ==============================================================

def main():
    # ==========================================================
    # 可调参数区
    # ==========================================================
    model_path = "./ppo_outputs/best_model.pt"

    # 回测局数：
    # - 想快速看结果：100 ~ 200
    # - 想更稳定定位 bad case：500 ~ 2000
    test_episodes = 1000

    # render 建议批量跑时设为 False
    render = False

    # 日志目录
    log_dir = "./badcase_logs"

    # 是否使用固定种子列表
    use_fixed_seed_list = True
    base_seed = 42

    # 是否保存“最差案例”的完整轨迹 JSON
    # 注意：如果局数很多，轨迹文件会比较大
    save_worst_trace_json = True

    # 每个主 bad case 类别保存多少个“最差案例”
    worst_trace_top_k = 5
    # ==========================================================

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    os.makedirs(log_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 先建一个临时环境，读取 state_dim / action_dim
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 加载模型
    # ----------------------------------------------------------
    model = model_v1(state_dim, action_dim)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ----------------------------------------------------------
    # 恢复训练时保存的 obs normalization 参数
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 生成种子列表
    # ----------------------------------------------------------
    if use_fixed_seed_list:
        # 固定种子列表：适合做 bad case 回归定位
        episode_seeds = [base_seed + i for i in range(test_episodes)]
    else:
        # 随机种子列表：适合做广泛随机评估
        rng = random.Random()
        episode_seeds = [rng.randint(0, 10**9) for _ in range(test_episodes)]

    results = []
    traces = []

    # ----------------------------------------------------------
    # 开始逐局测试
    # ----------------------------------------------------------
    for episode_idx, seed in enumerate(episode_seeds, start=1):
        episode_result, episode_trace = run_one_episode_with_trace(
            model=model,
            obs_rms=obs_rms,
            seed=seed,
            render=render,
            log_dir=log_dir
        )

        results.append(episode_result)
        traces.append({
            "seed": int(seed),
            "category_main": episode_result["category_main"],
            "reward": float(episode_result["reward"]),
            "steps": int(episode_result["steps"]),
            "trace": episode_trace
        })

        print(
            f"[BadCase Test {episode_idx:04d}/{test_episodes}] "
            f"Seed: {seed:<10d} | "
            f"Reward: {episode_result['reward']:8.2f} | "
            f"Steps: {episode_result['steps']:4d} | "
            f"Success: {episode_result['success']} | "
            f"Crash: {episode_result['crash']} | "
            f"Timeout: {episode_result['timeout']} | "
            f"MainType: {episode_result['category_main']}"
        )

    # ----------------------------------------------------------
    # 保存结果文件
    # ----------------------------------------------------------
    details_csv_path = os.path.join(log_dir, "badcase_episode_details.csv")
    summary_csv_path = os.path.join(log_dir, "badcase_category_summary.csv")
    seed_pool_csv_path = os.path.join(log_dir, "badcase_seed_pool.csv")
    examples_json_path = os.path.join(log_dir, "badcase_top_examples.json")
    figure_path = os.path.join(log_dir, "badcase_summary.png")

    save_episode_details_csv(results, details_csv_path)
    save_category_summary_csv(results, summary_csv_path)
    save_badcase_seed_pool_csv(results, seed_pool_csv_path)
    save_top_badcase_examples_json(
        results, examples_json_path, top_k_per_type=20)
    plot_badcase_summary(results, figure_path)
    print_summary(results)

    # ----------------------------------------------------------
    # 额外保存：每个主 bad case 类别中“最差的几个 episode”的完整轨迹
    # 方便你后续精准复现
    # ----------------------------------------------------------
    if save_worst_trace_json:
        grouped = defaultdict(list)
        for item in traces:
            grouped[item["category_main"]].append(item)

        worst_trace_dir = os.path.join(log_dir, "worst_traces")
        os.makedirs(worst_trace_dir, exist_ok=True)

        for cat, items in grouped.items():
            # success 不一定需要存轨迹，但这里保留，便于对照
            items_sorted = sorted(items, key=lambda x: x["reward"])

            # 每类取 reward 最差的 top-k
            selected = items_sorted[:worst_trace_top_k]

            save_path = os.path.join(
                worst_trace_dir, f"{cat}_worst_traces.json")
            payload = []
            for x in selected:
                payload.append({
                    "seed": x["seed"],
                    "category_main": x["category_main"],
                    "reward": x["reward"],
                    "steps": x["steps"],
                    "trace": x["trace"],
                })

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n输出文件如下：")
    print(f"1) 每局详细结果 CSV     : {details_csv_path}")
    print(f"2) bad case 汇总 CSV    : {summary_csv_path}")
    print(f"3) bad case 种子池 CSV  : {seed_pool_csv_path}")
    print(f"4) 代表案例 JSON        : {examples_json_path}")
    print(f"5) 总览图 PNG           : {figure_path}")
    if save_worst_trace_json:
        print(f"6) 最差案例轨迹目录     : {os.path.join(log_dir, 'worst_traces')}")


if __name__ == "__main__":
    main()
