import os
import csv
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import gymnasium as gym
import numpy as np


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    # 新增：终局标签，非终局时默认为 0
    success: int = 0
    crash: int = 0
    timeout: int = 0


@dataclass
class EpisodeStats:
    episode_idx: int
    total_reward: float
    step_count: int
    success: int
    crash: int
    timeout: int
    elapsed_time_sec: float


class LunarLanderGame:
    """
    LunarLander 环境封装层：
    1. 统一 reset / step 接口
    2. 自动日志记录
    3. 自动保存 episode 统计信息
    4. 可导出 action list
    5. 训练代码只需关心 state -> action -> result
    """

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        seed: Optional[int] = 42,
        render_mode: Optional[str] = None,
        log_dir: str = "./logs",
        save_step_trace: bool = False,
        max_action_trace_len: int = 5000,
    ):
        self.env_name = env_name
        self.seed = seed
        self.render_mode = render_mode
        self.log_dir = log_dir
        self.save_step_trace = save_step_trace
        self.max_action_trace_len = max_action_trace_len

        os.makedirs(self.log_dir, exist_ok=True)

        self.env = gym.make(self.env_name, render_mode=self.render_mode)

        self.state_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = self.env.action_space.n

        self.current_state: Optional[np.ndarray] = None
        self.current_episode_idx: int = 0
        self.current_episode_reward: float = 0.0
        self.current_episode_steps: int = 0
        self.current_episode_actions: List[int] = []
        self.current_episode_start_time: Optional[float] = None

        self.last_episode_stats: Optional[EpisodeStats] = None
        self.best_reward: float = -float("inf")

        self._init_logger()
        self._init_csv_files()

        self.logger.info("LunarLanderGame initialized.")
        self.logger.info(f"env_name={self.env_name}")
        self.logger.info(
            f"state_dim={self.state_dim}, action_dim={self.action_dim}")
        self.logger.info(f"seed={self.seed}, render_mode={self.render_mode}")

    def _init_logger(self) -> None:
        self.logger = logging.getLogger(f"LunarLanderGame_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, "game.log"),
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _init_csv_files(self) -> None:
        self.episode_csv_path = os.path.join(self.log_dir, "episode_stats.csv")
        self.step_csv_path = os.path.join(self.log_dir, "step_trace.csv")

        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode_idx",
                    "total_reward",
                    "step_count",
                    "success",
                    "crash",
                    "timeout",
                    "elapsed_time_sec"
                ])

        if self.save_step_trace and not os.path.exists(self.step_csv_path):
            with open(self.step_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode_idx",
                    "step_idx",
                    "action",
                    "reward",
                    "done",
                    "terminated",
                    "truncated",
                    "success",
                    "crash",
                    "timeout",
                    "state_json"
                ])

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            state, info = self.env.reset(seed=seed)
        else:
            state, info = self.env.reset()

        self.current_state = np.asarray(state, dtype=np.float32)
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.current_episode_actions = []
        self.current_episode_start_time = time.time()

        self.logger.info(
            f"[Episode {self.current_episode_idx}] reset | seed={seed}"
        )
        return self.current_state.copy()

    def step(self, action: int) -> StepResult:
        if self.current_state is None:
            raise RuntimeError(
                "Environment has not been reset. Call reset() before step().")

        next_state, reward, terminated, truncated, info = self.env.step(
            int(action))
        next_state = np.asarray(next_state, dtype=np.float32)
        done = bool(terminated or truncated)

        self.current_episode_reward += float(reward)
        self.current_episode_steps += 1

        if len(self.current_episode_actions) < self.max_action_trace_len:
            self.current_episode_actions.append(int(action))

        self.current_state = next_state

        # 默认非终局标签
        success = 0
        crash = 0
        timeout = 0

        if done:
            stats = self._finalize_episode(terminated=bool(
                terminated), truncated=bool(truncated))
            success = stats.success
            crash = stats.crash
            timeout = stats.timeout

        if self.save_step_trace:
            self._append_step_trace(
                episode_idx=self.current_episode_idx if not done else self.current_episode_idx - 1,
                step_idx=self.current_episode_steps,
                action=int(action),
                reward=float(reward),
                done=done,
                terminated=bool(terminated),
                truncated=bool(truncated),
                success=success,
                crash=crash,
                timeout=timeout,
                state=next_state,
            )

        return StepResult(
            state=next_state.copy(),
            reward=float(reward),
            done=done,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            success=success,
            crash=crash,
            timeout=timeout,
        )

    def close(self) -> None:
        self.env.close()
        self.logger.info("Environment closed.")

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_current_state(self) -> np.ndarray:
        if self.current_state is None:
            raise RuntimeError("No current state. Call reset() first.")
        return self.current_state.copy()

    def sample_random_action(self) -> int:
        return int(self.env.action_space.sample())

    def get_last_episode_stats(self) -> Optional[EpisodeStats]:
        return self.last_episode_stats

    def get_current_episode_actions(self) -> List[int]:
        return list(self.current_episode_actions)

    def save_action_list(
        self,
        file_path: str,
        actions: Optional[List[int]] = None,
        as_json: bool = False
    ) -> None:
        if actions is None:
            actions = self.current_episode_actions

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        if as_json:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(actions, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                for a in actions:
                    f.write(f"{a}\n")

        self.logger.info(f"Action list saved to: {file_path}")

    def run_one_episode(
        self,
        policy_fn,
        seed: Optional[int] = None,
        return_trajectory: bool = True
    ) -> Dict[str, Any]:
        state = self.reset(seed=seed)
        done = False
        trajectory = []

        while not done:
            action = int(policy_fn(state))
            result = self.step(action)

            if return_trajectory:
                trajectory.append({
                    "state": state.tolist(),
                    "action": action,
                    "reward": result.reward,
                    "next_state": result.state.tolist(),
                    "done": result.done,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "success": result.success,
                    "crash": result.crash,
                    "timeout": result.timeout,
                })

            state = result.state
            done = result.done

        output = {
            "total_reward": self.last_episode_stats.total_reward if self.last_episode_stats else None,
            "actions": self.get_current_episode_actions(),
            "success": self.last_episode_stats.success if self.last_episode_stats else 0,
            "crash": self.last_episode_stats.crash if self.last_episode_stats else 0,
            "timeout": self.last_episode_stats.timeout if self.last_episode_stats else 0,
        }

        if return_trajectory:
            output["trajectory"] = trajectory

        return output

    def _append_step_trace(
        self,
        episode_idx: int,
        step_idx: int,
        action: int,
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        success: int,
        crash: int,
        timeout: int,
        state: np.ndarray,
    ) -> None:
        with open(self.step_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_idx,
                step_idx,
                action,
                reward,
                int(done),
                int(terminated),
                int(truncated),
                int(success),
                int(crash),
                int(timeout),
                json.dumps(state.tolist(), ensure_ascii=False),
            ])

    def _append_episode_stats(self, stats: EpisodeStats) -> None:
        with open(self.episode_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                stats.episode_idx,
                stats.total_reward,
                stats.step_count,
                stats.success,
                stats.crash,
                stats.timeout,
                stats.elapsed_time_sec,
            ])

    def _finalize_episode(self, terminated: bool, truncated: bool) -> EpisodeStats:
        elapsed = time.time() - \
            self.current_episode_start_time if self.current_episode_start_time else 0.0

        # 这里仍保留你的“实用型近似判断”
        timeout = int(bool(truncated))
        success = int((not truncated) and (
            self.current_episode_reward >= 200.0))
        crash = int((not truncated) and (success == 0))

        stats = EpisodeStats(
            episode_idx=self.current_episode_idx,
            total_reward=float(self.current_episode_reward),
            step_count=int(self.current_episode_steps),
            success=success,
            crash=crash,
            timeout=timeout,
            elapsed_time_sec=float(elapsed),
        )

        self.last_episode_stats = stats
        self._append_episode_stats(stats)

        if stats.total_reward > self.best_reward:
            self.best_reward = stats.total_reward
            self.logger.info(
                f"[Episode {stats.episode_idx}] New best reward: {stats.total_reward:.3f}"
            )

        self.logger.info(
            f"[Episode {stats.episode_idx}] finished | "
            f"reward={stats.total_reward:.3f} | "
            f"steps={stats.step_count} | "
            f"success={stats.success} | crash={stats.crash} | timeout={stats.timeout} | "
            f"time={stats.elapsed_time_sec:.2f}s"
        )

        self.current_episode_idx += 1
        return stats

    def log_training_info(
        self,
        episode_idx: int,
        loss: Optional[float] = None,
        avg_reward: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = f"[Train] episode={episode_idx}"
        if loss is not None:
            msg += f" | loss={loss:.6f}"
        if avg_reward is not None:
            msg += f" | avg_reward={avg_reward:.3f}"
        if extra is not None:
            for k, v in extra.items():
                msg += f" | {k}={v}"
        self.logger.info(msg)


if __name__ == "__main__":
    game = LunarLanderGame(
        env_name="LunarLander-v3",
        seed=42,
        render_mode=None,
        log_dir="./logs",
        save_step_trace=False,
    )

    state = game.reset()
    done = False

    while not done:
        action = game.sample_random_action()
        result = game.step(action)
        state = result.state
        done = result.done

    print("Last episode stats:")
    print(asdict(game.get_last_episode_stats()))
    print("Last step result:")
    print(result)

    game.save_action_list("./logs/random_action_list.txt")
    game.close()
