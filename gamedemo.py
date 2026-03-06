"""
game_api_example.py

LunarLanderGame 简易接口文档（示例脚本）
-------------------------------------

本文件展示如何使用 game.py 中封装的 LunarLanderGame。
所有说明通过 Python 注释给出，可以直接运行测试。

目标：
训练代码只需要关注

    state -> action -> next_state

环境交互、日志记录、episode统计、action导出都由 game.py 处理。

依赖：
    pip install gymnasium[box2d]
"""

# ==============================
# 1. 导入接口
# ==============================

from game import LunarLanderGame


# ==============================
# 2. 创建游戏环境
# ==============================

# log_dir:
#   训练日志保存目录
# save_step_trace:
#   是否记录每一步的详细状态（调试时可以开启）

game = LunarLanderGame(
    env_name="LunarLander-v3",
    seed=42,
    render_mode="human",
    log_dir="./logs",
    save_step_trace=False
)


# ==============================
# 3. 获取环境信息
# ==============================

# 状态维度（LunarLander = 8）
state_dim = game.get_state_dim()

# 动作数量（LunarLander = 4）
action_dim = game.get_action_dim()

print("state_dim:", state_dim)
print("action_dim:", action_dim)


# ==============================
# 4. 开始一个 episode
# ==============================

# reset() 会：
#   1. 初始化环境
#   2. 返回初始状态

state = game.reset()


# ==============================
# 5. 进行环境交互
# ==============================

# step(action) 返回 StepResult 对象：
#
# result.state       -> 下一状态
# result.reward      -> 当前奖励
# result.done        -> episode 是否结束
# result.truncated   -> 是否超时结束
# result.info        -> 环境额外信息

done = False

while not done:

    # 示例：随机动作
    # 实际训练时这里应替换为 agent.select_action(state)

    action = game.sample_random_action()

    result = game.step(action)

    next_state = result.state
    reward = result.reward
    done = result.done

    # 更新状态
    state = next_state


# ==============================
# 6. 获取 episode 统计信息
# ==============================

stats = game.get_last_episode_stats()

print("episode reward:", stats.total_reward)
print("episode steps:", stats.step_count)


# ==============================
# 7. 保存动作序列
# ==============================

# 默认保存当前 episode 的 action list
game.save_action_list("./logs/action_list.txt")


# ==============================
# 8. 训练日志记录（可选）
# ==============================

# 可以在训练脚本中记录 loss / reward 等信息

game.log_training_info(
    episode_idx=0,
    loss=0.123,
    avg_reward=stats.total_reward
)


# ==============================
# 9. 关闭环境
# ==============================

game.close()


"""
训练代码推荐结构
----------------

for episode in range(N):

    state = game.reset()
    done = False

    while not done:

        action = agent.select_action(state)

        result = game.step(action)

        next_state = result.state
        reward = result.reward
        done = result.done

        agent.store_transition(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.update()

        state = next_state

    stats = game.get_last_episode_stats()

    game.log_training_info(
        episode_idx=episode,
        avg_reward=stats.total_reward
    )

"""
