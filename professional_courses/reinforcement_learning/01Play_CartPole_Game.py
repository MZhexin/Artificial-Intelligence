# Play CartPole Game
# 参考教程：https://www.bilibili.com/video/BV1TQ4y1v7H5

import gym

def run():
    env = gym.make('CartPole-v1', render_mode="rgb_array")       # make函数返回环境environment

    state = env.reset()     # 重置环境

    for t in range(100):
        env.render()        # 显示游戏的窗口
        print(state)

        action = env.action_space.sample()          # 这里的action是均匀抽样的（实际应用中很少有人这么干）
        step_result = env.step(action)
        done = step_result[2]     # 返回游戏状态、奖励、游戏是否结束和信息

        if done:
            print('Finished')
            break

    env.close()

if __name__ == '__main__':
    run()