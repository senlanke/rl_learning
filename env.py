import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

# 临时修复numpy兼容性问题
# if not hasattr(np, 'bool8'):
#     np.bool8 = np.bool_

class MyWrapper(gym.Wrapper):
    def __init__(self):  
        env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
        super().__init__(env)
        # 注意：不需要再次赋值 self.env，因为父类已经处理了
    
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated
        
        if not over:
            reward = -1  # 每步的小惩罚
        elif over and reward == 0:  # 修复：检查是否是失败结束
            reward = -100  # 失败惩罚
        # 如果成功到达目标(reward=1)，保持原奖励
        
        return state, reward, over
    
    def show(self):
        plt.figure(figsize=(3, 3))
        plt.imshow(self.env.render())
        plt.axis('off')  # 隐藏坐标轴
        plt.show()  # 添加show()来显示图像

# 测试代码
if __name__ == "__main__":
    env = MyWrapper()
    state = env.reset()
    print(f"初始状态: {state}")
    
    # 显示初始环境
    env.show()
    
    # 测试一步
    action = 1  # 向下
    next_state, reward, over = env.step(action)
    print(f"状态: {next_state}, 奖励: {reward}, 结束: {over}")
    
    # 显示执行动作后的环境
    env.show()