import gymnasium as gym
import torch
from dqn_agent import DQNAgent

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(4, 2, device)
agent.q_net.load_state_dict(torch.load("dqn_model.pth"))  # 사전 저장된 학습 모델 로드
agent.q_net.eval()

state, _ = env.reset()
done = False

while not done:
    action = agent.select_action(state)
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
