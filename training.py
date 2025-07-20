import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

env = gym.make("CartPole-v1", render_mode=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_dim=4, action_dim=2, device=device)

num_episodes = 500
target_update_freq = 10
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, reward, next_state, float(done))
        agent.update()
        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if episode % target_update_freq == 0:
        agent.update_target_network()

    rewards.append(total_reward)
    print(f"Episode {episode} | Total Reward: {total_reward} | Epsilon: {agent.epsilon:.3f}")

    if total_reward >= 500:
        print("🎉 500점 달성!")
        break

env.close()

# 학습 결과 시각화
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN on CartPole-v1")
plt.savefig("reward_plot.png")
