from ddpg_agent.utils import *
from ddpg_agent.networks import *
from ddpg_agent.policy import *
from ddpg_agent.buffer import *
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
import numpy as np

import torch.nn.functional as F
from torch.distributions import Normal
import wandb

# %matplotlib inline
# from src.ddpg_agent.policy import NormalizedActions
from ddpg_agent.policy import NormalizedActions

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda")
# device = torch.device("cpu")
# env = NormalizedActions(gym.make("Pendulum-v1"))
# env = NormalizedActions(gym.make("Pendulum-v0"))


class DDPG_Agent:
    def __init__(
        self,
        state_dim=3,
        action_dim=1,
        hidden_dim=256,
        init_w=1e-3,
        value_lr=1e-3,
        policy_lr=1e-4,
        replay_buffer_size=1000000,
        max_steps=4,
        max_frames=12000,
        batch_size=2,
        log_dir="./log/epochs",
    ):
        # super(DDPG_Agent, self).__init__()
        self.lr = 3e-2
        self.num_steps = 20  # number of iterations for each episodes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = []
        self.states = []

        # self.policy_network = policy_network

        self.log_probs = []
        self.values = []
        self.rewards = []  # rewards for each episode
        # self.masks     = []
        self.entropy = 0
        self.step_count = 0
        self.frame_idx = 0
        self.max_frames = max_frames  # number of episodes
        self.episode_reward = 0

        # self.env = NormalizedActions(gym.make("Pendulum-v0"))
        new_action_space = spaces.Box(low=0, high=1, shape=(action_dim,))
        self.ou_noise = OUNoise(new_action_space)

        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.shape[0]
        # state_dim = state_dim
        # action_dim = action_dim

        # FOR cuda ONLY
        # self.value_net = ValueNetwork(
        #     state_dim, action_dim, hidden_dim).to("cuda")
        # self.policy_net = PolicyNetwork(
        #     state_dim, action_dim, hidden_dim).to("cuda")
        # self.target_value_net = ValueNetwork(
        #     state_dim, action_dim, hidden_dim).to("cuda")
        # self.target_policy_net = PolicyNetwork(
        #     state_dim, action_dim, hidden_dim).to("cuda")


        self.value_net = ValueNetwork(
            state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(
            state_dim, action_dim, hidden_dim)
        self.target_value_net = ValueNetwork(
            state_dim, action_dim, hidden_dim)
        self.target_policy_net = PolicyNetwork(
            state_dim, action_dim, hidden_dim)

        

        self.memory = (
            Memory()
        )  # store all the (s, a, s', r) during the transition process
        # replay buffer used for main training
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.value_lr  = 1e-3
        # self.policy_lr = 1e-4

        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()
        self.step = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_dir = log_dir
        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def ddpg_update(
        self, gamma=0.99, min_value=-np.inf, max_value=np.inf, soft_tau=1e-2
    ):

        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )
        # print('DONE HAHAHAHAHAHA')
        # FOR cuda only
        # state = torch.FloatTensor(state).squeeze().to("cuda")
        # next_state = torch.FloatTensor(next_state).squeeze().to("cuda")
        # action = torch.FloatTensor(action).squeeze().to("cuda")
        # reward = torch.FloatTensor(reward).to("cuda")
        # done = torch.FloatTensor(np.float32(done)).to("cuda")


        # For CPU
        state = torch.FloatTensor(state).squeeze()
        next_state = torch.FloatTensor(next_state).squeeze()
        action = torch.FloatTensor(action).squeeze()
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))
        # print(f'state: {state.shape}')
        

        policy_loss = self.value_net(state, self.policy_net(state))
        # print(f'next state: {next_state.shape}')
        # print(policy_loss)
        policy_loss = -policy_loss.mean()
        # print(policy_loss.item())
        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        # print(f'DONE DONE DONE: {done.shape}')
        # print(f'Reward: {reward.shape}')
        # print(f'target_value: {target_value.shape}')

        expected_value = reward + (1.0 - done) * gamma * target_value.squeeze()
        expected_value = torch.clamp(expected_value, min_value, max_value)
        # print(f'action: {action.shape}')
        value = self.value_net(state, action).squeeze()
        # print(f'VALUE: {value.shape}')
        # print(f'E VALUE: {expected_value.detach().shape}')
        value_loss = self.value_criterion(
            value, expected_value)
        # print(f'value loss: {value_loss.item()}')

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def get_action(self, local_losses, local_n_samples, local_num_epochs, done):
        # reach to maximum step for each episode or get the done for this iteration

        state = get_state(
            losses=local_losses, epochs=local_num_epochs, num_samples=local_n_samples
        )
        # print(state)
        prev_reward = get_reward(local_losses)
        
        if self.step == self.max_steps - 1 or done:
            self.rewards.append(self.episode_reward)
            self.logging_per_round()
            state = self.reset_state()
            # self.ou_noise.reset()
            # self.episode_reward = 0

        if self.frame_idx >= self.max_frames:
            # maybe stop training?
            self.logging_per_round()
            state = self.reset_state()

        # state = torch.FloatTensor(state).unsqueeze(
        #     0).to("cuda")  # current state
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if prev_reward is not None:
            self.memory.update(r=prev_reward)
        action = self.policy_net.get_action(state)
        action = self.ou_noise.get_action(action, self.step)
        self.memory.act(state, action)

        # if self.step < self.max_steps:
        if self.memory.get_last_record() is None:
            self.step += 1
            self.frame_idx += 1
            return action
        s, a, r, s_next = self.memory.get_last_record()
        self.replay_buffer.push(s, a, r, s_next, done)
        if len(self.replay_buffer) >= self.batch_size:
            self.ddpg_update()
        self.episode_reward += prev_reward
        self.frame_idx += 1
        self.step += 1
        sample = {
            'episode_reward': self.episode_reward,
            'total_reward': self.rewards,
            'local_losses': local_losses,
            'mean_losses': np.mean(np.asarray(local_losses)),
            'std_losses': np.std(np.asarray(local_losses)),
        }
        wandb.log({'agent/ddpg_agent': sample, 'agent/local_losses': local_losses});

        if self.frame_idx % max(1000, self.max_steps + 1) == 0:
            plot(self.frame_idx, self.rewards)
            # plt.savefig()

        return action

    def reset_state(self):
        self.ou_noise.reset()
        self.episode_reward = 0
        self.step = 0
        return np.zeros(self.state_dim)
        # return env.reset()

    def logging_per_round(self):
        frame_idx, episode, total_reward = (
            self.frame_idx,
            self.episode_reward,
            self.episode_reward,
        )
        sample = {
            "frame_idx": frame_idx,
            "episode": episode,
            "total_reward": total_reward,
        }
        filename = self.log_dir + "/log_dqn.txt"
        with open(filename, "a+") as log_f:
            log_f.write(str(sample))


if __name__ == "__main__":
    max_frames = 12000
    max_steps = 16
    frame_idx = 0
    rewards = []
    batch_size = 128
    # Test simulator
    env = NormalizedActions(gym.make("Pendulum-v0"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 10
    agent = DDPG_Agent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
    )
    ou_noise = OUNoise(env.action_space)
    reward, done = None, None

    while frame_idx < max_frames:
        state = env.reset()
        ou_noise.reset()
        # episode_reward = 0

        for step in range(max_steps):
            # action = policy_net.get_action(state)
            # action = ou_noise.get_action(action, step)
            action = agent.get_action(state, reward, done)
            next_state, reward, done, _ = env.step(action)

            # replay_buffer.push(state, action, reward, next_state, done)
            # if len(replay_buffer) > batch_size:
            #     ddpg_update(batch_size)

            state = next_state

            # episode_reward += reward
            # frame_idx += 1

            # if frame_idx % max(1000, max_steps + 1) == 0:
            #     plot(frame_idx, rewards)

            # if done:
            #     break

        # rewards.append(episode_reward)
