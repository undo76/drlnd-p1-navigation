import numpy as np
import random
from collections import namedtuple
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

# DEFAULT PARAMETERS
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.995  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
DEFAULT_PRIORITY = 10.0  # priority for new experiences
PRIORITY_EPS = 0.0001  # Epsilon to add to priorities
PRIORITY_ETA = 0.5  # Exponent for priority
PRIORITY_WEIGHT_BETA = 1.0  # Exponent for the priority importance loss scaling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, gamma=GAMMA,
                 tau=TAU, lr=LR,
                 update_every=UPDATE_EVERY, default_priority=DEFAULT_PRIORITY, priority_eps=PRIORITY_EPS,
                 priority_eta=PRIORITY_ETA, priority_weight_beta=PRIORITY_WEIGHT_BETA):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            batch_size (int): minibatch size
            buffer_size (int): replay buffer size
            gamma (float): discount factor
            tau (float): Polyak update of target parameters
            lr (float): learning rate
            update_every (int): how often to update the network
            default_priority (float): priority for new experiences
            priority_eps (float): Epsilon to add to priorities
            priority_eta (float): Exponent for priority
            priority_wight_beta (float): Exponent for the priority importance loss scaling
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.default_priority = default_priority
        self.priority_eps = priority_eps
        self.priority_eta = priority_eta
        self.priority_weight_beta = priority_weight_beta
        random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, default_priority, priority_eps,
                                   priority_eta)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `update_every` time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, indices, probabilities) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Get max predicted Q values for next states using Double DQN
        q_targets_next_actions = self.qnetwork_local(next_states).detach().argmax(1, keepdim=True)
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, q_targets_next_actions)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss and scale it according to the priority weight
        loss = F.mse_loss(q_expected, q_targets, reduce=False)
        loss /= (len(self.memory) * weights) ** self.priority_weight_beta
        loss = loss.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.polyak_update(self.qnetwork_local, self.qnetwork_target)

        # Update replay memory priorities
        errors = (q_targets - q_expected.detach()).abs().cpu().numpy().squeeze()
        self.memory.update_priorities(errors, indices)

    def polyak_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,
                 default_priority, priority_eps, priority_eta):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            default_priority (float): initial priority for experiences
            priority_eps (float): Priority epsilon
            priority_eta (float): Priority exponent
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = np.empty(shape=buffer_size, dtype=np.object)
        self.priorities = np.zeros(shape=buffer_size, dtype=np.float64)
        self.count = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.rng = np.random.default_rng(seed=seed)
        self.default_priority = default_priority
        self.priority_eps = priority_eps
        self.priority_eta = priority_eta

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory[self.count % self.buffer_size] = self.experience(state, action, reward, next_state, done)
        self.priorities[self.count % self.buffer_size] = self.default_priority
        self.count += 1

    def update_priorities(self, priorities, indices):
        """Update the priorities of the samples"""
        self.priorities[indices] = (priorities + self.priority_eps) ** self.priority_eta

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        sample, indices, weights = self._sample_with_indices_and_weights()

        states = torch.from_numpy(np.vstack([e.state for e in sample])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in sample])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sample])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in sample])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in sample]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def _sample_with_indices_and_weights(self):
        # weights are initialized with np.zeros(), no need to slice
        weights = self.priorities / self.priorities.sum()
        indices = self.rng.choice(self.buffer_size, self.batch_size, p=weights, replace=False)
        return self.memory[indices], indices, weights[indices]

    def __len__(self):
        """Return the current size of internal memory."""
        return self.count
