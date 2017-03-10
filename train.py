from gridworld import GridWorld
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Database(object):
    def __init__(self):
        self.bad_trajectories = []
        self.good_trajectories = []
        self.concept_peak = np.zeros((7,13))
        self.lamda = 0.5
        self.threshold = 1
        self.log_dd = np.zeros((7,13))
        self.eps = 1e-6

    def update(self, trajectory, done):
        if done:
            self.good_trajectories.append(trajectory)
            p = np.zeros((7,13))
            idx_x = np.tile(np.arange(7), (13,1)).T
            idx_y = np.tile(np.arange(13), (7,1))
            for observation in trajectory:
                x, y = observation
                p += np.log(1 - np.exp(-1/2*((x-idx_x)**2 + (y-idx_y)**2))/np.sqrt(2*np.pi))
            self.log_dd += np.log(1 - np.exp(p))

        else:
            self.bad_trajectories.append(trajectory)
            p = np.zeros((7,13))
            idx_x = np.tile(np.arange(7), (13,1)).T
            idx_y = np.tile(np.arange(13), (7,1))
            for observation in trajectory:
                x, y = observation
                p += np.log(1 - np.exp(-1/2*((x-idx_x)**2 + (y-idx_y)**2))/np.sqrt(2*np.pi))
            self.log_dd += p

        x = self.log_dd.argmax(axis=0)
        y = self.log_dd.max(axis=0).argmax()
        x = x[y]
        self.concept_peak[x,y] = self.lamda*(self.concept_peak[x,y] + 1)

        if self.concept_peak[x,y] >= self.threshold:
            return x,y

class QLearning(object):
    def __init__(self):
        self.q = np.zeros((7,13,4))
        self.gamma = 0.9
        self.lr = 0.05
        self.epsilon = 0.9
        self.action_space = ['left', 'right', 'up', 'down']

    def get_action(self, observation):
        if np.random.binomial(1, self.epsilon):
            return np.random.choice(self.action_space)

        x, y = observation
        action = self.action_space[self.q[x, y].argmax()]

        return action

    def update(self, observation, action, new_observation, reward):
        x, y = observation
        new_x, new_y = new_observation
        action = self.action_space.index(action)
        self.q[x, y, action] += self.lr*(reward + self.gamma*self.q[new_x, new_y].max() - self.q[x, y, action])


env = GridWorld()
agent = QLearning()
db = Database()
print env.border.shape

for i_episode in range(10000):
    trajectory = []
    observation = env.reset()
    for t in range(1000):
        trajectory.append(observation)
        action = agent.get_action(observation)
        new_observation, reward, done, info = env.step(action)
        agent.update(observation, action, new_observation, reward)
        observation = new_observation
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    agent.epsilon = max(0.1, agent.epsilon - 0.001)
    peak = db.update(trajectory, done)
    if peak:
        print peak

print agent.epsilon
print agent.q.argmax(axis=2)
plt.figure()
plt.imshow(db.log_dd, cmap='gray', interpolation=None)
plt.show()
