"""
try:
    %tensorflow_version 1.x
    %matplotlib inline
except Exception:
    pas
"""

# import and test
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())

# we want TF 1.x : ```stable_baselines``` ne supporte pas ```tf>2.x```
assert tf.__version__ < "2.0"

import random
import json
import itertools
import datetime as dt
from IPython.display import clear_output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces

#! pip install stable-baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import torch
from torch import nn
import torch.nn.functional as F

# Tuteur

add = lambda x, y : x+y
mult = lambda x, y : x*y
subt = lambda x, y : x-y
eps = 1e-12 # avoid division by zero
div = lambda x, y : x/(y+eps)

mathematical_operators = {"+" : add, "*" : mult, "-" : subt, "/" : div}

def get_dataset(low, high, n, operator = "+", n_choices = 2) :
    assert operator in mathematical_operators.keys()
    assert high - low + 1 > 2*n
    assert n_choices >= 2
    op = mathematical_operators[operator]
    ds = random.sample(range(low, high), 2*n)
    ds = zip(ds[:n], ds[n:])

    qcms = []

    for x in ds :
        r = op(x[0], x[1])
        #choices = [r, r + random.randint(-high, high)] + [r + random.randint(-high, high) for _ in range(n_choices-1)]
        choices = [r] + [r + random.randint(-high, high) for _ in range(n_choices-1)]
        random.shuffle(choices)
        qcms.append({
            "question" : tuple(list(x) + [operator]), # str(x[0])+"+"+str(x[1]) 
            "choices" : choices, 
            "answer" : r
        })
    return qcms

# Apprenant
class Student(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, qcms, low, high, n_choices = 2, loss_threshold = 2):
        super(Student, self).__init__()
        assert n_choices >= 2

        self.qcms = qcms
        self.low = low
        self.high = high
        self.n_choices = n_choices
        self.loss_threshold = loss_threshold

        self.score = 0
        self.current_step = 0

        self.batching = False
        
        # Actions of the format : 0 or n_choices-1
        self.action_space = spaces.Discrete(n_choices)

        # Un qcm
        self.dic_spaces = {
          'question' : spaces.Box(low = np.array([self.low]*n_choices), high = np.array([self.high]*n_choices), dtype = np.int8),
          'choices' : self.action_space,
          'answer' : spaces.Box(low = self.low, high = self.high + self.high, shape = (1,), dtype = np.int8)
        }

        self.observation_space = spaces.Dict(self.dic_spaces)
        #self.observation_space = spaces.Box(low = np.array([self.low]*n_choices), high = np.array([self.high]**n_choices), dtype = np.int8)
        
    def prepare_dataset(self, batch_size = 20) :
        ds = []
        n_samples = len(self.qcms) 
        i = 0
        while n_samples > i :
            i += batch_size
            x = self.qcms[i-batch_size:i]
            ds.append({
                #"question" : [x_i["question"] for x_i in x],  
                "question" : [x_i["question"][:-1] for x_i in x], # ignore opérator
                "choices" : [x_i["choices"] for x_i in x], 
                "answer" : [x_i["answer"] for x_i in x]
            })

        self.ds = ds
        self.batching = True
        self.batch_size = batch_size
        self.action_space = spaces.Box(low = 0, high = self.n_choices-1, shape = (batch_size,), dtype = np.int8)
        self.dic_spaces = {
          'question' : spaces.Box(low = self.low, high = self.high, shape = (batch_size, self.n_choices), dtype = np.int8),
          'choices' : self.action_space,
          'answer' : spaces.Box(low = self.low, high = self.high + self.high, shape = (batch_size, 1,), dtype = np.int8)
        }
        self.observation_space = spaces.Dict(self.dic_spaces)

    def _next_observation(self):
        if self.batching :
            return self.ds[self.current_step]
        else :
            return self.qcms[self.current_step]

    def take_action(self, state, action) :
        return [x_i[y_i] for x_i, y_i in zip(state['choices'], action)]
        
    def _take_action(self, qcm, action):
        return qcm['choices'][action]

    def policy(self, state, Q) :
        y_pred = Q(torch.Tensor(state["question"])).reshape((self.batch_size,))
        action = []
        choices_tmp = np.arange(self.n_choices)
        for x_i, y_i in zip(state["choices"], y_pred.detach().numpy()) :
            try :
              a = x_i.index(y_i)
            except ValueError:
              idx = np.argmin(np.absolute(y_i - np.array(x_i)))
              if abs(x_i[idx] - y_i) <= self.loss_threshold :
                  a = idx
              else : 
                  a = np.random.choice(choices_tmp)
            action.append(a)
        return action, y_pred 

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        
        if not self.batching :
            if self.current_step > len(self.qcms) - 1 :
                self.current_step = 0
            qcm = self._next_observation()
            answer = self._take_action(qcm, action)
            reward = 1 if answer == qcm["answer"] else -0.25
            self.score += reward
            done = self.score / len(self.qcms) >= 0.5
            return qcm, reward, done, {}

        else :
            done = False
            if self.current_step > len(self.ds) - 1 :
                self.current_step = 0
                #done = True
            qcm = self._next_observation()
            answer = self.take_action(qcm, action)
            reward = (np.array(answer) == np.array(qcm["answer"])).astype(np.float)
            reward = [-0.25 if r == 0 else r for r in reward]
            reward = sum(reward) / self.batch_size 
            self.score += reward
            done = self.score >= 0.5
            return qcm, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.score = 0
        # Set the current step to a random point within the data frame
        if self.batching :
            self.current_step = random.randint(0, len(self.ds) - 1)
        else :
            self.current_step = random.randint(0, len(self.qcms) - 1)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        m = len(self.qcms) if not self.batching else len(self.ds)
        print(f'Score : {self.score} note {self.score / m}')

# Algorithme d'entrainement
def train(env, n_episodes, Q, log_interval = 1) :
    #%%time
    """Training the agent"""
  
    # Hyperparameters
    #epsilon = 0.1

    # For plotting metrics
    #all_epochs = []
    #all_penalties = []

    all_reward = {}
    all_loss = {}

    env.reset()
    Q.train()
    for i in range(1, n_episodes + 1):
        
        all_reward[i] = [] 
        all_loss[i] = []
        
        state = env.reset()

        epochs, total_reward = 0, 0
        done = False
        
        while not done:
            Q.optimizer.zero_grad()
            #cond =  random.uniform(0, 1) < epsilon
            cond = False
            if cond :
                action = env.action_space.sample() # Explore action space
                y_pred = env.take_action(state, action)
                #print(action, state, y_pred)
            else:
                action, y_pred = env.policy(state, Q) # Exploit learned values
            
            next_state, reward, done, _ = env.step(action) 
            y = torch.Tensor(state["answer"])
            loss = Q.criterion(y, y_pred)
            print("reward", reward, "loss", loss.item())
            all_loss[i].append(loss.item())
            all_reward[i].append(reward)
            total_reward += reward

            loss.backward()
            Q.optimizer.step()

            state = next_state
            epochs += 1

            env.render()
            
        if i % log_interval == 0:
            clear_output(wait=True)
            print(f"Episode: {i}, n_epochs : {epochs}, reward {total_reward}")

    print("Training finished.\n")
    return Q, all_loss, all_reward

# Algorithme d'évaluation
def evaluate(env, n_episodes, Q = None, log_interval = 2):
    #%%time
    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    frames_RL = {}
    
    if Q is not None :
        Q.eval()

    for i in range(1, n_episodes + 1):
        state = env.reset()
        epochs, total_reward = 0, 0
        
        done = False

        frames_RL[i] = []
        
        while not done:
            if Q is not None :
                action, y_pred = env.policy(state, Q)
                state, reward, done, _ = env.step(action)
                y = torch.Tensor(state["answer"])
                loss = Q.criterion(y, y_pred)
            else :
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)

            frames_RL[i].append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
                }
            )
            env.render()
            epochs += 1
            total_reward += reward

        total_epochs += epochs

        if i % log_interval == 0:
            clear_output(wait=True)
            print(f"Episode: {i}, n_epochs : {epochs}, reward {total_reward}")

    print(f"Results after {n_episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / n_episodes}")

    return frames_RL

# model
class Linear(nn.Module):
    """costomized linear layer"""
    def __init__(self, in_features, out_features, bias = True, activation_function = None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.activation_function = activation_function if activation_function else lambda x : x
      
    def forward(self, x):
        return self.activation_function(self.linear(x))

class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                        activation_function = None, init_weights = True, params_seed = 0):
        torch.manual_seed(params_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        super(MLP, self).__init__()
        net = []
        net.append(Linear(in_features, hidden_features, True, activation_function))  
        net += [Linear(hidden_features, hidden_features, True, activation_function) for _ in range(hidden_layers)]  
        net.append(Linear(hidden_features, out_features, True, None))
        self.net = nn.Sequential(*net)

        if init_weights :
            # init_weights : Motivated by "Implicit Neural Representations with Periodic Activation Functions" (https://arxiv.org/abs/2006.09661).
            hidden_omega_0 = 1.0
            with torch.no_grad():
                self.net[0].linear.weight.uniform_(-1 / in_features, 1 / in_features)      
                    
                for l in range(1, len(self.net)) :
                    self.net[l].linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                                        np.sqrt(6 / hidden_features) / hidden_omega_0)

    def forward(self, x):
        return self.net(x)

def get_Q(input_dim = 2, output_dim = 1, hidden_dim = 20, n_hidden = 1, g = torch.relu, lr = 0.0001):
    """
    input_dim = env.observation_space.n
    output_dim = env.action_space.n
    hidden_dim : dimension des couches cachées
    n_hidden : nombre de couches cachées
    g : fonction d'activation (torch.relu, F.softplus, ...)
    lr : pas d'apprentissage
    """
    Q = MLP(in_features = input_dim, hidden_features = hidden_dim, hidden_layers = n_hidden, 
            out_features = output_dim, activation_function = g, init_weights = True, params_seed = 0)

    Q.criterion = torch.nn.MSELoss()
    Q.optimizer = torch.optim.Adam(Q.parameters(), lr = lr)

    return Q

# Affichage
def plot_frames(frames, start = 1, end = None):
    y = [] 
    end = len(frames.keys()) + 1 if end is None else end
    assert 1 <= start <= end <= len(frames.keys()) + 1
    if start == end :
        end += 1
    for i in range(start, end):
        l = len(frames[i])
        y = y + [frames[i][j]["reward"] for j in range(l)] 
    plt.plot(range(len(y)), y)

def standard_plot(dic, start = 1, end = None):
    y = [] 
    end = len(dic.keys()) + 1 if end is None else end
    assert 1 <= start <= end <= len(dic.keys()) + 1
    if start == end :
        end += 1
    for i in range(start, end):
        l = len(dic[i])
        y = y + [dic[i][j] for j in range(l)] 
    plt.plot(range(len(y)), y)