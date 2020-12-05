import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from utils import time_format, eval_policy, create_buffer
from dqn_agent import DQNAgent
from framestack import FrameStack



def train_agent(env, config):
    """
    Args:
    """
    
    # create CNN convert the [1,3,84,84] to [1, 200]
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    #pathname = str(args.locexp) + "/" + str(args.env_name) + '_agent_' + str(args.policy)
    #pathname += "_batch_size_" + str(args.batch_size) + "_lr_act_" + str(args.lr_actor) 
    #pathname += "_lr_critc_" + str(args.lr_critic) + "_lr_decoder_"
    pathname = dt_string 
    tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
    agent = DQNAgent(state_size=200, action_size= env.action_space.n,  config=config)
    writer = SummaryWriter(tensorboard_name)
    print("action_size {}".format(env.action_space.n))
    # eval_policy(env, agent, writer, 0, config)
    memory =  ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], int(config["image_pad"]), config["device"])
    if config["create_buffer"]:
        create_buffer(env, memory, config)
        memory.load_memory("/export/leiningc/" + config["buffer_path"])
    else:
        print("load Buffer")
        memory.load_memory("/export/leiningc/" + config["buffer_path"])
        print("Buffer size {}".format(memory.idx))
    eps = config["eps_start"]
    eps_end = config["eps_end"]
    eps_decay = config["eps_decay"]
    scores_window = deque(maxlen=100)
    scores = [] 
    t0 = time.time()
    for i_episode in range(config["train_episodes"]):
        obs = env.reset()
        score = 0
        for t in range(config["max_t"]):
            action = agent.act(obs, eps)
            # action = env.action_space.sample()
            next_obs, reward, done_no_max, _ = env.step(action)
            done = done_no_max
            if t + 1 == config["max_t"]:
                print("t ", t)
                done = 0
            memory.add(obs, action, reward, next_obs, done, done_no_max)
            agent.step(memory, writer)
            obs = next_obs
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent scor
        scores.append(score)              # save most recent score
        ave_score = np.mean(scores_window)
        writer.add_scalar("ave_score", ave_score, i_episode)
        writer.add_scalar("episode_score", score, i_episode)
        print('\rEpisode {} score {} \tAverage Score: {:.2f}  eps: {:.2f} time: {}'.format(i_episode, score, np.mean(scores_window), eps, time_format(time.time() - t0)), end="")         
        if i_episode % config["eval"] == 0:
            eval_policy(env, agent, writer, i_episode, config)
            agent.save(str(config["locexp"]) + "/models/eval-{}/".format(i_episode))
            print('Episode {} Average Score: {:.2f}  eps: {:.2f} time: {}'.format(i_episode, np.mean(scores_window), eps, time_format(time.time() - t0)),)

