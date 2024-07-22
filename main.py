# 주요 라이브러리 및 모듈 임포트
from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os
import random

from agents.Hebbianppo import HebbianPPO
from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG

from utils.utils import make_transition, Dict, RunningMeanStd
os.makedirs('./model_weights', exist_ok=True)

# 명령행 인자 파싱
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 명령행 인자 파싱
parser = ArgumentParser('parameters')
parser.add_argument("--env_name", type=str, default = 'Hopper-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ppo)')
parser.add_argument("--train", type=str2bool, default=True, help="(default: True)")
parser.add_argument("--render", type=str2bool, default=False, help="(default: False)")
parser.add_argument("--epochs", type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=str2bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default=100, help='save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default=1, help='print interval(default : 20)')
parser.add_argument("--use_cuda", type=str2bool, default=True, help='cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default=0.1, help='reward scaling(default : 0.1)')
parser.add_argument("--seed", type=int, default = 0, help="This is seed that we can choose. It can reimplement easily")
args = parser.parse_args()

# ConfigParser 초기화 및 설정 파일 읽기
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo)

os.makedirs('./model_weights/'+args.env_name+"/"+args.algo+"/", exist_ok=True)

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
    
# TensorBoard 설정
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(f'./log/{args.env_name}/{args.algo}/', exist_ok=True)
    n_num = len(os.listdir(f'./log/{args.env_name}/{args.algo}/'))
    log_name = f"{n_num+1}"
    writer = SummaryWriter(log_dir=f'./log/{args.env_name}/{args.algo}/'+log_name)
else:
    writer = None
# 시드 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(args.seed)
    
# 환경 초기화
env = gym.make(args.env_name, render_mode='human', healthy_z_range=(0.3,1.5))
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)

# 알고리즘 선택 및 에이전트 초기화
if args.algo == 'ppo':
    agent = PPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'hebbianppo':
    agent = HebbianPPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'sac':
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'ddpg':
    from utils.noise import OUNoise
    noise = OUNoise(action_dim, 0)
    agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()


# 모델 로드
if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/" +args.env_name+"/"+args.algo+"/"+ args.load))
    print("loaded")
score_lst = []
state_lst = []

if args.train == True:
    print("Train Mode")
    # on-policy 알고리즘
    if agent_args.on_policy:
        score = 0.0
        state_ = (env.reset())
        state = np.clip((state_[0] - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        for n_epi in range(args.epochs):
            for t in range(agent_args.traj_length):
                if args.render:
                    env.render()
                state_lst.append(state_)
                mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                next_state_, reward, done, _, _ = env.step(action.cpu().numpy())
                next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                transition = make_transition(state,
                                            action.cpu().numpy(),
                                            np.array([reward * args.reward_scaling]),
                                            next_state,
                                            np.array([done]),
                                            log_prob.detach().cpu().numpy()
                                            )
                agent.put_data(transition)
                score += reward

                if done :
                    state_ = (env.reset())
                    state = np.clip((state_[0] - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                    score_lst.append(score)
                    if args.tensorboard:
                        writer.add_scalar("score/score", score, n_epi)
                    score = 0
                else:
                    state = next_state
                    state_ = next_state_

            agent.train_net(n_epi)

            state_lst_ = []
            for item in state_lst:
                if isinstance(item, tuple):
                    state_lst_.append(item[0])
                elif isinstance(item, np.ndarray):
                    state_lst_.append(item)
            state_rms.update(np.vstack(state_lst_))
            if n_epi % args.print_interval == 0 and n_epi != 0:
                avg_score = sum(score_lst) / len(score_lst) if len(score_lst)!=0 else 0
                print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
                score_lst = []
            if n_epi % args.save_interval == 0 and n_epi != 0:
                torch.save(agent.state_dict(), './model_weights/'+args.env_name+'/'+args.algo+'/'+'agent_'+ str(n_epi))
                
    # off-policy 알고리즘
    else:
        for n_epi in range(args.epochs):
            score = 0.0
            state = env.reset()
            done = False
            while not done:
                if args.render:
                    env.render()
                action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
                action = action.cpu().detach().numpy()
                next_state, reward, done, _ = env.step(action)
                transition = make_transition(state,
                                            action,
                                            np.array([reward * args.reward_scaling]),
                                            next_state,
                                            np.array([done])
                                            )
                agent.put_data(transition)

                state = next_state
                score += reward
                if agent.data.data_idx > agent_args.learn_start_size:
                    agent.train_net(agent_args.batch_size, n_epi)
            score_lst.append(score)
            if args.tensorboard:
                writer.add_scalar("score/score", score, n_epi)
            if n_epi % args.print_interval == 0 and n_epi != 0:
                avg_score = sum(score_lst) / len(score_lst) if score_lst else 0
                print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
                score_lst = []
            if n_epi % args.save_interval == 0 and n_epi != 0:
                torch.save(agent.state_dict(), './model_weights/agent_' + str(n_epi))

else:
    print("Evaluation Mode")
    # 평가 모드
    for n_epi in range(args.epochs):
        score = 0.0
        state_ = env.reset()
        state = np.clip((state_[0] - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        done = False
        while not done:
            if args.render:
                env.render()
            action, _, _ = agent.eval(state)
            next_state_, reward, done, _, _ = env.step(action.reshape(-1))
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            state = next_state
            score += reward
        print("Episode: {}, Score: {:.1f}".format(n_epi, score))