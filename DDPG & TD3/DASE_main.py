import numpy as np
import torch
import gym
import argparse
import os
import threading
import socket

import utils
import DASE_DDPG
import DASE_TD3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(agent, env_name, seed, agent_id, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, Agent: {agent_id + 1}")
    print("---------------------------------------")
    return avg_reward


def main(args, replay_buffer, agent_id):
    file_name = f"{args.policy}_{args.env}_{args.seed}_Agent_{agent_id}"

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Agent: {agent_id}")
    print("---------------------------------------")

    agent_seed = args.seed + agent_id * 100

    env = gym.make(args.env)

    # Set seeds
    env.seed(agent_seed)
    env.action_space.seed(agent_seed)
    torch.manual_seed(agent_seed)
    np.random.seed(agent_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "agent_id": agent_id,
        "gpu": args.gpu,
        "discount": args.discount,
        "tau": args.tau,
        "kl_div_var": args.kl_div_var
    }

    # Initialize policy
    if args.policy == "DASE_DDPG":
        agent = DASE_DDPG.DASE_DDPG(**kwargs)
    elif args.policy == "DASE_TD3":
        agent = DASE_TD3.DASE_TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    # Evaluate untrained policy
    evaluations = [f"HOST: {socket.gethostname()}", f"GPU: {torch.cuda.get_device_name(args.gpu)}",
                   evaluate_policy(agent, args.env, args.seed, agent_id)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0

    for t in range(int(args.max_time_steps)):

        episode_time_steps += 1

        # Select action randomly or according to policy
        if t < args.start_time_steps:
            action = env.action_space.sample()
        else:
            action = (
                    agent.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_time_steps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, agent_id, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_time_steps:
            agent.update_parameters(replay_buffer, args.batch_size)

        if done:
            print(
                f"Agent: {agent_id + 1} Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps}"
                f" Reward: {episode_reward:.3f}")

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(evaluate_policy(agent, args.env, args.seed, agent_id))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: agent.save(f"./models/{file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DASE_TD3", help='Algorithm (default: DASE_TD3)')
    parser.add_argument("--kl_div_var", default=0.1, type=float, help='Diagonal entries of the reference Gaussian for '
                                                                      'the Deterministic SAC')
    parser.add_argument("--env", default="Hopper-v2", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=0, type=int,
                        help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=25000, type=int, metavar='N',
                        help='Number of exploration time steps sampling random actions (default: 1000)')
    parser.add_argument("--buffer_size", default=1000000, type=int,
                        help='Size of the experience replay buffer (default: '
                             '1000000)')
    parser.add_argument("--eval_freq", default=1e3, metavar='N', help='Evaluation period in number of time '
                                                                      'steps (default: 1000)')
    parser.add_argument("--max_time_steps", default=1000000, type=int, metavar='N',
                        help='Maximum number of steps (default: 1000000)')
    parser.add_argument("--expl_noise", default=0.1, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=256, metavar='N',
                        help='Batch size (default: 256)')
    parser.add_argument("--discount", default=0.99, metavar='G',
                        help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=0.005, type=float, metavar='G',
                        help='Learning rate in soft/hard updates of the target networks (default: 0.005)')
    parser.add_argument("--policy_noise", default=0.2, metavar='G', help='Noise added to target policy during critic '
                                                                         'update')
    parser.add_argument("--noise_clip", default=0.5, metavar='G', help='Range to clip target policy noise')
    parser.add_argument("--policy_freq", default=2, type=int, metavar='N', help='Frequency of delayed policy updates')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    args = parser.parse_args()

    if args.policy == "DASE_DDPG":
        args.tau = 0.001

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    tmp_env = gym.make(args.env)

    state_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.shape[0]
    max_action = float(tmp_env.action_space.high[0])

    torch.cuda.set_device(args.gpu)

    replay_buffer = utils.SharedExperienceReplayBuffer(state_dim, action_dim, args.gpu, max_size=args.buffer_size)

    del tmp_env

    num_agents = 2
    threads = []

    # Start multi-thread training
    for agent_id in range(num_agents):
        thread = threading.Thread(target=main, args=(args, replay_buffer, agent_id), daemon=True)
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()
