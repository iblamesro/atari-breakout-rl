
import argparse, os, csv, time
import numpy as np
import torch
import gymnasium as gym
from collections import deque
from tqdm import tqdm

from wrappers import make_env
from agent_ddqn import DDQNAgent

def evaluate(env, agent, episodes=10):
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            # low exploration for eval
            old = agent.eps_final
            agent.eps_final = 0.05
            a = agent.act(s)
            agent.eps_final = old
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total-frames', type=int, default=2_000_000)
    parser.add_argument('--log-every', type=int, default=10_000)
    parser.add_argument('--eval-every', type=int, default=100_000)
    parser.add_argument('--save-dir', type=str, default='runs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = make_env(args.env, seed=args.seed)

    n_actions = env.action_space.n
    agent = DDQNAgent(n_actions=n_actions)

    # seeding

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    s, _ = env.reset()
    episode_return = 0.0
    returns_hist = []
    recent = deque(maxlen=100)
    last_log = 0
    last_eval = 0

    csv_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame','episode','ep_return','mean100','epsilon','loss'])

    ep = 0
    for frame in tqdm(range(1, args.total_frames+1)):
        agent.frame_idx = frame
        a = agent.act(s)
        ns, r, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        agent.store(s, a, r, ns, float(d))
        loss = agent.train_step()
        s = ns
        episode_return += r

        if d:
            ep += 1
            recent.append(episode_return)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame, ep, episode_return, np.mean(recent) if recent else 0.0, agent.epsilon(), loss if loss is not None else ''])
            s, _ = env.reset()
            episode_return = 0.0

        if frame - last_log >= args.log_every:
            last_log = frame

        if frame - last_eval >= args.eval_every:
            last_eval = frame
            mean_ret, std_ret = evaluate(env, agent, episodes=5)
            torch.save(agent.online.state_dict(), os.path.join(args.save_dir, f'ckpt_{frame}.pt'))
            print(f"\n[Eval] frame {frame}: return {mean_ret:.1f} ± {std_ret:.1f} | eps {agent.epsilon():.3f}")

    torch.save(agent.online.state_dict(), os.path.join(args.save_dir, 'final.pt'))
    print('Training finished.')

if __name__ == '__main__':
    main()
