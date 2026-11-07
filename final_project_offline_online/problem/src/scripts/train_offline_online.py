import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


################################
# answer code
def _make_agent_and_dataset(config, args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    env, dataset = config["make_env_and_dataset"]()
    sample = dataset.sample(1)
    agent = agents[config["agent"]](sample["observations"].shape[1:],
                                    sample["actions"].shape[-1], **config["agent_kwargs"])
    return agent.to(ptu.device), env, dataset
################################


################################
# answer code
def _evaluate(agent, env, logger, args, step):
    spec = getattr(env, "spec", None)
    horizon = getattr(spec, "max_episode_steps", None) or getattr(env, "max_episode_steps", 1000)
    successes, returns = [], []
    was_training = agent.training
    agent.eval()
    with torch.no_grad():
        for episode in range(args.num_eval_trajectories):
            obs, info = env.reset(seed=args.seed + 10000 + episode)
            total_reward, success = 0.0, False
            for _ in range(horizon):
                action = np.clip(agent.get_action(obs), env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                success = success or bool(info.get("success", False))
                if terminated or truncated:
                    break
            successes.append(success)
            returns.append(total_reward)
    logger.log({"eval/success_rate": float(np.mean(successes)),
                "eval/average_return": float(np.mean(returns))}, step=step)
    agent.train(was_training)
################################


################################
# answer code
def _update(agent, batch, step):
    batch = ptu.from_numpy(batch)
    return agent.update(batch["observations"], batch["actions"], batch["rewards"],
                        batch["next_observations"], batch["dones"], step)
################################


################################
# answer code
def run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0):
    agent, env, dataset = _make_agent_and_dataset(config, args)
    _evaluate(agent, env, eval_logger, args, start_step)
    for local_step in tqdm.trange(1, args.offline_training_steps + 1, dynamic_ncols=True):
        step = start_step + local_step
        metrics = _update(agent, dataset.sample(config["batch_size"]), step)
        if local_step % args.log_interval == 0 or local_step == args.offline_training_steps:
            train_logger.log(metrics, step=step)
        if local_step % args.eval_interval == 0 or local_step == args.offline_training_steps:
            _evaluate(agent, env, eval_logger, args, step)
    path = dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    env.close()
    return path
################################


################################
# answer code
def run_online_training_loop(config, train_logger, eval_logger, args, agent_path=None, start_step=0):
    agent, env, dataset = _make_agent_and_dataset(config, args)
    if agent_path is not None:
        agent.load_state_dict(torch.load(agent_path, map_location=ptu.device, weights_only=True))
    if hasattr(agent, "online_training"):
        agent.online_training = True
    eval_env, unused_dataset = config["make_env_and_dataset"]()
    del unused_dataset
    replay = ReplayBuffer(args.replay_buffer_capacity)
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    _evaluate(agent, eval_env, eval_logger, args, start_step)
    for local_step in tqdm.trange(1, args.online_training_steps + 1, dynamic_ncols=True):
        step = start_step + local_step
        with torch.no_grad():
            action = (env.action_space.sample() if local_step <= args.online_random_steps
                      else agent.get_action(obs))
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Time limits stop collection but do not remove the Bellman bootstrap.
        replay.insert(observation=obs, action=action, reward=float(reward),
                      next_observation=next_obs, done=bool(terminated))
        episode_return += float(reward)
        obs = next_obs
        episode_metrics = {}
        if terminated or truncated:
            episode_metrics = {"online/episode_return": episode_return,
                               "online/episode_success": float(info.get("success", False))}
            obs, _ = env.reset()
            episode_return = 0.0
        metrics = {}
        if local_step > args.online_warmup_steps:
            for _ in range(args.online_updates_per_step):
                n_offline = int(config["batch_size"] * args.offline_ratio)
                n_online = config["batch_size"] - n_offline
                pieces = []
                if n_offline:
                    pieces.append(dataset.sample(n_offline))
                if n_online:
                    pieces.append(replay.sample(n_online))
                batch = {key: np.concatenate([piece[key] for piece in pieces], axis=0)
                         for key in pieces[0]}
                metrics = _update(agent, batch, step)
        if local_step % args.log_interval == 0 or local_step == args.online_training_steps:
            train_logger.log({**metrics, **episode_metrics, "online/steps": local_step}, step=step)
        if local_step % args.eval_interval == 0 or local_step == args.online_training_steps:
            _evaluate(agent, eval_env, eval_logger, args, step)
    path = dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    env.close()
    eval_env.close()
    return path
################################


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    
    # Online retention of offline data
    ################################
    # answer code
    parser.add_argument("--offline_ratio", type=float, default=0.5,
                        help="Fraction of offline samples in each online training batch")
    ################################
    
    # WSRL
    ################################
    # answer code
    parser.add_argument("--online_warmup_steps", type=int, default=0,
                        help="Collect online data before starting online gradient updates (WSRL)")
    parser.add_argument("--online_random_steps", type=int, default=0)
    parser.add_argument("--online_updates_per_step", type=int, default=1)
    ################################
    

    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config = configs.configs[args.base_config](args.env_name)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    
    ################################
    # answer code
    if not 0 <= args.offline_ratio <= 1:
        raise ValueError("offline_ratio must be between zero and one")
    if args.online_updates_per_step < 1 or args.replay_buffer_capacity < 1:
        raise ValueError("Online update count and replay capacity must be positive")
    if min(args.offline_training_steps, args.online_training_steps, args.online_warmup_steps,
           args.online_random_steps) < 0:
        raise ValueError("Training step counts must be nonnegative")
    if min(args.log_interval, args.eval_interval, args.num_eval_trajectories) < 1:
        raise ValueError("Logging/evaluation intervals and trajectory count must be positive")
    config.update(offline_ratio=args.offline_ratio, online_warmup_steps=args.online_warmup_steps,
                  online_random_steps=args.online_random_steps,
                  online_updates_per_step=args.online_updates_per_step)
    ################################

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    ################################
    # answer code
    agent_path = None
    ################################
    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        # TODO(student): Implement offline training loop
        # Hint: You might consider passing the agent's path to the online training loop
        ################################
        # answer code
        agent_path = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        ################################
        start_step = args.offline_training_steps
        
    
    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        # TODO(student): Implement online training loop
        ################################
        # answer code
        run_online_training_loop(config, train_logger, eval_logger, args, agent_path, start_step=start_step)
        ################################


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        ################################
        # answer code
        main_njobs(job_specs=args.job_specs, njobs=args.njobs, entrypoint="train_offline_online")
        ################################
    else:
        # Run a single job
        main(args)
