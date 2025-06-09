import logging
import time
from contextlib import nullcontext
from pprint import pformat
from termcolor import colored

import torch
from torch.amp import GradScaler
from torch.nn import Sequential
import numpy as np

from lerobot.common.datasets.online_buffer import OnlineBuffer
from lerobot.common.datasets.sampler import RejectionSampler
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.scripts.train import update_policy
from gymnasium import make_vec as gym_make_vec

@parser.wrap()
def train_online(cfg: TrainPipelineConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    #--------------------------------------------------------------------------
    # Hacking the TD-MPC implementation for the NO LATENT SPACE experiment
    #--------------------------------------------------------------------------
    logging.info("Modifying TD-MPC config for NO LATENT SPACE experiment")
    cfg.policy.latent_dim = cfg.env.state_numel
    cfg.policy.normalization_mapping[FeatureType.ACTION] = NormalizationMode.IDENTITY

    logging.info("Creating and modifying TD-MPC policy for NO LATENT SPACE experiment")
    policy = make_policy(cfg.policy, env_cfg=cfg.env)
    del policy.model._encoder
    del policy.model_target._encoder
    policy.model.encode = lambda x: x["observation.state"]
    policy.model_target.encode = lambda x: x["observation.state"]
    policy.model._dynamics = Sequential(*list(policy.model._dynamics.children())[:-2])
    policy.model_target._dynamics = Sequential(*list(policy.model_target._dynamics.children())[:-2])
    #--------------------------------------------------------------------------

    cfg.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg._save_pretrained(cfg.output_dir)

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating training env")
    env = gym_make_vec(f"{cfg.env.module}:{cfg.env.task}",
                       wrappers=cfg.env.wrappers if hasattr(cfg.env, "wrappers") else None,
                       disable_env_checker=True, **cfg.env.gym_kwargs)
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating eval envs")
        eval_env = gym_make_vec(f"{cfg.env.module}:{cfg.env.task}", 
                                num_envs=cfg.eval.batch_size, vectorization_mode="async" if cfg.eval.use_async_envs else "sync",
                                wrappers=cfg.env.wrappers if hasattr(cfg.env, "wrappers") else None,
                                disable_env_checker=True, **cfg.env.gym_kwargs)

    if cfg.seed is not None:
        env.action_space.seed(cfg.seed)
        env.reset(seed=cfg.seed)
        if eval_env:
            eval_env.action_space.seed(cfg.seed)
            eval_env.reset(seed=cfg.seed)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    policy.train()

    logging.info("Creating replay buffer")
    buffer = make_online_buffer(cfg)
    dataloader = torch.utils.data.DataLoader(
        buffer,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=RejectionSampler(buffer, buffer._data["last"]),
        pin_memory=device.type != "cpu"
    )
    dl_iter = iter(dataloader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "episode_reward": AverageMeter("ep_rwrd", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, buffer._buffer_capacity, env.spec.max_episode_steps, train_metrics, initial_step=0)

    step = 0
    while step < cfg.steps:
        if step == 0:
            logging.info("Populating buffer with random trajectories...")
            while buffer.num_frames < cfg.seed_steps:
                episode, _ = roll_single_episode(env)
                buffer.add_data(episode)
            num_updates_to_do = cfg.seed_steps
            logging.info(f"Buffer populated with {buffer.num_frames} transitions ({cfg.seed_steps=}) corresponding to {buffer.num_episodes} episodes. Accordingly, the policy will be updated {num_updates_to_do} times...")
        
        else:
            if step == cfg.seed_steps: logging.info("Starting online training...")
            episode, episode_length = roll_single_episode(env, policy)
            buffer.add_data(episode)
            num_updates_to_do = episode_length
        
        for _ in range(num_updates_to_do):
            start_time = time.perf_counter()
            batch = next(dl_iter)
            train_tracker.dataloading_s = time.perf_counter() - start_time
            
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp
            )
            train_tracker.episode_reward = episode["next.reward"][:-1].sum()

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            # Log progress
            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            # Save checkpoint
            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # Evaluate policy
            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                policy.eval()
                with torch.no_grad(), torch.autocast(device_type=policy.device.type) if cfg.policy.use_amp else nullcontext():
                    eval_info = eval_policy(
                        eval_env, policy, cfg.eval.n_episodes, 
                        start_seed=cfg.seed, 
                        max_episodes_rendered=2, 
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",)
                policy._prev_mean = None
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, 1, 1, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict()} #, **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
                policy.train()
    
    env.close()
    if eval_env:
        eval_env.close()
    logging.info("End of training")

def make_online_buffer(cfg: TrainPipelineConfig):
    delta_timestamps = {
        'observation.state': cfg.policy.observation_delta_indices,
        'action': cfg.policy.action_delta_indices,
        'next.reward': cfg.policy.reward_delta_indices,
    }
    data_spec = {
            'observation.state': {'shape': cfg.env.features['flattened_state'].shape, 'dtype': np.dtype('float32')},
            'action': {'shape': cfg.env.features['action'].shape, 'dtype': np.dtype('float32')},
            'next.reward': {'shape': (), 'dtype': np.dtype('float32')},
            'next.terminated': {'shape': (), 'dtype': np.dtype('bool')},
            'next.truncated': {'shape': (), 'dtype': np.dtype('bool')},
            'next.done': {'shape': (), 'dtype': np.dtype('bool')},
            'last': {'shape': (), 'dtype': np.dtype('bool')},
    }
    buffer = OnlineBuffer(
        write_dir=cfg.output_dir/"buffer",
        data_spec=data_spec,
        buffer_capacity=np.clip(cfg.steps//5, 1_000, 1_000_000), # TODO: deal with this
        fps=1, #cfg.env.fps,
        delta_timestamps=delta_timestamps
    )
    return buffer

def roll_single_episode(env, policy=None):
    if policy is None:
        select_action = lambda x: env.action_space.sample()
    else:
        device = next(policy.parameters()).device
        select_action = lambda obs: policy.select_action( {"observation.state": torch.from_numpy(obs).to(device)} ).numpy(force=True)
    
    episode = []
    obs, info = env.reset()
    episode_step = 0
    done = False
    while not done:
        action = select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode.append({
            "observation.state": obs.squeeze(0),
            "action": action.squeeze(0),
            "next.reward": reward.squeeze(0),
            "next.terminated": terminated.squeeze(0),
            "next.truncated": truncated.squeeze(0),
            "next.done": (done := terminated | truncated).squeeze(0),
            "last": False,
        })

        episode_step += 1
        obs = next_obs
    
    episode.append({
        "observation.state": obs.squeeze(0),
        "action": np.zeros_like(action.squeeze(0)),  # will be masked (shouldn't be nan or inf)
        "next.reward": 0,                            # will be masked (shouldn't be nan or inf)
        "next.terminated": True,
        "next.truncated": True,
        "next.done": True,
        "last": True,
    })

    episode_length = len(episode)
    episode = {k:np.stack([t[k] for t in episode]) for k in episode[0].keys()}
    episode["timestamp"] = np.arange(episode_length, dtype=np.float64) # / env.metadata["render_fps"]
    episode["index"] = np.arange(episode_length)
    episode["episode_index"] = np.zeros(episode_length, dtype=np.int64)
    episode["frame_index"] = -np.ones(episode_length, dtype=np.int64)
    return episode, episode_length

if __name__ == "__main__":
    init_logging()
    train_online()