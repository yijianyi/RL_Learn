################################
# answer code
from __future__ import annotations

"""CPU checks for the answer algorithms, without Gym/HF datasets or model downloads.

The source loader injects tensor dependencies and executes production definitions.
It removes import-time integrations and compilation decorators so the same eager
checks can also run in an older, locally available PyTorch environment.
This is not an integration test of the assignment environments or torch.compile.
"""

import abc
import ast
import copy
import dataclasses
import itertools
import math
from pathlib import Path
import sys
import tempfile
import types
import typing
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class EagerDefinitions(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node.decorator_list = [
            d for d in node.decorator_list
            if ast.unparse(d) not in ("torch.compile", "torch.compiler.disable")
        ]
        return node


def source(path, names=None, **dependencies):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if names is not None:
        tree.body = [n for n in tree.body if getattr(n, "name", None) in names]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    tree = ast.fix_missing_locations(EagerDefinitions().visit(tree))
    module = types.ModuleType("answer_test_" + str(len(sys.modules)))
    sys.modules[module.__name__] = module
    module.__dict__.update({k: v for k, v in vars(typing).items() if not k.startswith("__")})
    module.__dict__.update(torch=torch, nn=nn, F=F, D=torch.distributions,
                           distributions=torch.distributions, np=np, abc=abc,
                           math=math, itertools=itertools, copy=copy,
                           optim=torch.optim, dataclass=dataclasses.dataclass,
                           replace=dataclasses.replace, PolicyModel=nn.Module,
                           PreferenceBatch=object, TypeAlias=typing.Any)
    module.__dict__.update(dependencies)
    exec(compile(tree, str(ROOT / path), "exec"), module.__dict__)
    return module


def ptu_for(prefix):
    module = source(prefix + "/infrastructure/pytorch_util.py")
    module.device = torch.device("cpu")
    return module


def distributions():
    return source("hw5/src/infrastructure/distributions.py",
                  names={"make_multi_normal", "make_tanh_transformed"})


def networks(prefix, ptu):
    dist = distributions()
    return source(prefix + "/networks/rl_networks.py", ptu=ptu,
                  make_multi_normal=dist.make_multi_normal,
                  make_tanh_transformed=dist.make_tanh_transformed)


def finite_metrics(metrics):
    return all(math.isfinite(float(v)) for v in metrics.values())


class ImitationAndPG(unittest.TestCase):
    def test_mse_learns_and_flow_shapes_and_euler(self):
        torch.manual_seed(2)
        m = source("hw1/src/hw1_imitation/model.py")
        policy = m.MSEPolicy(3, 2, 4, (32, 32))
        states = torch.randn(32, 3)
        target = (states[:, :2] * 0.5).unsqueeze(1).expand(-1, 4, -1)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
        initial = policy.compute_loss(states, target).item()
        for _ in range(80):
            loss = policy.compute_loss(states, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(policy.compute_loss(states, target).item(), initial * 0.05)
        flow = m.FlowMatchingPolicy(3, 2, 4, (16,))
        flow.compute_loss(states, target).backward()
        self.assertTrue(any(p.grad is not None for p in flow.parameters()))
        self.assertEqual(flow.sample_actions(states, num_steps=3).shape, (32, 4, 2))

        class Constant(nn.Module):
            def forward(self, x):
                return x.new_full((x.shape[0], 8), 2)

        flow.velocity = Constant()
        torch.manual_seed(3)
        expected = torch.randn(32, 8).reshape(32, 4, 2) + 2
        torch.manual_seed(3)
        torch.testing.assert_close(flow.sample_actions(states, num_steps=5), expected)
        with self.assertRaises(ValueError):
            flow.sample_actions(states, num_steps=0)

    def pg(self, discrete=True, gae=None, normalize=False):
        ptu = ptu_for("hw2/src")
        policies = source("hw2/src/networks/policies.py", ptu=ptu)
        critics = source("hw2/src/networks/critics.py", ptu=ptu)
        agent = source("hw2/src/agents/pg_agent.py", ptu=ptu,
                       MLPPolicyPG=policies.MLPPolicyPG, ValueCritic=critics.ValueCritic).PGAgent(
            2, 2, discrete, 1, 16, 0.9, 0.01, True, True, 0.01, 2, gae, normalize)
        return agent

    def test_returns_and_terminal_gae(self):
        agent = self.pg(gae=1.0)
        np.testing.assert_allclose(agent._discounted_return([1, 2, 3]), [5.23] * 3)
        np.testing.assert_allclose(agent._discounted_reward_to_go([1, 2, 3]), [5.23, 4.7, 3])

        class FixedValues(nn.Module):
            def forward(self, obs):
                return obs[:, 0]

        agent.critic = FixedValues()
        obs = np.array([[0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0]], np.float32)
        rewards = np.array([1, 2, 3, 4], np.float32)
        expected = np.array([2.8, 2, 6.6, 4]) - obs[:, 0]
        advantages = agent._estimate_advantage(obs, rewards, np.zeros(4), np.array([0, 1, 0, 1]))
        np.testing.assert_allclose(advantages, expected, rtol=1e-6)

    def test_discrete_and_continuous_pg_update(self):
        for discrete in (True, False):
            agent = self.pg(discrete=discrete, normalize=True)
            observations = [np.random.randn(3, 2).astype(np.float32) for _ in range(2)]
            actions = [np.array([0, 1, 0]) if discrete else np.random.randn(3, 2).astype(np.float32)
                       for _ in range(2)]
            before = [p.detach().clone() for p in agent.actor.parameters()]
            result = agent.update(observations, actions, [np.array([1, 2, 3], np.float32)] * 2,
                                  [np.array([0, 0, 1], np.float32)] * 2)
            self.assertTrue(finite_metrics(result))
            self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before, agent.actor.parameters())))
            action = agent.actor.get_action(observations[0][0])
            self.assertEqual(np.asarray(action).shape, () if discrete else (2,))

    def test_rollout_both_gym_apis_and_length_limit(self):
        functions = source("hw2/src/infrastructure/utils.py", names={"sample_trajectory"})

        class Env:
            def __init__(self, modern):
                self.modern = modern

            def reset(self):
                return (np.zeros(2), {}) if self.modern else np.zeros(2)

            def step(self, action):
                if self.modern:
                    return np.zeros(2), 1.0, False, False, {}
                return np.zeros(2), 1.0, False, {}

        policy = types.SimpleNamespace(get_action=lambda obs: 0)
        for modern in (True, False):
            trajectory = functions.sample_trajectory(Env(modern), policy, 3)
            np.testing.assert_array_equal(trajectory["terminal"], [0, 0, 1])
            self.assertEqual(trajectory["observation"].shape, (3, 2))


class QLearning(unittest.TestCase):
    def test_dqn_and_double_dqn_targets(self):
        ptu = ptu_for("hw3/src")
        cls = source("hw3/src/agents/dqn_agent.py", ptu=ptu).DQNAgent
        for double, expected in ((False, 6.0), (True, 1.5)):
            agent = cls((2,), 2, lambda shape, dim: nn.Linear(2, 2),
                        lambda p: torch.optim.SGD(p, lr=0.01),
                        lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1),
                        0.9, 2, use_double_q=double)
            with torch.no_grad():
                agent.critic.weight.zero_()
                agent.critic.bias.copy_(torch.tensor([1., 2.]))
                agent.target_critic.weight.zero_()
                agent.target_critic.bias.copy_(torch.tensor([10., 0.]))
            self.assertEqual(agent.get_action(np.zeros(2, np.float32)), 1)
            stats = agent.update_critic(torch.zeros(2, 2), torch.tensor([0, 1]),
                                        torch.tensor([1., 2.]), torch.zeros(2, 2), torch.tensor([0., 1.]))
            self.assertAlmostEqual(stats["target_values"], expected)
            self.assertTrue(all(p.grad is None for p in agent.target_critic.parameters()))

    def test_sac_updates_backups_and_temperature_direction(self):
        ptu = ptu_for("hw3/src")
        dist = distributions()
        policies = source("hw3/src/networks/policies.py", ptu=ptu,
                          make_tanh_transformed=dist.make_tanh_transformed,
                          make_multi_normal=dist.make_multi_normal)
        critics = source("hw3/src/networks/critics.py", ptu=ptu)
        cls = source("hw3/src/agents/sac_agent.py", ptu=ptu).SoftActorCritic
        for entropy_bonus, autotune in ((False, False), (True, False), (True, True)):
            agent = cls((3,), 2,
                        lambda shape, dim: policies.MLPPolicy(dim, 3, False, 1, 16, use_tanh=True),
                        lambda p: torch.optim.Adam(p, lr=0.001),
                        lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1),
                        lambda shape, dim: critics.StateActionCritic(3, dim, 1, 16),
                        lambda p: torch.optim.Adam(p, lr=0.001),
                        lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1),
                        0.99, soft_target_update_rate=0.05, num_critic_updates=2,
                        num_critic_networks=2, target_critic_backup_type="min",
                        use_entropy_bonus=entropy_bonus, temperature=0.1, auto_tune_temperature=autotune)
            torch.testing.assert_close(agent.q_backup_strategy(torch.tensor([[1., 4.], [2., 3.]])),
                                       torch.tensor([[1., 3.], [1., 3.]]))
            before = [p.detach().clone() for p in agent.actor.parameters()]
            stats = agent.update(torch.randn(8, 3), torch.randn(8, 2).tanh(), torch.randn(8),
                                  torch.randn(8, 3), torch.zeros(8), 1)
            self.assertTrue(finite_metrics(stats))
            self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before, agent.actor.parameters())))
            self.assertTrue(all(p.grad is None for p in agent.target_critics.parameters()))
            if autotune:
                old_alpha = agent.get_temperature()
                agent.update_alpha(torch.full((8,), 10.0))
                self.assertGreater(agent.get_temperature(), old_alpha)


class OfflineRL(unittest.TestCase):
    def test_parallel_training_dispatch_preserves_entrypoint(self):
        calls = []
        def import_training(name):
            calls.append(name)
            return types.SimpleNamespace(setup_arguments=lambda args: args,
                                         main=lambda args: calls.append(args))
        workers = source("final_project_offline_online/problem/src/scripts/run_njobs.py", names={"_worker"},
                         importlib=types.SimpleNamespace(import_module=import_training), shlex=__import__("shlex"))
        workers._worker("JOB --offline_training_steps 2 --online_training_steps 3", "train_offline_online")
        self.assertEqual(calls[0], "scripts.train_offline_online")
        self.assertEqual(calls[1], ["--offline_training_steps", "2", "--online_training_steps", "3"])

    def test_project_config_factories_construct_agents(self):
        prefix = "final_project_offline_online/problem/src"
        ptu = ptu_for(prefix)
        nets = networks(prefix, ptu)
        dependencies = {name: value for name, value in vars(nets).items() if isinstance(value, type)}
        for algorithm, clsname in (("ifql", "IFQLAgent"), ("qsm", "QSMAgent"), ("dsrl", "DSRLAgent")):
            with self.subTest(algorithm=algorithm):
                config_module = source(prefix + f"/configs/{algorithm}_config.py",
                                       names={f"{algorithm}_config"}, **dependencies)
                config = getattr(config_module, f"{algorithm}_config")("synthetic", hidden_size=16, num_layers=1)
                agent_module = source(prefix + f"/agents/{algorithm}_agent.py", ptu=ptu)
                agent = getattr(agent_module, clsname)((3,), 2, **config["agent_kwargs"])
                self.assertIsInstance(agent, nn.Module)
                self.assertTrue(any(isinstance(m, nn.LayerNorm) for m in agent.critic.modules()))

    def test_offline_online_checkpoint_mixing_and_time_limits(self):
        prefix = "final_project_offline_online/problem/src"
        ptu = ptu_for(prefix)
        replay = source(prefix + "/infrastructure/replay_buffer.py").ReplayBuffer
        updates = []
        environments = []

        class Agent(nn.Module):
            def __init__(self, shape, dim):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(()))
                self.online_training = False

            def get_action(self, obs):
                return np.zeros(2, np.float32)

            def update(self, obs, actions, rewards, next_obs, dones, step):
                updates.append((step, obs.clone(), dones.clone(), self.online_training))
                with torch.no_grad():
                    self.weight.add_(1)
                return {"loss": self.weight.item()}

        class Env:
            spec = types.SimpleNamespace(max_episode_steps=3)
            action_space = types.SimpleNamespace(low=-np.ones(2), high=np.ones(2),
                                                 sample=lambda: np.zeros(2, np.float32))

            def reset(self, seed=None):
                self.steps = 0
                return np.zeros(3, np.float32), {}

            def step(self, action):
                self.steps += 1
                # A time limit must still bootstrap the critic.
                return np.zeros(3, np.float32), 1., False, self.steps == 2, {"success": True}

            def close(self):
                self.closed = True

        def dataset_sample(n):
            return dict(observations=np.full((n, 3), 42, np.float32), actions=np.zeros((n, 2), np.float32),
                        rewards=np.ones(n, np.float32), next_observations=np.zeros((n, 3), np.float32),
                        dones=np.ones(n, np.float32))

        def make_env_and_dataset():
            env = Env()
            environments.append(env)
            return env, types.SimpleNamespace(sample=dataset_sample)

        class Logger:
            def __init__(self):
                self.rows = []

            def log(self, row, step):
                self.rows.append((step, row))

        def dump(agent, train_logger, eval_logger, config, save_dir):
            path = str(Path(save_dir) / "agent.pt")
            torch.save(agent.state_dict(), path)
            return path

        # The available PyTorch 1.12 predates weights_only; production targets 2.x.
        compat = types.SimpleNamespace(**vars(torch))
        def load(path, **kwargs):
            if int(torch.__version__.split(".")[0]) < 2:
                kwargs.pop("weights_only", None)
            return torch.load(path, **kwargs)
        compat.load = load
        names = {"_make_agent_and_dataset", "_evaluate", "_update", "run_offline_training_loop", "run_online_training_loop"}
        functions = source(prefix + "/scripts/train_offline_online.py", names=names,
                           torch=compat, ptu=ptu, ReplayBuffer=replay, agents={"fake": Agent},
                           dump_log=dump, tqdm=types.SimpleNamespace(trange=lambda *a, **k: range(*a)))
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(seed=1, no_gpu=True, which_gpu=0, num_eval_trajectories=1,
                                         offline_training_steps=2, online_training_steps=3,
                                         log_interval=1, eval_interval=2, save_dir=directory,
                                         replay_buffer_capacity=10, online_random_steps=0,
                                         online_warmup_steps=1, online_updates_per_step=2, offline_ratio=0.5)
            config = dict(agent="fake", agent_kwargs={}, make_env_and_dataset=make_env_and_dataset, batch_size=6)
            train_logger, eval_logger = Logger(), Logger()
            checkpoint = functions.run_offline_training_loop(config, train_logger, eval_logger, args)
            functions.run_online_training_loop(config, train_logger, eval_logger, args, checkpoint, start_step=2)
            self.assertEqual([row[0] for row in updates], [1, 2, 4, 4, 5, 5])
            for _, obs, dones, online in updates[2:]:
                self.assertTrue(online)
                self.assertEqual((obs[:, 0] == 42).sum().item(), 3)
                self.assertEqual(dones.sum().item(), 3)
            self.assertTrue(all(env.closed for env in environments))
            self.assertEqual(eval_logger.rows[-1][0], 5)

    def test_all_agents_updates_actions_and_polyak(self):
        torch.manual_seed(8)
        for prefix, algorithms in [("hw5/src", ("sacbc", "iql", "fql")),
                                   ("final_project_offline_online/problem/src", ("sacbc", "fql", "ifql", "qsm", "dsrl"))]:
            ptu = ptu_for(prefix)
            nets = networks(prefix, ptu)
            optimizer = lambda p: torch.optim.Adam(p, lr=0.001)
            critic = lambda shape, dim: nets.EnsembleCritic(3, dim, 1, 16, 2)
            flow = lambda shape, dim: nets.VectorFieldPolicy(dim, 3, 1, 16)
            policy = lambda shape, dim: nets.Policy(dim, 3, False, 1, 16, use_tanh=True, state_dependent_std=True)
            for algorithm in algorithms:
                with self.subTest(prefix=prefix, algorithm=algorithm):
                    module = source(prefix + f"/agents/{algorithm}_agent.py", ptu=ptu)
                    cls = getattr(module, {"sacbc": "SACBCAgent", "iql": "IQLAgent", "fql": "FQLAgent",
                                           "ifql": "IFQLAgent", "qsm": "QSMAgent", "dsrl": "DSRLAgent"}[algorithm])
                    kwargs = dict(make_critic=critic, make_critic_optimizer=optimizer,
                                  discount=0.99, target_update_rate=0.05)
                    if algorithm in ("sacbc", "iql"):
                        kwargs.update(make_actor=policy, make_actor_optimizer=optimizer, alpha=1.)
                    if algorithm == "sacbc":
                        kwargs.update(make_beta=nets.LogParam, make_beta_optimizer=optimizer)
                    if algorithm == "iql":
                        kwargs.update(make_actor=lambda shape, dim: nets.Policy(dim, 3, False, 1, 16, fixed_std=True),
                                      make_value=lambda shape: nets.Value(3, 1, 16), make_value_optimizer=optimizer, expectile=0.8)
                    if algorithm == "fql":
                        kwargs.update(make_bc_actor=flow, make_bc_actor_optimizer=optimizer,
                                      make_onestep_actor=flow, make_onestep_actor_optimizer=optimizer, flow_steps=3, alpha=1.)
                    if algorithm == "ifql":
                        kwargs.update(make_actor_flow=flow, make_actor_flow_optimizer=optimizer,
                                      make_value=lambda shape: nets.Value(3, 1, 16), make_value_optimizer=optimizer,
                                      flow_steps=3, num_samples=4)
                    if algorithm == "qsm":
                        kwargs.update(make_actor=flow, make_actor_optimizer=optimizer, flow_steps=3, alpha=1., inv_temp=1.)
                    if algorithm == "dsrl":
                        kwargs.update(make_bc_flow_actor=flow, make_bc_flow_actor_optimizer=optimizer,
                                      make_noise_actor=policy, make_noise_actor_optimizer=optimizer,
                                      make_z_critic=critic, make_z_critic_optimizer=optimizer, flow_steps=3)
                    agent = cls((3,), 2, **kwargs)
                    for step in range(2):
                        stats = agent.update(torch.randn(8, 3), torch.randn(8, 2).tanh(), torch.randn(8),
                                              torch.randn(8, 3), torch.tensor([0., 1.] * 4), step)
                        self.assertTrue(finite_metrics(stats), stats)
                    action = agent.get_action(np.zeros(3, np.float32))
                    self.assertEqual(action.shape, (2,))
                    self.assertTrue(np.isfinite(action).all())
                    self.assertTrue((np.abs(action) <= 1.00001).all())
                    self.assertTrue(all(p.grad is None for p in agent.target_critic.parameters()))
                    with torch.no_grad():
                        for p in agent.critic.parameters():
                            p.fill_(1)
                        for p in agent.target_critic.parameters():
                            p.zero_()
                    agent.update_target_critic()
                    for p in agent.target_critic.parameters():
                        torch.testing.assert_close(p, torch.full_like(p, 0.05))
                    if algorithm == "dsrl":
                        agent.online_training = True
                        before = [p.detach().clone() for p in agent.bc_flow_actor.parameters()]
                        agent.update(torch.randn(8, 3), torch.randn(8, 2).tanh(), torch.randn(8),
                                     torch.randn(8, 3), torch.zeros(8), 3)
                        self.assertTrue(all(torch.equal(old, p) for old, p in zip(before, agent.bc_flow_actor.parameters())))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 8)
        self.head = nn.Linear(8, 11)
        self.config = types.SimpleNamespace(use_cache=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        return types.SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class LLMAlgorithms(unittest.TestCase):
    def test_online_extension_cli_defaults(self):
        module = source("final_project_llm_rl/llm_rl_final_proj/online/train_rm_grpo.py",
                        names={"OnlineRMGRPOConfig", "parse_args"}, argparse=__import__("argparse"))
        argv = sys.argv
        try:
            sys.argv = ["test", "--reward_adapter_path", "synthetic-adapter"]
            for algorithm in ("grpo", "dr_grpo", "gspo", "ppo", "online_dpo"):
                cfg = module.parse_args(default_algo=algorithm)
                self.assertEqual(cfg.algo, algorithm)
                self.assertEqual(cfg.max_new_tokens, 256)
        finally:
            sys.argv = argv

    def helpers(self, prefix):
        logp = source(prefix + "/models/logprobs.py")
        buffer = source(prefix + "/rollout/rollout_buffer.py")
        base = source(prefix + "/rl/base.py", RolloutBatch=buffer.RolloutBatch)
        utils = source(prefix + "/utils/torch_utils.py")
        deps = {k: v for k, v in vars(logp).items() if callable(v)}
        deps.update(RLAlgorithm=base.RLAlgorithm, RolloutBatch=buffer.RolloutBatch,
                    iter_minibatches=buffer.iter_minibatches, clip_grad_norm_=utils.clip_grad_norm_)
        return logp, buffer, base, deps

    def rollout(self, model, logp, buffer):
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 0], [1, 2, 3, 8, 9],
                            [1, 2, 5, 6, 0], [1, 2, 7, 8, 9]])
        attention = (ids != 0).long()
        mask = logp.build_completion_mask(ids, attention, 2, 0)
        old = logp.compute_per_token_logprobs(model, ids, attention, enable_grad=False)
        return buffer.RolloutBatch(ids, attention, mask, old, old.clone(),
                                   torch.tensor([0., 1., 2., -1., 0.5]), torch.tensor([-1., 1., 1., -1., 0.5]),
                                   task_names=list(range(5)), completion_texts=[str(i) for i in range(5)])

    def test_logprob_mask_kl_and_minibatch_alignment(self):
        for prefix in ("hw4/hw4", "final_project_llm_rl/llm_rl_final_proj"):
            logp, buffer, _, _ = self.helpers(prefix)
            model = TinyLM()
            batch = self.rollout(model, logp, buffer)
            result = logp.compute_per_token_logprobs(model, batch.input_ids, batch.attention_mask)
            logits = model(batch.input_ids, batch.attention_mask).logits[:, :-1]
            expected = logits.log_softmax(-1).gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
            torch.testing.assert_close(result, expected)
            self.assertFalse(batch.old_logprobs.requires_grad)
            self.assertTrue(result.requires_grad)
            torch.testing.assert_close(batch.completion_mask[1], torch.tensor([0., 1., 1., 0.]))
            self.assertEqual(logp.approx_kl_from_logprobs(result, result, batch.completion_mask).item(), 0)
            self.assertEqual(logp.approx_kl_from_logprobs(result, result - 1, torch.zeros_like(result)).item(), 0)
            kl = logp.approx_kl_from_logprobs(result, result - 1, batch.completion_mask)
            self.assertAlmostEqual(kl.item(), math.exp(-1), places=5)
            batches = list(buffer.iter_minibatches(batch, 2, generator=torch.Generator().manual_seed(7)))
            self.assertEqual([len(b.rewards) for b in batches], [2, 2, 1])
            seen = []
            for mb in batches:
                for i, original in enumerate(mb.task_names):
                    seen.append(original)
                    self.assertEqual(mb.completion_texts[i], str(original))
                    torch.testing.assert_close(mb.input_ids[i], batch.input_ids[original])
            self.assertEqual(sorted(seen), list(range(5)))

    def test_group_normalization_and_reward_model_gradients(self):
        hw = source("hw4/hw4/train.py", names={"compute_group_advantages", "maybe_normalize_advantages"})
        rewards = torch.tensor([1., 3., 5., 5.])
        torch.testing.assert_close(hw.compute_group_advantages(rewards, 2), torch.tensor([-1., 1., 0., 0.]), atol=2e-6, rtol=0)
        torch.testing.assert_close(hw.compute_group_advantages(rewards, 1), torch.zeros(4))
        with self.assertRaises(ValueError):
            hw.compute_group_advantages(rewards, 3)
        online = source("final_project_llm_rl/llm_rl_final_proj/online/train_rm_grpo.py",
                        names={"_compute_group_advantages", "_algo_divides_advantages_by_std"})
        torch.testing.assert_close(online._compute_group_advantages(torch.tensor([0., 4.]), 2, divide_by_std=False),
                                   torch.tensor([-2., 2.]))
        self.assertFalse(online._algo_divides_advantages_by_std("dr_grpo"))
        self.assertTrue(online._algo_divides_advantages_by_std("gspo"))
        rm = source("final_project_llm_rl/llm_rl_final_proj/reward_model/train.py", names={"_compute_pair_metrics"})
        chosen = torch.zeros(3, requires_grad=True)
        rejected = torch.zeros(3, requires_grad=True)
        result = rm._compute_pair_metrics(chosen, rejected)
        self.assertAlmostEqual(result["loss_tensor"].item(), math.log(2), places=6)
        result["loss_tensor"].backward()
        self.assertTrue((chosen.grad < 0).all())
        self.assertTrue((rejected.grad > 0).all())

    def test_offline_preference_analytic_objectives(self):
        module = source("final_project_llm_rl/llm_rl_final_proj/offline/losses.py")
        chosen = torch.tensor([1., 2., 3.], requires_grad=True)
        rejected = torch.zeros(3, requires_grad=True)
        policy = module.SequenceScores(chosen, rejected, chosen / 2, rejected)
        ref = module.SequenceScores(torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3))
        for algo in ("dpo", "ipo", "aot"):
            result = module.compute_offline_preference_loss(algo=algo, beta=0.5, policy_scores=policy, reference_scores=ref)
            expected = ((chosen - 1).square().mean() if algo == "ipo" else -F.logsigmoid(0.5 * chosen).mean())
            torch.testing.assert_close(result.loss, expected)
            result.loss.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(chosen.grad).all())
        # AOT compares sorted distributions, independently of row order.
        permuted = module.SequenceScores(chosen.flip(0), rejected, chosen.flip(0) / 2, rejected)
        first = module.compute_offline_preference_loss(algo="aot", beta=0.5, policy_scores=policy, reference_scores=ref)
        second = module.compute_offline_preference_loss(algo="aot", beta=0.5, policy_scores=permuted, reference_scores=ref)
        torch.testing.assert_close(first.loss, second.loss)

    def test_all_llm_updates_and_accumulation(self):
        for prefix in ("hw4/hw4", "final_project_llm_rl/llm_rl_final_proj"):
            logp, buffer, base, deps = self.helpers(prefix)
            grpo = source(prefix + "/rl/grpo.py", **deps)
            algorithms = [(grpo.GRPO, {})]
            reinforce = source(prefix + "/rl/reinforce.py", **deps)
            algorithms.append((reinforce.Reinforce, {}))
            if prefix.startswith("final"):
                for filename, clsname in (("dr_grpo", "DrGRPO"), ("gspo", "GSPO"), ("ppo", "PPO")):
                    module = source(prefix + f"/rl/{filename}.py", GRPO=grpo.GRPO, **deps)
                    algorithms.append((getattr(module, clsname), {}))
            for cls, kwargs in algorithms:
                with self.subTest(prefix=prefix, algorithm=cls.__name__):
                    model = TinyLM()
                    batch = self.rollout(model, logp, buffer)
                    cfg = base.AlgoConfig(ppo_epochs=2, minibatch_size=2)
                    algo = cls(cfg, **kwargs)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    before = model.head.weight.detach().clone()
                    stats = algo.update(model, optimizer, batch, grad_accum_steps=2)
                    self.assertTrue(finite_metrics(stats))
                    self.assertFalse(torch.equal(before, model.head.weight))
                    expected_steps = 2 if cls.__name__ == "Reinforce" else 3
                    self.assertEqual(stats["train/count_optimizer_steps_per_training_iteration"], expected_steps)

    def test_online_dpo_pairs_and_update(self):
        prefix = "final_project_llm_rl/llm_rl_final_proj"
        logp, buffer, base, deps = self.helpers(prefix)
        module = source(prefix + "/rl/online_dpo.py", **deps)
        model = TinyLM()
        batch = self.rollout(model, logp, buffer)
        # Use one group of five completions; best and worst rewards differ.
        algo = module.OnlineDPO(base.AlgoConfig(ppo_epochs=2), group_size=5)
        stats = algo.update(model, torch.optim.Adam(model.parameters(), lr=0.001), batch, 2)
        self.assertTrue(finite_metrics(stats))
        self.assertEqual(stats["train/online_dpo_pairs"], 1)
        self.assertEqual(stats["train/count_optimizer_steps_per_training_iteration"], 1)

    def test_grpo_family_clipping_and_length_normalization(self):
        prefix = "final_project_llm_rl/llm_rl_final_proj"
        _, _, base, deps = self.helpers(prefix)
        grpo = source(prefix + "/rl/grpo.py", **deps).GRPO
        dr = source(prefix + "/rl/dr_grpo.py", GRPO=grpo, **deps).DrGRPO
        gspo = source(prefix + "/rl/gspo.py", GRPO=grpo, **deps).GSPO
        cfg = base.AlgoConfig(clip_eps=0.2, clip_eps_high=0.4, max_completion_length=4)
        log_ratios = torch.tensor([[math.log(2), math.log(0.5), 50.]])
        mask, advantages = torch.tensor([[1., 1., 0.]]), torch.ones(1)
        for cls, expected in ((grpo, -0.95), (dr, -0.475), (gspo, -1.0)):
            loss, _ = cls(cfg).policy_objective(log_ratios, advantages, mask)
            self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_partial_gradient_accumulation_preserves_update_scale(self):
        for prefix in ("hw4/hw4", "final_project_llm_rl/llm_rl_final_proj"):
            logp, buffer, base, deps = self.helpers(prefix)
            for filename, clsname in (("reinforce", "Reinforce"), ("grpo", "GRPO")):
                cls = getattr(source(prefix + f"/rl/{filename}.py", **deps), clsname)
                first, second = TinyLM(), TinyLM()
                second.load_state_dict(first.state_dict())
                batch = self.rollout(first, logp, buffer)
                cfg = base.AlgoConfig(ppo_epochs=1, minibatch_size=8, max_grad_norm=10)
                cls(cfg).update(first, torch.optim.SGD(first.parameters(), lr=0.01), batch, 1)
                cls(cfg).update(second, torch.optim.SGD(second.parameters(), lr=0.01), batch, 4)
                for a, b in zip(first.parameters(), second.parameters()):
                    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
################################
