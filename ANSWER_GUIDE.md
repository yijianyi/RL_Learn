# answer 分支：参考答案索引

已补全 `hw1`–`hw5` 和两个期末项目的具体实现。新增或修改的 Python 代码用以下注释包围；可在编辑器中搜索 `# answer code` 定位：

```python
################################
# answer code
# 这里是补全的代码
################################
```

原有题目提示尽量保留。抽象基类中的 `NotImplementedError` 是要求子类覆盖的接口，保留这些声明；具体算法中已无待实现的异常或省略号占位。

| 目录 | 已补全的内容 | 主要位置 |
| --- | --- | --- |
| `hw1` | MSE 行为克隆、条件流匹配、Euler 采样、训练和评估循环 | `src/hw1_imitation/model.py`、`train.py` |
| `hw2` | 离散/连续策略、价值基线、策略梯度、折扣回报、reward-to-go、GAE、轨迹采样 | `src/agents/pg_agent.py`、`src/networks/`、`src/infrastructure/utils.py`、`src/scripts/run.py` |
| `hw3` | DQN、Double DQN、SAC、熵正则、温度自动调节、多 critic、训练循环 | `src/agents/`、`src/scripts/run_dqn.py`、`run_sac.py` |
| `hw4` | token 对数概率、completion mask、KL、组内优势、minibatch、REINFORCE、GRPO | `hw4/models/logprobs.py`、`hw4/rollout/rollout_buffer.py`、`hw4/rl/`、`hw4/train.py` |
| `hw5` | SAC+BC、IQL、FQL 和目标网络更新 | `src/agents/` |
| `final_project_offline_online/problem` | SAC+BC、FQL、IFQL、QSM、DSRL、网络工厂、LayerNorm、离线到在线训练、离线数据保留和在线预热 | `src/agents/`、`src/configs/`、`src/infrastructure/pytorch_util.py`、`src/scripts/train_offline_online.py` |
| `final_project_llm_rl` | DPO、IPO、AOT、Bradley–Terry 奖励模型、GRPO、DrGRPO、GSPO、共享辅助函数及可选在线算法 | `llm_rl_final_proj/offline/losses.py`、`reward_model/train.py`、`rl/`、`online/` |

离线到在线训练的本地入口是 `final_project_offline_online/problem/src/scripts/train_offline_online.py`，Modal 入口沿用 `src/scripts/modal_run.py`。`--offline_ratio` 控制在线更新中离线样本的比例，`--online_warmup_steps` 控制开始更新前收集多少在线数据。多进程入口也会使用对应的训练脚本。

实现选择：

- QSM 使用带行为克隆正则的版本：`alpha` 加权噪声预测 BC 损失，`inv_temp` 缩放 Q 对动作的梯度，策略更新不反向传播整条扩散采样链。核心方法参考 [Q-score matching 论文](https://arxiv.org/abs/2312.11752)及[作者实现](https://github.com/escontra/score_matching_rl)。
- DSRL 在离线阶段学习 BC flow，在线阶段固定它，通过噪声策略和噪声 critic 引导动作；噪声 critic 使用相互匹配的噪声与生成动作。方法参考 [DSRL 论文](https://arxiv.org/abs/2506.15799)。
- DrGRPO 使用组内去均值优势和固定生成预算归一化；GSPO 对序列内 token 的 log-ratio 求均值后计算比例并裁剪。分别参考 [DrGRPO 论文](https://arxiv.org/abs/2503.20783)和 [GSPO 论文](https://arxiv.org/abs/2507.18071)。
- 可选的 `online/train_rm_ppo.py` 实现简单的 PPO 扩展：用冻结的 prompt embedding 均值训练奖励基线，再使用 token 级 PPO 裁剪；这里没有 token value/GAE 模型。`online/train_rm_online_pref.py` 在每个 prompt 的候选组中选取奖励最高/最低的回答，执行在线 DPO。两个入口复用奖励打分、采样、评估和保存流程，支持共享训练参数；在线 DPO 额外支持 `--online_preference_beta`。

## 验证

在安装了 NumPy 和 PyTorch 的环境中，从仓库根目录运行：

```bash
python tests/test_answers.py -v
```

本次通过 **18 项 CPU 测试**，包括各个 agent 的实际参数更新、解析损失、梯度方向、采样维度、终止状态、Polyak 更新、优势归一化、minibatch 对齐、梯度累积、配置工厂、多进程入口分发，以及离线权重保存/加载和在线数据混合。另检查了全部仓库 Python 文件的语法，以及 `git diff --check`。

本机可用测试环境为 Python 3.9.25、PyTorch 1.12.0，使用 CPU。测试从源码提取定义并注入张量依赖，跳过 Gym、Hugging Face、WandB 等集成导入及 `torch.compile` 装饰器，以便验证 eager 模式中的核心逻辑；训练循环使用合成环境和数据。生产运行仍应使用各项目要求的 Python 和依赖版本。

未执行各作业的完整依赖安装、`torch.compile`/GPU 集成验证、真实模拟器训练、大模型训练或远程评分。因此这些答案尚未经过收敛性能和评分门槛验证。
