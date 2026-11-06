# Conda 环境配置

这 5 份配置用于在 **Linux x86_64 + NVIDIA GPU 服务器**上运行作业。

| 作业 | 配置文件 | 环境名 | Python | PyTorch | CUDA 构建 | NumPy |
| --- | --- | --- | --- | --- | --- | --- |
| hw1 | [hw1/environment.yml](hw1/environment.yml) | `cs285-hw1` | 3.11 | 2.9.1 | cu128 | 2.3.5 |
| hw2 | [hw2/environment.yml](hw2/environment.yml) | `cs285-hw2` | 3.10 | 2.9.1 | cu128 | 1.26.4 |
| hw3 | [hw3/environment.yml](hw3/environment.yml) | `cs285-hw3` | 3.10 | 2.10.0 | cu128 | 1.26.4 |
| hw4 | [hw4/environment.yml](hw4/environment.yml) | `cs285-hw4` | 3.12 | 2.6.0 | cu124 | 1.26.4 |
| hw5 | [hw5/environment.yml](hw5/environment.yml) | `cs285-hw5` | 3.11 | 2.9.1 | cu128 | 2.2.6 |

## 配置依据

- Python 版本遵循各作业的 `pyproject.toml` 和已有 `.python-version`；hw5 未指定具体版本，这里选择满足要求的 3.11。
- Python 直接依赖和传递依赖按各自 `uv.lock` 固定版本，并筛选对应 Python 版本、Linux x86_64 平台的依赖分支。保留 Gym、WandB 等包的 extras。
- hw4 包含 `remote` 可选依赖，因此可用于运行 LLM/LoRA 训练，而不仅是提交 Modal 任务。
- PyTorch 显式选择 NVIDIA GPU 构建；CUDA 依赖版本沿用锁文件。hw4 的 cu124 也与其 Modal 训练脚本一致。
- Conda 管理 Python、pip、setuptools、wheel 和基础库，pip 管理来自原 uv 锁文件的 Python 依赖。hw2/hw3 额外提供 SWIG、CMake 和 C++ 编译器，用于旧版 Gym/Box2D 的构建；Conda SWIG 在 pip 阶段开始前可用。
- 环境文件不包含本机路径，也不自动安装本仓库源码。Python 补丁版本、pip、wheel 和基础库未完全锁定；这些 YAML 不是包含所有二进制构建和哈希的 Conda 锁文件。

## 在服务器上创建和运行

将仓库复制到服务器，进入需要运行的作业目录。以 hw3 为例：

```bash
cd /path/to/RL_Learn/hw3
conda env create -f environment.yml
conda activate cs285-hw3

# 将该作业的源码安装到当前环境，保持 YAML 中的依赖版本。
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check

python src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml
```

其他作业同样从各自目录创建和运行，将目录名及环境名替换为对应作业。源码安装需要保留完整的作业目录。

| 作业 | 在作业目录内运行的命令示例 |
| --- | --- |
| hw1 | `python src/hw1_imitation/train.py --help` |
| hw2 | `python src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole` |
| hw3 | `python src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml` |
| hw4 | `python -m hw4.train --help` |
| hw5 | `python src/scripts/run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0` |

激活 Conda 环境后，原 README 中的 `uv run python ...` 改为 `python ...`，`uv run src/...` 改为 `python src/...`，`uv run modal ...` 改为 `modal ...`。这样运行使用的是当前 Conda 环境。

## GPU、渲染与数据

服务器需具备兼容所选 CUDA 构建的 NVIDIA 驱动。环境文件安装 CUDA 用户态依赖，不安装主机驱动。创建环境后可在服务器检查 GPU：

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

如果服务器不支持当前 CUDA 构建，应重新选择 PyTorch 构建并同步调整 YAML 中的 `nvidia-*` 和 `triton` 依赖；不要只改 PyTorch 下载地址。这些配置针对上表的平台，不是 Windows、macOS、ARM 或 CPU 专用配置。

对于 hw2/hw3/hw5，在无显示器的 NVIDIA 服务器上使用 MuJoCo 渲染时，可在运行前设置：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

hw3 的 Atari 任务还需要 ROM；hw1、hw5 等作业的数据集也需要在服务器下载。配置文件本身不执行数据集下载，也不写入 WandB、Hugging Face 或 Modal 凭证。

如果继续使用 `modal run`，Modal 容器仍由现有脚本中的 `.uv_sync()` 创建；这些 Conda 配置不会替换容器的环境定义。直接在自己的服务器训练时，使用上面的 Python 入口。

## 检查范围与参考

已静态检查 YAML 结构、依赖版本与 uv 锁文件的对应关系，以及所选 Python 版本是否满足项目要求。没有执行远端依赖求解、环境安装或 GPU 运行验证。

- [Conda 环境文件规范](https://conda.org/learn/specifications/exchange/environment-yml/)
- [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)
- [Gym 0.25.2 的 extras 与构建依赖](https://github.com/openai/gym/blob/0.25.2/setup.py)
