from __future__ import annotations

################################
# answer code
"""Train ppo using the shared learned-reward rollout/evaluation pipeline."""
################################


def main() -> None:
    ################################
    # answer code
    from llm_rl_final_proj.online.train_rm_grpo import main as train
    train(default_algo="ppo")
    ################################


if __name__ == "__main__":
    main()
