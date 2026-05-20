import time

import numpy as np
import torch
import wandb


class Evaluator:
    def __init__(self, env, backbone, n_eval_episodes: int = 20, max_steps: int = 200):
        self.env = env
        self.backbone = backbone
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps

    def evaluate(self, policy, step: int) -> dict[str, float]:
        successes = []
        inference_times_ms = []

        for _ in range(self.n_eval_episodes):
            img, instruction = self.env.reset()
            done = False
            success = False

            for _t in range(self.max_steps):
                if done:
                    break

                with torch.no_grad():
                    cond = self.backbone.encode([img], [instruction])

                    start = time.perf_counter()
                    action = policy(cond)
                    end = time.perf_counter()

                inference_times_ms.append((end - start) * 1000.0)

                action_np = action.squeeze(0).detach().cpu().numpy()

                try:
                    img, reward, done, info = self.env.step(action_np)
                except ValueError as e:
                    if "terminated episode" in str(e):
                        done = True
                        break
                    raise

                success = (
                    success
                    or bool(info.get("success", False))
                    or bool(info.get("is_success", False))
                    or bool(reward > 0)
                )

                if done:
                    break

            successes.append(float(success))

        metrics = {
            "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
            "eval/inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        }

        wandb.log(metrics, step=step)

        return metrics