from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from src.models.flow_head import FlowMatchingHead


class BCTrainer:
    """
    Behavior cloning trainer for the flow matching head.

    The VLM backbone stays frozen. Only the flow head is trained.
    """

    def __init__(
        self,
        flow_head: FlowMatchingHead,
        backbone,
        dataloader,
        evaluator,
        cfg,
        device: torch.device,
    ):
        self.flow = flow_head.to(device)
        self.backbone = backbone
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.global_step = 0

    def _build_optimizer(self):
        if self.cfg.optimizer.name.lower() == "adamw":
            return optim.AdamW(
                self.flow.net.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")

    def _build_scheduler(self):
        if self.cfg.scheduler.name.lower() == "cosine":
            t_max = self.cfg.scheduler.t_max

            if t_max is None:
                t_max = self.cfg.training.n_epochs * len(self.dataloader)

            return CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
            )

        if self.cfg.scheduler.name.lower() in {"none", "disabled"}:
            return None

        raise ValueError(f"Unknown scheduler: {self.cfg.scheduler.name}")

    def train(self):
        for epoch in range(self.cfg.training.n_epochs):
            self.flow.net.train()
            epoch_loss = 0.0

            for batch in self.dataloader:
                loss = self._train_step(batch)
                epoch_loss += loss

            avg_loss = epoch_loss / len(self.dataloader)

            wandb.log(
                {
                    "train/bc_loss": avg_loss,
                    "epoch": epoch,
                },
                step=self.global_step,
            )

            if (
                self.evaluator is not None
                and self.cfg.eval.enabled
                and epoch % self.cfg.eval.eval_every_n_epochs == 0
            ):
                metrics = self.evaluator.evaluate(
                    policy=self._policy_fn(),
                    step=self.global_step,
                )
                print(
                    f"Epoch {epoch} | loss {avg_loss:.4f} | "
                    f"success {metrics['eval/success_rate']:.2%}"
                )

            self._save_checkpoint(epoch)

    def _train_step(self, batch: dict) -> float:
        images = batch["image"]
        actions = batch["action"].to(self.device)
        instructions = batch["instruction"]

        import torchvision.transforms.functional as TF

        pil_imgs = [TF.to_pil_image(img) for img in images]

        with torch.no_grad():
            cond = self.backbone.encode(pil_imgs, instructions)

        self.optimizer.zero_grad()

        loss = self.flow.cfm_loss(actions, cond)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.flow.net.parameters(),
            self.cfg.training.grad_clip,
        )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.flow.update_ema()

        self.global_step += 1

        wandb.log(
            {
                "train/step_loss": loss.item(),
            },
            step=self.global_step,
        )

        return loss.item()

    def _policy_fn(self):
        self.flow.eval()

        def policy(cond: torch.Tensor) -> torch.Tensor:
            return self.flow.sample(
                cond,
                n_steps=self.cfg.ablation.n_euler_steps,
                use_ema=self.cfg.ablation.use_ema,
            )

        return policy

    def _save_checkpoint(self, epoch: int):
        run_id = wandb.run.id if wandb.run is not None else "debug"

        checkpoint_dir = Path(self.cfg.logging.checkpoint_dir) / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f"flow_bc_epoch{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "flow_state_dict": self.flow.net.state_dict(),
                "ema_state_dict": self.flow.ema_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            path,
        )