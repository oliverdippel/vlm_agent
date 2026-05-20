import wandb
from omegaconf import DictConfig, OmegaConf

from src.data.ingestion.libero_dataset import get_dataloader
from src.models.backbone import FrozenVLMBackbone
from src.models.mock_backbone import MockVLMBackbone
from src.models.flow_head import FlowMatchingHead
from src.models.train import BCTrainer
from src.utils import device
from src.utils.device import get_device
from tests.unit.test_flow_head import cfg


def run_bc_pipeline(cfg: DictConfig) -> None:
    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=(
            f"{cfg.wandb.run_name}_"
            f"{cfg.ablation.head_type}_"
            f"nfe{cfg.ablation.n_euler_steps}_"
            f"ema{cfg.ablation.use_ema}"    
        ),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    device = get_device()
    print(f"Using device: {device}")

    # Components
    if cfg.backbone.name == "mock":
        backbone = MockVLMBackbone(
            conditioning_dim=cfg.flow.conditioning_dim,
            device=device,
        )
    elif cfg.backbone.name == "paligemma":
        backbone = FrozenVLMBackbone(
            model_id=cfg.backbone.model_id,
            cache_dir=cfg.backbone.cache_dir,
        )
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone.name}")

    flow_head = FlowMatchingHead(cfg.flow)

    dataloader = get_dataloader(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.shuffle,
    )

    if cfg.eval.enabled:
        from libero.libero.benchmark.libero_suite_task_map import libero_task_map
        from src.envs.libero_wrapper import LIBEROWrapper
        from src.pipelines.eval import Evaluator

        task_suite_name = cfg.env.task_suite_name

        if cfg.env.task_name is None:
            task_name = libero_task_map[task_suite_name][cfg.env.task_id]
        else:
            task_name = cfg.env.task_name

        env = LIBEROWrapper(
            task_suite_name=task_suite_name,
            task_id=cfg.env.task_id,
            task_name=task_name,
            img_size=cfg.env.img_size,
        )

        evaluator = Evaluator(
            env=env,
            backbone=backbone,
            n_eval_episodes=cfg.eval.n_eval_episodes,
        )
    else:
        evaluator = None

    trainer = BCTrainer(
        flow_head=flow_head,
        backbone=backbone,
        dataloader=dataloader,
        evaluator=evaluator,
        cfg=cfg,
        device=device,
    )

    trainer.train()

    run.finish()