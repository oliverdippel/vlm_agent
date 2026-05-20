import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines.train import run_bc_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_bc_pipeline(cfg)


if __name__ == "__main__":
    main()