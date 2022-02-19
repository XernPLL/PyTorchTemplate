import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
import omegaconf


class Writer(SummaryWriter):
    def __init__(self, cfg, logdir):
        self.cfg = cfg
        if cfg.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if cfg.log.use_wandb:
            wandb_init_conf = cfg.log.wandb_init_conf
            wandb.init(config=omegaconf.OmegaConf.to_container(cfg), **wandb_init_conf)
            wandb.run.log_code(".")
            artifact = wandb.Artifact('mnist',"all")
            #artifact.add(table, "my_table")
            artifact.add_file('trainer.py')
            #wandb.log_artifact(artifact)


    def logging_with_step(self, value, step, logging_name):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.cfg.log.use_wandb:
            wandb.log({logging_name: value}, step=step)

