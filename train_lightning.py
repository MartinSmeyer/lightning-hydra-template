from typing import List, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="config.yaml")
def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils
    from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
    from pytorch_lightning import LightningDataModule, LightningModule
    from pytorch_lightning import Callback, Trainer, seed_everything
    ###########################################################################
    
    # Pretty print config using Rich library and get logger
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    log = utils.get_logger(__name__)
    
    
    if 'trial' in config:
        config.callbacks.pytorch_lightning_pruning_callback.trial = config.trial

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = instantiate(config.model)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = instantiate(config.trainer, 
                                   callbacks=callbacks, 
                                   logger=logger, 
                                   _convert_="partial")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(config=config,
                              model=model,
                              trainer=trainer)

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Return metric score for hyperparameter optimization
    optimized_metric = trainer.callback_metrics[config.get("optimized_metric")]
    
    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test()

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            import wandb
            wandb.finish()
    
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")

    return optimized_metric

if __name__ == "__main__":
    train()
