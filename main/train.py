from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.logger.basic_logger import BasicLogger
import torch
torch.multiprocessing.set_start_method('spawn')


if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    config.load_config(args.cfg_dir)
    config.overwrite_config_by_shell_args(args)
    logger = BasicLogger.get_logger(config)

    # instantiating all modules by non-singleton factory
    dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
    model = ModelFactory.get_model(config.model_config)
    trainer = TrainerFactory.get_trainer(config.training_config)

    trainer.set_model(model)
    trainer.set_dataset(dataset)
    trainer.run()
    pass
