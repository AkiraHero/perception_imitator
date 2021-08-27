from utils.config.Configuration import Configuration
from model.model_factory import ModelFactory
from dataset.dataset_factory import DatasetFactory
from trainer.trainer_factory import TrainerFactory


if __name__ == '__main__':
    # manage config
    config = Configuration()
    args = config.get_shell_args_train()
    config._load_root_config_file(args.cfg_file)
    config.overwrite_config_by_shell_args(args)

    # instantiating all modules by non-singleton factory
    model = ModelFactory.get_model(config.model_config)
    dataset = DatasetFactory.get_data_loader(config.dataset_config)
    trainer = TrainerFactory.get_trainer(config.training_config)

    trainer.set_model(model)
    trainer.set_dataset(dataset)
    trainer.run()
    pass
