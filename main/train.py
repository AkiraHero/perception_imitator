import logging
import signal
import sys
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.logger.basic_logger import BasicLogger


def sigint_handler(sig, frm):
    print("You kill the program.")
    try:
        if args.screen_log is not None:
            logger.copy_screen_log(args.screen_log)
        exit(0)
    except Exception as e_:
        print(e_)
        print("fail to copy screen log.")
        exit(-1)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        # manage config
        logging_logger = logging.getLogger()
        logging_logger.setLevel(logging.NOTSET)
        config = Configuration()
        args = config.get_shell_args_train()
        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)
        logger = BasicLogger.get_logger(config)
        logger.log_config(config)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
        model = ModelFactory.get_model(config.model_config)
        trainer = TrainerFactory.get_trainer(config.training_config)

        trainer.set_model(model)
        trainer.set_dataset(dataset)
        trainer.set_logger(logger)
        logging.info("Preparation done! Trainer run!")
        trainer.run()
        if args.screen_log is not None:
            logger.copy_screen_log(args.screen_log)
    except Exception as e:
        print(e)
        if args.screen_log is not None:
            logger.copy_screen_log(args.screen_log)
