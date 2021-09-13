import logging
import traceback
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory


# todo: support 1.distributed testing 2.logger custimized for testing
if __name__ == '__main__':

    try:
        # manage config
        logging_logger = logging.getLogger()
        logging_logger.setLevel(logging.NOTSET)

        config = Configuration()
        args = config.get_shell_args_train()

        # default test settings
        args.for_train = False
        args.shuffle = False

        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
        data_loader = dataset.get_data_loader()

        model = ModelFactory.get_model(config.model_config)
        model.load_model_paras(args.check_point_file)
        model.set_eval()

        for step, data in enumerate(data_loader):
            # 0.data preparation
            # trans all data to gpu device
            data_loader.dataset.load_data_to_gpu(data)

            # get target model output
            target_res = model.target_model(data)
            target_boxes = target_res['dt_lidar_box']

            # align the gt_boxes and target_res_processed
            gt_box = self.pre_process_gt_box(data['gt_boxes'])
            gt_valid_mask = (gt_box[:, :, -1] > 0).to(self.device)
            gt_valid_elements = gt_valid_mask.sum()
            if not gt_valid_elements > 0:
                raise ZeroDivisionError("wrong gt valid number")

            if gt_box.shape != target_boxes.shape:
                raise TypeError("gt_box and target_box must have same shape")



            generator_input = data['points']
            generator_output, point_feature, _, _ = model.generator(generator_input)
            pass




    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
