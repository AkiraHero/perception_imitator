import logging
import os
import pickle
import traceback
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
import numpy as np

# todo: support 1.distributed testing 2.logger custimized for testing
if __name__ == '__main__':

    try:
        config = Configuration()
        args = config.get_shell_args_train()

        # default test settings
        args.for_train = False
        args.shuffle = False
        args._batch_size = 1

        config.load_config(args.cfg_dir)
        config.overwrite_config_by_shell_args(args)

        # instantiating all modules by non-singleton factory
        dataset = DatasetFactory.get_singleton_dataset(config.dataset_config)
        data_loader = dataset.get_data_loader()

        output_file = "all_boxes_kitti.pkl"
        limit_radius = 6  # 6m
        output_data = {}

        for step, data in enumerate(data_loader):
            logging.error(f"step:{step}/{len(data_loader)}, frm_id:{data['frame_id']}")
            # 0.data preparation
            # trans all data to gpu device

            # align the gt_boxes and target_res_processed
            gt_box = data['gt_boxes'].squeeze(0)
            with open(os.path.join("/home/xlju/Project/ModelSimulator/data/kitti/pvrcnn_fake_model", data['frame_id'][0]+'.pkl'), 'rb') as f:
                dt_res = pickle.load(f)
                assert gt_box.shape[0] == len(dt_res['gt_valid_inx'])
            dt_res['ordered_gt_boxes'] = gt_box
            output_data[data['frame_id'][0]] = dt_res


        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)


    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
