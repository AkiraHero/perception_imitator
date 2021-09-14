import logging
import os
import pickle
import traceback
from utils.config.Configuration import Configuration
from factory.model_factory import ModelFactory
from factory.dataset_factory import DatasetFactory
import numpy as np
from analysis.visualization.point_cloud_visualization import PointCloudVisualization
# todo: support 1.distributed testing 2.logger custimized for testing
if __name__ == '__main__':
    p = PointCloudVisualization()

    try:
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

        output_file = "pt_inx.pkl"
        limit_radius = 6  # 6m
        output_data = {}
        for step, data in enumerate(data_loader):
            logging.error(f"step:{step}/{len(data_loader)}, frm_id:{data['frame_id']}")
            # 0.data preparation
            # trans all data to gpu device

            # align the gt_boxes and target_res_processed
            gt_box = data['gt_boxes']
            gt_valid_mask = (gt_box[:, :, -1] > 0)
            gt_valid_elements = gt_valid_mask.sum()
            if not gt_valid_elements > 0:
                raise ZeroDivisionError("wrong gt valid number")

            for batch_inx in range(gt_valid_mask.shape[0]):
                cloud_bin = os.path.join("/home/xlju/Project/ModelSimulator/data/kitti/training/velodyne",
                                         data['frame_id'][batch_inx] + '.bin')
                cloud_ptr = p.load_kitti_bin("/home/akira/000004.bin")
                p.add_point_cloud(cloud_ptr)
                p.show()

                box_indices = gt_valid_mask[batch_inx].nonzero()
                data_xyz = data['points'][:, 1:4]
                gt_point_inx_list = []
                for boxinx in box_indices[0]:
                    box_here = gt_box[0, boxinx]
                    x, y, z = box_here[:3]
                    residual = data_xyz - np.repeat(box_here[:3].reshape(1, -1), data_xyz.shape[0], axis=0)
                    valid_inx = (residual[:, 0] ** 2 + residual[:, 1] ** 2 + residual[:, 2] ** 2 < limit_radius ** 2).nonzero()
                    gt_point_inx_list.append(valid_inx[0])
                output_data[data['frame_id'][batch_inx]] = gt_point_inx_list
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)


    except Exception as e:
        logging.exception(traceback.format_exc())
        exit(-1)
