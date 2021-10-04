from dataset.box_only_dataset import BoxOnlyDataset
from analysis.model_error_analysis.get_detection3d_error_stats import plot_error_statistics, plot_error_distribution

config = {
    'paras': {
        "for_train": True,
        "batch_size": 100000000,
        "data_root": "../../data/kitti",
        "num_workers": 0
    }
}
dataset = BoxOnlyDataset(config)
data_loader = dataset.get_data_loader()
for step, data in enumerate(data_loader):
    f = plot_error_statistics(data['gt_box'], data['dt_box'])
    f.savefig("kitti_instance_error_pvrcnn.png")
    # get error distribution curve

    plot_error_distribution(data['gt_box'], data['dt_box'])



    break
pass
