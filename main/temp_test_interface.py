from main.baseline_interface import BaselineInterface

if __name__ == '__main__':

    ckpt_file = "C:/Users/Sunyyyy/Desktop/Study/PJLAB/Code/ADModel_Pro/output/baseline_kitti_range/50.pt"
    cfg_dir = "./utils/config/samples/sample_baseline"
    model = BaselineInterface(ckpt_file, cfg_dir)

    print(model(2))

