import pickle
from main.baseline_interface import BaselineInterface

if __name__ == '__main__':

    ckpt_file = "./output/pointpillar_baseline/30.pt"
    cfg_dir = "./utils/config/samples/sample_baseline"
    model = BaselineInterface(ckpt_file, cfg_dir)

    with open("./data/test_interface_data.pkl", 'rb') as f:
        input_data = pickle.load(f)
    input_list = input_data['input_list']
    HD_map = input_data['HD_map']
    
    corners = model(input_list, HD_map)
    print(corners)

