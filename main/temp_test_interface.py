import pickle
from main.baseline_interface import BaselineInterface

if __name__ == '__main__':

    ckpt_file = "./output/pointpillar_baseline/98.pt"
    cfg_dir = "./utils/config/samples/sample_baseline"
    model = BaselineInterface(ckpt_file, cfg_dir)

    with open("./data/test_interface_data.pkl", 'rb') as f:
        input_data = pickle.load(f)
    input_list = input_data['input_list']
    HD_map = input_data['HD_map']

    print(len(input_list), len(input_list[0]))
    print(HD_map.shape)
    
    corners = model(input_list, HD_map)
    print(corners)

