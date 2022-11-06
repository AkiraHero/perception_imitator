# Perception Imitation: Towards Data-free Simulator for Autonomous Vehicles



## Getting Start

```python
conda create --name imitator python=3.9
conda activate imitator
pip install -r requirements.txt
```

## Data Preparation

1. Download the source data of [nuScenes](https://www.nuscenes.org/nuscenes#download) and [CARLA](https://mycuhk-my.sharepoint.com/personal/1155167065_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155167065_link_cuhk_edu_hk%2FDocuments%2FPerceptionImitation%2FFilesSummary%2FImitatorData%2FCARLA.zip&parent=%2Fpersonal%2F1155167065_link_cuhk_edu_hk%2FDocuments%2FPerceptionImitation%2FFilesSummary%2FImitatorData), and extract them into `~/Datasets/`.

2. Download the target models' results ([ImitatorData](https://mycuhk-my.sharepoint.com/personal/1155167065_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155167065_link_cuhk_edu_hk%2FDocuments%2FPerceptionImitation%2FFilesSummary%2FImitatorData)）and put them in `~/Perception_Imitation/data/`. The relationship of different target model trained with different dataset and its results are illustrated in the table:

   | Dataset      | Results                                         |
   | ------------ | ----------------------------------------------- |
   | **nuScenes** | **nuscenes_** *%TargetModel* **_match_gt.json** |
   | **Kitti**    | **kitti_***%TargetModel* **_match_gt.pkl**      |
   | **CARLA**    | **carla_** *%TargetModel* **_match_gt.pkl**     |

   where TargetModel = {‘pp’, ‘cp’, ‘pvrcnn’}.

3. The data is structured as:

   ~~~
   |— Datasets 
   |  |— carla 
   |  |  |— carla_new 
   |  |  |— Maps 
   |  |— nuScenes 
   |  |  |— maps 
   |  |  |— v1.0-mini 
   |  |  |— v1.0-trainval 
   |  |  |— ... 
   |— Perception_Imitation 
   |  |— data 
   |  |  |— nuscenes_pp_match_gt.json 
   |  |  |— kitti_cp_match_gt.pkl 
   |  |  |— ... 
   |  |— dataset 
   |  |  |— ... 
   |  |— ... 
   ~~~

## Training

### Baseline

1. Running code :

   ~~~python
   python main/train.py --cfg_dir utils/config/samples/sample_carla 
   ~~~

2. Change datasets

   Replace "sample_carla" with "sample_nuscenes" or "sample_kitti".

3. Change target model

   Modify different value of the key "target_model" in file `sample_carla/dataset/scene_occ_xxx.yaml `.

### Our imitator

1. Running code:

   ~~~
   python main/train.py --cfg_dir utils/config/samples/sample_carla_improve 
   ~~~

2. The rest are the same as the baseline.

## Evaluation

### Gaussian and Multimodal

Running code:

~~~
python main/test_baseline_gaussian.py --cfg_dir utils/config/samples/sample_carla_improve/ 
~~~

### Baseline or Imitator

Running code:

~~~
python main/test_baseline_evaluate.py --cfg_dir utils/config/samples/sample_carla_improve/ 
~~~

### Visualization

Running code:

~~~
python main/plot_qualitative_results.py 
~~~

and the images will be saved in the format of eps in `~/ADModel_Pro/ output/pic/`.

