datapath=/cw/dtaijupiter/NoCsBack/dtai/mariette/AdversariApple/Data/mvtec_anomaly_detection
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

gpu=0
sampling_ratio=0.0001

############# Detection
### IM224:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0
# Performance: Instance AUROC: 0.992, Pixelwise AUROC: 0.981, PRO: 0.944
env PYTHONPATH=src python bin/run_patchcore.py --gpu "$gpu" --seed 0 --save_patchcore_model --log_group PINACS_mvtec --log_project MVTecAD_Results results \
patch_core -b resnet18 -le layer2 -le layer3 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p "$sampling_ratio" approx_greedy_coreset dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" mvtec $datapath


datapath=/cw/dtaijupiter/NoCsBack/dtai/mariette/AdversariApple/Data/VisA_20220922
datasets=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
### IM224:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0
# Performance: Instance AUROC: 0.992, Pixelwise AUROC: 0.981, PRO: 0.944
env PYTHONPATH=src python bin/run_patchcore.py --gpu "$gpu" --seed 0 --save_patchcore_model --log_group PINACS_visa --log_project VisA_Results results \
patch_core -b resnet18 -le layer2 -le layer3 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p "$sampling_ratio" approx_greedy_coreset dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" visa $datapath
