### Usage

Sample command ro tun SMICLR
```
$ python3 main.py --epochs=101 --lr=1e-3 --no-lr-decay --temperature=.1 --batch=256 --percent=0.4 --dataset-path='data' --output='smiles_noise_node_drop' --node-drop --smiles-noise
```
Sample command to run supervised training
```
$ python3 main.py --target=0 --epochs=301 --sup --batch=32 --dataset-path='data' --load-weights='smiles_noise_node_drop' --output='sup_smiles_noise_node_drop'
```
