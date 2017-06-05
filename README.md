tensorboard --logdir=/tmp/tensorflow/ali

- Train a model
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=loam
```

- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```
