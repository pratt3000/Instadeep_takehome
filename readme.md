# How to run:

```export PYTHONPATH="${PYTHONPATH}:/Users/pratt/Documents/Instadeep_takehome/"```

### Docker setup.
```docker build . -t instadeep:latest```

### Open docker bash
CPU only: <br>
```docker run --rm -it --entrypoint bash instadeep:latest```

GPU: <br>
```docker run --rm -it --entrypoint bash --gpus=all instadeep:latest```

### Visualize input data

```python src/visualizations/visualize.py --data_dir data/random_split --save_path reports/data_visualizations --partition "train"```

Many other options are available as well, pl see ```python src/visualizations/visualize.py --help```

### Train model

```python src/train.py --batch_size=256```

Many other options are available as well, pl see ```python src/train.py --help```

### Get prediction for a single test sample

```python src/predict.py --input_seq="ABCDE" --model_checkpoint="lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt"```

Many other options are available as well, pl see ```python src/predict.py --help```

### Evaluate trained model of test set

```python src/evaluate.py --gpu --model_checkpoint="lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt" --test_set_dir="data/random_split/test"```

Many other options are available as well, pl see ```python src/evaluate.py --help```

