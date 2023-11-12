# How to run:

## Using Docker

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
(Note: batch_size needs to be much smaller on CPU (bs=1). To use GPU use the --gpu flag.) <br>
```python src/train.py --batch_size=256```

Many other options are available as well, pl see ```python src/train.py --help```

### Get prediction for a single test sample

```python src/predict.py --input_seq="ABCDE" --model_checkpoint="lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt"```

Many other options are available as well, pl see ```python src/predict.py --help```

### Evaluate trained model of test set

```python src/evaluate.py --gpu --model_checkpoint="lightning_logs/version_10/checkpoints/epoch=2-step=12738.ckpt" --test_set_dir="data/random_split/test"```

Many other options are available as well, pl see ```python src/evaluate.py --help```

## Without docker
1. Install requirements: ```pip install -r requirements.txt```
2. Export python path:
```export PYTHONPATH="${PYTHONPATH}:/Users/pratt/Documents/Instadeep_takehome/"```
3. Run any of the above commands. (Tested on Python version 3.10)

## TODO:
1. Add tests.
2. Add more visualizations for training loss graphs.
3. Add logger.
4. Add more comments.
5. Train new model.
6. Create approach explaination pdf.