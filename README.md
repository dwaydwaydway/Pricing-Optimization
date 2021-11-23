# Pricing-Optimization

## Environment
```bash=
pip install -r requirements.txt
```
or
```bash=
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```
## Usage
### Prepeocess Data
```python=
python src/preprocess.py -c config.yaml
```
### Train a Model for Classification
```python=
python src/train_model.py -c config.yaml
```
### Evaluate Pricing
```python=
python src/pricing.py -c config.yaml
```
or you can put all the congiuration files(all the experiments) in the ```exp2run/``` folder, and run
```python=
bash src/run_exps.sh
``` 
### Everything can be configured in ```config.yaml```

To experiment with new solutions, add a new class in ```preprocessors.py```, ```strategies.py```, ```models.py``` or ```pricer.py```

 - A strategy class should define who the high-level of our algorithm runs
 - A preprocessor class should define how to acquire the embeddings for the cold-starting users
 - A model class should define how the predictions are made
 - A pricer class should define how the to acquire the prices that optimize the revenue

## Experiment
After running the script, a folder with the name of the ```exp_name``` you define in ```config.yaml``` will be automatically created inside ```exp/```. A copy of the ```config.yaml``` you run with and the experiment result will be in it.

# TODO
 - Experiment with different model
 - Start working on part 2
 - Buy milk. I ran out of milk yesterday.