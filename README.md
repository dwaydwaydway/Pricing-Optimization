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
```python=
python src/preprocess.py -c config.yaml
python src/main.py -c config.yaml
```
## Code
### Everything can be configured in ```config.yaml```

To experiment with new solutions, add a new class in ```preprocessors.py```, ```strategies.py```, ```models.py``` or ```pricer.py```

 - A strategy class should define who the high-level of our algorithm runs
 - A preprocessor class should define how to acquire the embeddings for the cold-starting users
 - A model class should define how the predictions are made
 - A pricer class should define how the to acquire the prices that optimize the revenue

After you define your strategy/preprocessor/model, change the name of those modules in ```config.yaml``` and run with that configuration file
```python=
python src/preprocess.py -c config.yaml
python src/main.py -c config.yaml
```

## Experiment
After running the script, a folder with the name of the 'exp_name' you define in ```config.yaml``` will be automatically created inside ```exp/```. A copy of the ```config.yaml``` you run with and the experiment result will be in it.

## TODO
 - Figure out what the hell am I doing here (X)
 - Write the scripts that generated our final predictions
