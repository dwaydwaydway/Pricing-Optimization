# Pricing-Optimization
## Usage
```python=
python src/main.py -c config.yaml
```
## Code
# Everything can be configured in config.yaml

To experiment with new solutions, add a new class in strategies.py, models.py or preprocessors.py

A strategy class should define who the high-level of our algorithm runs
A preprocessor class should defines how to acquire the embeddings for the cold-starting users
A model class should how the predictions are made

After you define your strategy/preprocessor/model, change the name of those modules in config.yaml and run with that configuration file
```python=
python src/main.py -c config.yaml
```
