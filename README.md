# Capstone Project - Will it spin?
# Predictng wind farm electricity output based on day ahead wind forecasts

"Voltcasters": Christine, Ferdinand, Moritz, Jerome

## About
Forecasting the wind energy production will grow in importance as wind energy is one of the fast-growing renewable energy sources in the world.
In this capstone we worked in a group of 4 people. The goal was to predict the wind energy generation 24 h ahead, with an hourly resolution, for 10 wind farms located in Australia, based on wind forecasts. 
We built several machine learning regression models in Python, to find our best model. This included extensive explanatory data analysis  and defining the metric according to which we were going to optimize our prediction models, as well as feature engineering.
We also implemented a dashboard to visualize our results. 


The project has several notebooks: <br>
- The baseline model (see [here](baseline_model.ipynb)) 
- The models we used (see [here](models/)) 
- EDA (see [here](EDA.ipynb)) 

The presentation slides are here : [slides](presentation.pdf)

## Dashboard
heroku link?

## Data
The Data is from the 2014 Global Energy Forecasting Competerion. <br>
~ 18k data rows per wind farm

## Requirements
- pyenv with Python: 3.9.4

## Environment
```
pyenv local 3.9.4
python -m venv .venv
pip install --upgrade pip
pip install -r requirements_dev.txt
source .venv/bin/activate
```


 