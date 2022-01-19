# Capstone Project - Will it spin?
# Predicting wind farm electricity output based on day ahead wind forecasts

Team "Voltcasters": Christine, Ferdinand, Moritz, Jerome

## About
Forecasting the wind energy production will grow in importance as wind energy is one of the fast-growing renewable energy sources in the world.
In this capstone we worked in a group of 4 people. The goal was to predict the wind energy generation 24 h ahead with an hourly resolution, for 10 wind farms located in Australia, based on wind forecasts. 
We built several machine learning models in Python, to find our best model. This included extensive explanatory data analysis, feature engineering as well as finding an appropriate metric. We developed a dashboard to visualize the results of our final model and deployed it on Heroku.


The slides of our presentation: [slides](Capstone_WindPowerPrediciting.pdf)
[video](https://www.youtube.com/watch?v=NEy4wG9iWeU&t=2s)

The project has several notebooks: <br>
- EDA (see [here](notebooks/1_EDA.ipynb)) 
- The baseline model (see [here](notebooks/2_Baseline.ipynb)) 
- The models we used (see [here](notebooks/3_Modelling.ipynb)) 
- The error analysis is done in [here](notebooks/4_Error_Analysis_all_models.ipynb) and [here](notebooks/4_Error_Analysis_Random_Forest.ipynb)

## Dashboard
Link to the dashbaord on Heroku: \
[Energy Output Forecast for the Next 24 Hours](https://windpower-forecast.herokuapp.com)

Dashboard Repository: \
[https://github.com/fklein21/windpower_dashboard](https://github.com/fklein21/windpower_dashboard)

## Data
The Data is from the 2014 Global Energy Forecasting Competition. <br>
~ 18k data rows per wind farm


## Environment
```
pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```


 
