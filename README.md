# Capstone Project - Will it spin?
# Predicting wind farm electricity output based on day ahead wind forecasts

Team "Voltcasters": Christine, Ferdinand, Moritz, Jerome

## About
Forecasting the wind energy production will grow in importance as wind energy is one of the fast-growing renewable energy sources in the world.
In this capstone we worked in a group of 4 people. The goal was to predict the wind energy generation 24 h ahead with an hourly resolution, for 10 wind farms located in Australia, based on wind forecasts. 
We built several machine learning models in Python, to find our best model. This included extensive explanatory data analysis, feature engineering as well as finding an appropriate metric. We developed a dashboard to visualize the results of our final model and deployed it on Heroku.


The slides of our presentation: [slides](presentation.pdf)
[video](https://www.youtube.com/watch?v=NEy4wG9iWeU&t=2s)

## Contents: <br>
 1. In the first notebook we defined the dataset and added more features [here](notebooks/1_Dataset.ipynb)
 2. Then second part was data cleaning and EDA 
    - Data cleaning and histograms [part1](notebooks/2_1_EDA_Data_cleaning.ipynb), 
    - Inspecting the Zero Values of the target [part2](notebooks/2_2_EDA_Zero_Values.ipynb),
    - Analysing feature behavior over time [part3](notebooks/2_3_Time_Analysis_EDA.ipynb)
    - Analysing effects of windspeed and winddirection on the target[part4](notebooks/2_4_EDA_Wind.ipynb)
 3. Feature engineering [here](notebooks/2_5_Feature_engineering.ipynb)
 4. The baseline model [here](notebooks/3_Baseline.ipynb)
 5. The models we used [here](notebooks/4_Modeling.ipynb)
 6. With our best model we made the predictions of the target [here](notebooks/5_Target.ipynb)
 7. Error analysis [part1](notebooks/6_Error_Analysis_all_models.ipynb), 
 [part2](notebooks/6_Error_Analysis_Random_Forest.ipynb)

Python script for model training, making predictions, writing scores and best feature combinations to .csv files, logging to MLflow : [script](modeling/functions.py) 

## Dashboard
Link to the dashbaord on Heroku: \
[Energy Output Forecast for the Next 24 Hours](https://windpower-forecast.herokuapp.com)

Dashboard Repository: \
[https://github.com/christine-berlin/windpower_dashboard](https://github.com/christine-berlin/windpower_dashboard)

## Data
The Data is from the 2014 Global Energy Forecasting Competition, and consists of
weather forecasts, given as u and v components (zonal wind, flowing in west-east direction, and meridional wind, flowing in north-south direction), 
which we transformed to wind speed and direction.
The forecasts were given at two heights, 10m amd 100m above grond level.
~ 18k data rows per wind farm.


## Environment
```
make setup
```
#or


```
pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```


 
