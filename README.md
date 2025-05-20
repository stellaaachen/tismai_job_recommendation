# tismai_job_recommendation

This is my final project for the Software: Making an Impact class. This repo has 2 components:

1. The EDA and modeling that explores the dataset and tries to predict the experience level of a job posting based on descriptions. 

2. The salary predictor tool that has a front end to give a salary median given factors such as titles, job descriptions, etc.

To run the first component, simply hit run all of the ```eda.ipynb``` notebook and it should all be self contained. 

To run the second component, you can either:
- if you don't want retraining of the model: ```streamlit run salary_prediction_dashboard.py```
- if you want to retrain the model: download the data from this folder: https://drive.google.com/drive/folders/1sdbUJJ-GiofheswP9WmqlkXxsTFO2jgH?usp=sharing and then place all of the data in the data folder. Then first run ```python3 salary_prediction_model``` to generate the pipeline. Then run ```streamlit run salary_prediction_dashboard.py```

Either way you should see a streamlit webpage pop up. 

Video that runs through the code: https://drive.google.com/file/d/1P8PdSDMpNc8o6qcMQYCPVoRHlMVk8ErS/view?usp=sharing


