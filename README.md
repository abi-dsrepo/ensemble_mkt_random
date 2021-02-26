
## Solution
Similar to any datascience approach, I have performed some EDA, ETL and Model training.
### EDA

1. We check for null values in dataframe
2. Only customer rank has null values
3. Most of the time when customer rank is null the customer is returning
4. We also see that though the data is only till Feb 2017,the number of customers returning is maximun for the year 2017
5. More than 50 percent of customers in 2017 are returning
You can check the pandas profiling reports that are stored at `src/output` directory executed from `notebooks/eda.upynb`.
    - raw_data_report.html
    - null_data_report.html
    - duplicate_datareport.html
    - etl_data_report.html (larger in size, please run the code and obtain the same)
### Data cleansing and Feature Engineering
1. The orders data is summarized and we create many variables that describe the mean,min,max,median and sum of the variables
2. We create order)date_amin and order_date_amax for every customer(min and max order date)
3. We bucket the order hours and create 3 variables that describe if the orders are in day,evening or night
4. We create other variables that calulates the total of nonzerovoucher,voucher_amount,is_failed,amount_paid,delivery_fee,hour_class_day,hour_class_evening,hour_class_night
5. We careate a recenency score variable for every customer.If the customer last order date is purchased more recently then more chances he will return again


### Model
1. I have implemented Randomforest model as classifier
2. In order to overcome data imbalance I have done undersapmling of the majority class for different k cross validations and built the randomforest on different holdout datasets
3. The final model is a combination of different random forest models
4. By this way I am able to predict the number of returning customers more 

### Running the model
1. Create a python virtualenv and execute `pip install -r requirements.txt`
2. Activate the env and run, `python src/main.py`
    - NOTE: If you experience any module not found error, add the root directory to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:${pwd}`
    ```
    > python src/main.py
    2021-02-25 17:42:02,326 - [INFO] - [Delivery_Hero_ETL] : Reading csv data..
    2021-02-25 17:42:03,891 - [INFO] - [Delivery_Hero_ETL] : Adding time related additional columns
    ```

3. You should be seeing the predicted results in the `output` folder. Also, you can track the progress in the mlflow GUI.
    - NOTE: 
        I have tested the model against the whole dataset by removing duplicates and saved the result in `output/all_pred_wholedf.csv` file.  Below, is a simple crosstab of the same.

        ![Predictions](pred.png)
### Testing
I have created some unit testing which is about ~45% coverage. But, this can be improved by adding more unit tests

### Logging
Finally, added logging to stdout for better debugging purposes.

### TODO
With the emerging use of MLflow, the models could be retrieved from disk and can be ensembled. I might need to make use of pyfunc.model wrapper exposing the estimators (right now, the default method available is predict)
