# Power-Outage-Classification-Model

## Introduction

Finding the cause of an outage may save a lot of time for repair workers. If they’re aware of the cause (i.e. severe weather, intentional attack, or equipment failure), workers may have a better idea of which equipments were more likely damaged during the event.

For example, if severe winds were the cause of the outage, workers may focus their attention on large power lines or towers that most likely knocked down during high winds. Knowing this information 

Therefore, our goal is to predict the cause of outages using a multi-class classification model. 

We will use f1-score to evaluate the performance of our classification model. We chose f1-score over accuracy because our dataset is unbalanced (i.e. 49.74% of outages were due to severe weather). For our f1-score, we will take the weighted average because we want to assign greater contribution to classes with more examples in the dataset. 

We will being using a dataset acquired from DOE׳s Office of Electricity Delivery and Energy Reliability and U.S. Energy Information Administration.

Rows and Columns

There are 1534 rows and 6 columns in the dataset that are relevant to our classification model.
1. The column ‘CAUSE.CATEGORY’ is our response variable (i.e. the variable we are predicting).
2. The column 'CLIMATE.REGION' contains the region of where outages occurred. Some regions may be more susceptible to certain cause of outages. 
3. The column ‘ANOMALY.LEVEL’ contains the (ONI) index referring to the cold and warm episodes by season. Low anomaly levels may lead to more severe weather, causing power outages. 
4. The column 'OUTAGE.DURATION’ marks duration of the outage in minutes. Outages caused by fuel supply emergency may last longer on average than other causes. 
5. The column 'CUSTOMERS.AFFECTED’ counts the number of customers affected by the outage. Outages caused by severe weather (i.e. hurricanes or tornadoes) may affect more customers than outages caused by slanting which usually affects smaller groups of customers. 
6. The column 'CLIMATE.CATEGORY' contains represents the climate episodes based on a threshold of ± 0.5 °C for the Oceanic Niño Index (ONI). Severe weathers such as hurricanes are more likely to happen during higher temperatures with warm gusts of winds. 

### Data Cleaning 

To ensure that the insights and conclusions drawn from the data are accurate and reliable, we cleaned our dataset in the following manner:
**1. Excel to CSV Format**
We converted the Excel file to CSV format. Since CSV only accepts data points separated by commas, we deleted the title and description in the Excel file so that the data is readable into pandas DataFrame.

**2. Filter Out Unnecessary Columns**
Out of the 55 columns, we kept 6 and created 1 new column. The reason for dropping over a dozen columns were because they were unnecessary fitting our multiclass classification model. For example, we did not require the percentage of inland water area in the state. While it may be important data points, it was unnecessary for the scope of this project.

**3. Fill NaN values in OUTAGE.DURATION**
From previous data exploration, we discovered that the missingness of OUTAGE.DURATION depends on ‘CAUSE.CATEGORY’ with a statistically significant p-value of 0.002.  Therefore, we imputed the mean OUTAGE.DURATION conditioned on CAUSE.CATEGORY.

**4. Handle NaN values in ANOMALY.LEVEL**
Out of 1534 values, the ANOMALY.LEVEL columns contained 9 NaN values. Through data exploration, we discovered the missingness did neither depended on the region or the cause of outage. Therefore, due to its low significance, we dropped the 9 values from the DataFrame.

**5. Fill NaN values in 'CUSTOMERS.AFFECTED'**
From previous data exploration, we discovered that the missingness of 'CUSTOMERS.AFFECTED' depends on ‘CAUSE.CATEGORY’ with a statistically significant p-value of 0.0.  Therefore, we imputed the mean 'CUSTOMERS.AFFECTED' conditioned on CAUSE.CATEGORY.

## Baseline Model

### Columns Used

Our baseline model predicts cause of the power outage, `CAUSE.CATEGORY` using basic quantitative data and a qualitative data.\
\
**The columns we used to extract data are the following:**\
\
`ANOMALY.LEVEL`: Contains quantitative continuous data. Represents the oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season.\
\
`OUTAGE.DURATION`: Contains quantitative discrete data. Represents the duration of outage events in minutes.\
\
`CLIMATE.REGION`: Contains qualitative nominal data. Represents U.S. Climate regions as specified by National Centers for Environmental Information.\
To use this data in our model, we changed it to numerical data using One Hot Encoding.

### Baseline Model
We first input the collected data into Column Transformer use it in Pipeline. We kept all the numerical columns (`ANOMALY.LEVEL` and `OUTAGE.DURATION`) as they are, and modified the qualitative column `CLIMATE.REGION` using One Hot Encoder.\
After creating Column Transformer with all the data, we initiated Pipeline along with Decision Tree Classifier.\
**Prediction**\
We used train test split method to prove the accuracy of the Classifier.\
Using pl.score(X_test, y_test) method, we calcultated the accuracy of the Classifier.\
As a result, we got 0.638743.\
Our current model does extremely bad job at predicting the cause of power outages because our calculated accuracy shows that the Classifier is very inconsistent with prediction.


**Gridsearch to find best hyperparameter**\
To improve our model, we decided to do Gridsearch to find the best hyperparameter. Using GridSearchCV with hyperparameters for Decision Tree Classifier (max_depth, min_samples_splot, and criterion), we found out that the Classifier works the best when criterion as gini, max_depth as 10, and min_sampls_split as 100. Inputting those values to our Pipeline, we got 0.670157.\
This increased accuracy by about 0.04, which is a great improvement thinking that in term of probability. However we still considered our model inaccurate because we were aiming over 0.8 accuracy.

## Final Model

We trained a decision tree classifier to predict CAUSE.CATEGORY by using two featured engineered columns and three original columns.

**Feature Engineered:**

 **1. Devastating amount**

Data type: Quantitative discrete data type

We engineered a new feature that multiplies `CUSTOMERS.AFFECTED` with `OUTAGE.DURATION`. 
Multiplying the two features has the synonymous affect of taking the **area** of how devastating an outage was. For example, an outage that has a long duration and affects a large group of people will have a larger area than an outage that last a short duration and affects small group of people. 

 **2. Mean customers affected by outages in certain seasonal climates in specific regions**

Data type: Quantitative continous data type 

We engineered a new feature that takes the average customers affected during a certain seasonal period in a specific region. For example, the region West North Central has the least amount of customers affected by outages during a warm climate period. This makes sense, as power grids located in the West North Central are less likely to experience severe weather in warm climate periods. We performed this transformation by taking the mean of `CUSTOMERS.AFFECTED` after being grouped by `['CLIMATE.CATEGORY','CLIMATE.REGION']`.

**Remaining Features:**

3. `ANOMALY.LEVEL`: Quantitative continuous data
3. `OUTAGE.DURATION`: Quantitative discrete data
5. `CUSTOMERS.AFFECTED`: Quantitative discrete data
6. `CLIMATE.REGION`: Qualitative nominal data

Since`CLIMATE.REGION`is qualitative, we must use one hot encoding to transform the categorical feature into several binary features.

`# Preprocess`
`preprocess_data = ColumnTransformer(`
`transformers= [`
        `('cat_cols', OneHotEncoder(), ['CLIMATE.REGION']),`
        `('keep_outage_duration',FunctionTransformer(),['CUSTOMERS.AFFECTED', 'OUTAGE.DURATION']),`
        `('affected_region', mean_pipe, ['CLIMATE.CATEGORY','CLIMATE.REGION']),`
        `('devastating_degree', prod_pipe, ['CUSTOMERS.AFFECTED', 'OUTAGE.DURATION']),`
    `],`
    `remainder = 'passthrough'`
`)`
`# Create pipeline`
`pl = Pipeline([`
    `('preprocessor', preprocess_data),`
   ` ('dt', DecisionTreeClassifier(max_depth=3))`
`])`
`# Fit data on Decision Tree Classifier`
`pl.fit(X_train, y_train)`
`# Accuracy of Classifer`
`pl.score(X_test, y_test)`


## Fairness Analysis

We were curious to know if our model was fair in predicting the cause of outages for Western regions and Non-Western regions. 

**C**: Decision Tree Classifier (1 if predicts severe weather as cause of outage , 0 if predicts otherwise)\
**Y**: Whether or not cause of outage was truly because of severe weather (1) or due to other reason (0)\
**A**: Whether or not the outage occurred in western region (1) or non-western region (0)\

**Null Hypothesis**: The classifier’s accuracy is the same for both western regions and non-western regions, and any differences are due to random chance.
**Alternative Hypothesis**: The classifier’s accuracy is higher for western-regions. 

**Relevant Columns**\
The three main columns necessary to perform out Permutation Test is newly generated columns, `is_west` (True if the event happened in Western state, false otherwise), `is_severe` (Whether the causation of the outage is severe weather), and `prediction`, which is predicted `is_west` based on all the other columns we used to create our final model and the newly created columns. Since we are performing the test under the null hypothesis, we must generate our test statistic from shuffling the `is_west`.

**Test Statistics**\
After shuffling, we must find if Western and non-Western states have the same accuracy. 
We decided to use **Difference in accuracy** on Western states and non-Western states. 

Repeatedly computing the difference in accuracy will generate an empirical distribution of the difference under the null hypothesis. 

**Empirical Distribution of Difference in Accuracy**\
Red Line = Observed TVD

<iframe src="assets/difference_in_accuracy_fairness.html" width=700 height=500 frameBorder=0></iframe>

**p-value: 0.0514**

**Conclusion**\
The difference in accuracy across the two groups in not significant because the p-value is above the significance level of 0.05. **This means we fail to reject the null hypothesis, and C likely achieves accuracy parity.** Therefore, the classifier C is likely to be fair as it performs the same for outages that occurred in western regions and non-western regions 

# MODIFIED OUTAGE DF

|   ANOMALY.LEVEL |   OUTAGE.DURATION | CLIMATE.REGION     |   CUSTOMERS.AFFECTED | CAUSE.CATEGORY     | CLIMATE.CATEGORY   | POSTAL.CODE   |
|----------------:|------------------:|:-------------------|---------------------:|:-------------------|:-------------------|:--------------|
|            -0.3 |              3060 | East North Central |                70000 | severe weather     | normal             | MN            |
|            -0.1 |                 1 | East North Central |                 1790 | intentional attack | normal             | MN            |
|            -1.5 |              3000 | East North Central |                70000 | severe weather     | cold               | MN            |
|            -0.1 |              2550 | East North Central |                68200 | severe weather     | normal             | MN            |
|             1.2 |              1740 | East North Central |               250000 | severe weather     | warm               | MN            |

# CLIMATE CATEGORY & REGION CUSTOMERS AFFECTED


| CLIMATE.CATEGORY   | CLIMATE.REGION     |   CUSTOMERS.AFFECTED |
|--------------------|--------------------|----------------------|
| cold               | Central            |              98794.7 |
| cold               | East North Central |              98521.8 |
| cold               | HI                 |             294000   |
| cold               | Northeast          |             107051   |
| cold               | Northwest          |              24056.5 |
| cold               | South              |             128787   |
| cold               | Southeast          |             138481   |
| cold               | Southwest          |              53630.7 |
| cold               | West               |             156325   |
| cold               | West North Central |              55577.6 |
| normal             | Central            |             137665   |
| normal             | East North Central |             135104   |
| normal             | HI                 |              45650   |
| normal             | Northeast          |              97431.7 |
| normal             | Northwest          |              25508.8 |
| normal             | South              |             170177   |
| normal             | Southeast          |             153197   |
| normal             | Southwest          |              18916.5 |
| normal             | West               |             135404   |
| normal             | West North Central |              20328.4 |
| warm               | Central            |              98845.8 |
| warm               | East North Central |             101711   |
| warm               | HI                 |             175443   |
| warm               | Northeast          |             107387   |
| warm               | Northwest          |              95453   |
| warm               | South              |              94509.3 |
| warm               | Southeast          |             266902   |
| warm               | Southwest          |              40143.1 |
| warm               | West               |             186607   |
| warm               | West North Central |              15709.5 |


# Baseline confusion Matrix
<iframe src="assets/baseline_confusion_matrix.html" width=700 height=500 frameBorder=0></iframe>

# Causality_customers
<iframe src="assets/causality_customers_affected.html" width=700 height=500 frameBorder=0></iframe>

# Customers_causality
<iframe src="assets/customers_affected_causality.html" width=700 height=500 frameBorder=0></iframe>

# histogram of anomaly level
<iframe src="assets/histogram_anomaly.html" width=700 height=500 frameBorder=0></iframe>

# optimal_hyperparameter_final_model
<iframe src="assets/optimal_hyperparameter.html" width=700 height=500 frameBorder=0></iframe>
