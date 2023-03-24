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

###Data Cleaning 

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
