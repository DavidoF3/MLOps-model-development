# Model evaluation

## Contents
* Partitioning data
* Choosing an evaluation metric & error analysis
* Model selection and versioning for deployment

## Partitioning data

Usually we partition data into train, validation, test. 

Issue with data partitioning. Traps:
* Make sure that partitions are drawn from same distribution. So that there is no fundamental difference between train/valid.test sets.
* Validation and test sets should look like the problem in production. 
* There should be no data `leakage` between data partitions.

### Examples of real world data `leakage`
* Data `leakage` can happen when a feature in the model development dataset is available, but it would not be available at the time of using the model - leading to an optimistic model performance. Hence, important to understand the business problem to prevent `leakage`. 
* AirBnB: first booking date on customer table, aimed at listing customers signup date to AirBnB. Here we are leaking data from the future.
* Github: when training an LLM, encountered nearly duplicate code across train/ test sets from different repos (cause people copy each others code).
* Hospital: predicting patient readmission, but dataset containing whether patient currently dead or not.
  
### Data partitioning schemes
1. Group partitionsing
   * In many cases images come from the same camera and close in time. We don't want two similar images (from same camera and close in time both in train and validation sets). This type of data `leakage` would yield a more optimistic view of model performance, since model would have already seen that data during training. Remedy: data that is not independent should not cross a partition boundary.
2. Stratified partitioning
   * Useful when you have rare labels (eg. only 5% of images in the AV dataset have bicycles). Instead of randomly sampling your data and placing a random sample into the train/valid sets, you want to make sure that you fairly represent all the classes in your sets. 
3. Stratified + Group partitioning
   * Combination of two schemes above.
   * Pictures from same camera connot cross partitions + each partition has same % of classes.
   * Tools like sklearn have methods to do such partitioning ([see notebook](../1_build_end2end_prototype/02_PrepareDataForTrain.ipynb)).
4. Time Series partitionsing
   * Cannot split data randombly in time series data. Should do `time series cross validation`. When doing this, we need to think how the model will be used in practice. For example, how many steps ahead do are we trying to predict? Hence for each snapshot we can take different lengths in time as `training` data, and use the data `x time ahead` as `validation`.
5. Time + Group partitioning
   * Same as scheckme 4 (time Series partitionsing) i.e. respecting time gap between train and validation data in each data stream; but also make sure that data from same source is not leaked between different sets (train/valid/test).
6. K-fold cross validation
   * Useful when we do not have much data, and we want multiple estimates of the model performance because of the small validation dataset size.
7. Random tain/valid/test splits
   * Common on datasets where there is no time element and no info leackage (dependencies) across examples. 

### Best practices - data partitioning with W&B 
1. Record the partitioning of the data within a field in the data ([see notebook](../1_build_end2end_prototype/02_PrepareDataForTrain.ipynb)). 
   * For example: in a dataframe, it's useful to have a column indicating what partition each datapoint (eg. image and mask) belongs to. 
   * Useful for filtering data and creating reports (keep a record on how was data partitioned) + for debugging models.

2. Don't check model performance on test set until after you have selected the model.


## Choosing an evaluation metric

### Tips:
* Pick a single-number evaluation metric. Precision and Recall are two numbers. Use F1 score instead.
* Try to focus your attention on `one metric` when optimising a model (don't combine many different metrics). If you need to consider other metrics, just determine a minimum threshold for these othere metrics. For example, optimise model accuracy and just set a minimum threshold for inference speed (latency).

### Case study example:
* In this repository (using AV dataset) we log the model IoU's for person, bicycle, vehicle, road, traffic light, traffic sign and background. These are a lot af metrics. Based on the above tips, we don't want to use all these metrics for model selection (would be confusing) - `choose one`. If stakeholders are really interested in person accuracy, we would optimise for person IoU and set minimum thresholds for all the other IoU metrics. 

### Applying tips on W&B:
* Apply filters to all the sweep runs (eg. background_iou >= 0.8, traffic_light_iou >= 0.1, etc.)




With metric that are ratios (eg. IoU) need to:
* Be aware of imbalances in terms of how to aggregate these metrics.