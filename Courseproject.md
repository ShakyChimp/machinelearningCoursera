---
title: "Course Project Machine Learning"
author: "Martijn Bonnema"
date: "24 april 2018"
output: 
  html_document:
    keep_md: TRUE
---
# Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

The goal of this course project is to predict the manner in which they did the exercise.

## Data
For this project we use the weight lifting Exercises Dataset

Training data: [Training data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

We use the training data to build the model

Testing data: [Testing data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

we use the testing data to predict 20 testing cases

# Loading and processing data

Load required packages and set working directory

```r
# set working directory
setwd("~/Your Professionals/Coursera/Data Science R/7. Machine Learning/Course Project")

# Loading the packages used for this project
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(tidyverse)
```

```
## -- Attaching packages ------------------------------------------------------------------------------------------------------ tidyverse 1.2.1 --
```

```
## v tibble  1.4.2     v purrr   0.2.4
## v tidyr   0.8.0     v dplyr   0.7.4
## v readr   1.1.1     v stringr 1.3.0
## v tibble  1.4.2     v forcats 0.3.0
```

```
## -- Conflicts --------------------------------------------------------------------------------------------------------- tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
## x purrr::lift()   masks caret::lift()
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

## Load train en test dataset 

```r
# training data
training <- read.csv("pml-training.csv", header = T, sep = ",", dec = ".", na.strings = c("", "#DIV/0!", "NA"))


# testing data
testing <- read.csv("pml-testing.csv", header = T, sep = ",", dec = ".", na.strings = c("", "#DIV/0!", "NA"))


# We look at the structure of the data
str(training)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

The structure of the data shows a lot of missing data plus some variables aren't used for prediction:
- X
- User_name
- raw_timestamp_part_1
- raw_timestamp_part_2
- cvtd_timestamp
- new_window
- num_window


```r
# remove useless columns in training
training[c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")] <- list(NULL)

# remove useless columns in testing set
testing[c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")] <- list(NULL)
```


## Missing data
We remove columns with more than 50% missing values


```r
# find the column number with more than 50% missing
missing <- c()
for(i in 1:ncol(training)) {
  if(length(which(is.na(training[,i]))) > 0.5*nrow(training)) missing <- append(missing,i)
}

# remove the columns from dataset
training <- training[,-missing]
testing <- testing[,-missing]
```

## Near zero columns
we also want to remove the near zero columns



```r
# find the columns with near zero variance and remove them from the datasets
zero <- nearZeroVar(training, saveMetrics = T)
table(zero$zeroVar) # all false so we don't have to remove any variable
```

```
## 
## FALSE 
##    53
```

## Test training set, data partitioning

Trainig set is 70% of the data and training is 30%


```r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
ttraining <- training[inTrain, ]
ttesting <- training[-inTrain, ]
dim(ttraining)
```

```
## [1] 13737    53
```

```r
dim(ttesting)
```

```
## [1] 5885   53
```

# Model building

## CART

First we start with a simple model with cross validation to see its performance with 5-fold cross validation 


```r
control <- trainControl(method = "cv", 5)

set.seed(1234)

modeltree <- train(classe~., method = "rpart", data = ttraining, trControl = control)
modeltree
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10988, 10990, 10989, 10991, 10990 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03387244  0.5058608  0.35495569
##   0.06109924  0.4185667  0.21182573
##   0.11565456  0.3152122  0.04704745
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03387244.
```
Prediction with the CART model


```r
ttesting$predCART <- predict(modeltree, newdata = ttesting)

# confusion matrix
confusionMatrix(ttesting$predCART, ttesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1510  474  467  396  164
##          B   38  366   40  185  146
##          C  124  299  519  383  290
##          D    0    0    0    0    0
##          E    2    0    0    0  482
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4889         
##                  95% CI : (0.476, 0.5017)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3327         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9020  0.32133  0.50585   0.0000  0.44547
## Specificity            0.6436  0.91382  0.77444   1.0000  0.99958
## Pos Pred Value         0.5015  0.47226  0.32136      NaN  0.99587
## Neg Pred Value         0.9429  0.84873  0.88126   0.8362  0.88891
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2566  0.06219  0.08819   0.0000  0.08190
## Detection Prevalence   0.5116  0.13169  0.27443   0.0000  0.08224
## Balanced Accuracy      0.7728  0.61758  0.64014   0.5000  0.72253
```

As we can see this model has only an accuracy of 58% on the test data. The out-of-sample error is 1-accuracy = 42. This model is not good so we try another one


## Generalized Boosted Regression Models (GBM)
our next model is al boosted model to get extra accuracy


```r
set.seed(1234)
modelgbm <- train(classe~., method = "gbm", data = ttraining, verbose = F, trControl = control)
modelgbm
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10988, 10990, 10989, 10991, 10990 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7519113  0.6854160
##   1                  100      0.8220125  0.7746928
##   1                  150      0.8544068  0.8157328
##   2                   50      0.8543349  0.8154495
##   2                  100      0.9066014  0.8817929
##   2                  150      0.9298963  0.9112837
##   3                   50      0.8958998  0.8682241
##   3                  100      0.9399427  0.9240102
##   3                  150      0.9589431  0.9480568
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
And we predict on the ttesting dataset


```r
ttesting$predGBM <- predict(modelgbm, newdata = ttesting)

# confusion matrix with the accuracy on the ttest data
confusionMatrix(ttesting$predGBM, ttesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641   33    0    0    2
##          B   16 1070   32    1   10
##          C   13   34  983   29   12
##          D    3    1    9  926   12
##          E    1    1    2    8 1046
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9628          
##                  95% CI : (0.9576, 0.9675)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9529          
##  Mcnemar's Test P-Value : 5.89e-07        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9803   0.9394   0.9581   0.9606   0.9667
## Specificity            0.9917   0.9876   0.9819   0.9949   0.9975
## Pos Pred Value         0.9791   0.9477   0.9178   0.9737   0.9887
## Neg Pred Value         0.9922   0.9855   0.9911   0.9923   0.9925
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2788   0.1818   0.1670   0.1573   0.1777
## Detection Prevalence   0.2848   0.1918   0.1820   0.1616   0.1798
## Balanced Accuracy      0.9860   0.9635   0.9700   0.9778   0.9821
```
As we can see this model has aan accuracy of over 96%. The out-of-sample error is 4%. This model is way beter than the CART three model.

## Random Forest Model

The last model I build is the Random Forest Model because it is often used for classification prediction and is seen as one of the best performing algoritm with high accuracy.


```r
set.seed(1234)
modelRF <- train(classe~., data = ttraining, method = "rf", prox = T, trControl = control)
modelRF
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10988, 10990, 10989, 10991, 10990 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9909008  0.9884886
##   27    0.9912644  0.9889488
##   52    0.9826016  0.9779903
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

And predict with the Random Forest Model


```r
ttesting$predRF <- predict(modelRF, newdata = ttesting)

# build a confusion matrix with the accuracy
confusionMatrix(ttesting$predRF, ttesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    0 1134    6    0    3
##          C    0    3 1019    9    1
##          D    0    0    1  955    3
##          E    1    0    0    0 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9929, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9956   0.9932   0.9907   0.9935
## Specificity            0.9995   0.9981   0.9973   0.9992   0.9998
## Pos Pred Value         0.9988   0.9921   0.9874   0.9958   0.9991
## Neg Pred Value         0.9998   0.9989   0.9986   0.9982   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1927   0.1732   0.1623   0.1827
## Detection Prevalence   0.2846   0.1942   0.1754   0.1630   0.1828
## Balanced Accuracy      0.9995   0.9969   0.9953   0.9949   0.9967
```

As we can see this model has aan accuracy of 99.4% on te ttesting dataset. The out-of-sample error is thereby 0.6%. This model performs very good and we use this to predict on the testing set with 20 cases.

# Test cases

We use the Random Forest model to test the 20 cases in the testing dataset and use the results for the quiz answers



```r
testing$predRF <- predict(modelRF, newdata = testing)

Answers <- testing %>% select(problem_id, predRF)
Answers # these are the predictive class for the quiz answers
```

```
##    problem_id predRF
## 1           1      B
## 2           2      A
## 3           3      B
## 4           4      A
## 5           5      A
## 6           6      E
## 7           7      D
## 8           8      B
## 9           9      A
## 10         10      A
## 11         11      B
## 12         12      C
## 13         13      B
## 14         14      A
## 15         15      E
## 16         16      E
## 17         17      A
## 18         18      B
## 19         19      B
## 20         20      B
```




