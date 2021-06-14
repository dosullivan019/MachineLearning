Classification of Mushrooms
================

This notebook will explore what the best machine learning technique is
to classify a mushroom as edible or poisonous.

**Data Source:** <https://www.kaggle.com/uciml/mushroom-classification>

Techniques used here are:

  - Decision Trees
  - Random Forest
  - Logistic Regression

<!-- end list -->

``` r
library(ggplot2)        # for plotting
library(dplyr)
library(tidyr)
library(rpart)          # decision tree
library(randomForest)   # random forest
library(pROC)
```

``` r
mushrooms = read.csv('data/mushrooms.csv', stringsAsFactors = TRUE)
```

### Data Exploration

The dataset contains 8124 records and 23 variables. The outcome which we
want to predict is class (edible or poison). The other 22 variables will
be used as predictor variables. These relate to the characteristics of
the mushroom, it’s population and habitat type. When predicting a binary
variable it’s important to ensure the outcome variable is balanced and
that one class is not dominant in the dataset.

``` r
ggplot(data = mushrooms, aes(x=class)) +
  geom_histogram(stat='count') +
  labs(title = 'Distribution of outcome variable: Edible vs Poisonous') +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5))
```

![](classification_mushrooms_edible_poisonous_files/figure-gfm/plot%20outcome%20variable-1.png)<!-- -->

The outcome variable looks goods with an almost 50:50 split between
edible and poisonous mushrooms. Now we will explore the predictor
variables and look for any missing data.

``` r
# checking for NAs within the dataset
sapply(mushrooms, function(x) sum(is.na(x))) # This line of code counts how many NA's in each column
```

    ##                    class                cap.shape              cap.surface 
    ##                        0                        0                        0 
    ##                cap.color                  bruises                     odor 
    ##                        0                        0                        0 
    ##          gill.attachment             gill.spacing                gill.size 
    ##                        0                        0                        0 
    ##               gill.color              stalk.shape               stalk.root 
    ##                        0                        0                        0 
    ## stalk.surface.above.ring stalk.surface.below.ring   stalk.color.above.ring 
    ##                        0                        0                        0 
    ##   stalk.color.below.ring                veil.type               veil.color 
    ##                        0                        0                        0 
    ##              ring.number                ring.type        spore.print.color 
    ##                        0                        0                        0 
    ##               population                  habitat 
    ##                        0                        0

There is no NAs in any column so we don’t need to clean the data for
this purpose.

``` r
mushrooms %>% pivot_longer(cols=!which(colnames(mushrooms)=='class')) %>% 
  ggplot() +
  geom_histogram(aes(x=value, fill=class), stat='count') + 
  facet_wrap(~name)
```

![](classification_mushrooms_edible_poisonous_files/figure-gfm/predictor%20variables-1.png)<!-- -->

### Data Preparation

To create a reliable model the dataset must be divided into a train and
test set, 70% of the data will be used to train and 30% to test. It’s
also important that the outcome variable is balanced in both the train
and test set.

``` r
set.seed(1994)
n_train <- floor(nrow(mushrooms) * 0.7)
train_rows <- sample(nrow(mushrooms), n_train)
train_set <- mushrooms[train_rows,]
test_set <- mushrooms[-train_rows,]

ggplot(data = train_set, aes(x=class)) +
  geom_histogram(stat='count') +
  labs(title = 'Training Set outcome variable: Edible vs Poisonous') +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5))

ggplot(data = test_set, aes(x=class)) +
  geom_histogram(stat='count') +
  labs(title = 'Test Set outcome variable: Edible vs Poisonous') +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5))
```

<img src="classification_mushrooms_edible_poisonous_files/figure-gfm/divide train test-1.png" width="50%" /><img src="classification_mushrooms_edible_poisonous_files/figure-gfm/divide train test-2.png" width="50%" />
The split between edible and poisonous in the train set is very similiar
to the entire dataset and the test set is balanced so we now continue
with our classification.

### Classification

Before we begin to classify we must decide on a model evaluation
technique. In this case we have a balanced binary outcome so can use
both the Receiver Operating Curve (ROC) and Area Under the Curve (AUC)
to compare machine learning techniques.

We will use the AUC and the model which has the highest AUC will be
considered the best model.

``` r
tree <- rpart(class~., data=train_set, method="class")
plot(tree)
text(tree)
```

![](classification_mushrooms_edible_poisonous_files/figure-gfm/decision%20tree-1.png)<!-- -->

``` r
tree_predict <- predict(tree, test_set)
# find which class has the highest probability
max_prob_tree <- colnames(tree_predict)[max.col(tree_predict,ties.method="first")]
roc_tree = roc(test_set$class, factor(max_prob_tree, ordered = TRUE))
auc_tree = auc(roc_tree)
auc_tree
```

    ## Area under the curve: 0.9934

The decision tree has 99% accuracy which is very high, normally this
would indicate the model has been over-fitted but from looking at the
decision tree it has been dominated by odor as the predictor variable.
We saw in the data exploration that odor was correlated strongly with
mushroom edibility.

``` r
rf <- randomForest(class~., data=train_set)
predictions_rf = predict(rf, test_set)

roc_rf = roc(test_set$class, factor(predictions_rf, ordered = TRUE))
auc_rf = auc(roc_rf)
auc_rf
```

    ## Area under the curve: 1

Random Forest also gives 100% accuracy and in the plot below we can see
that odor is again much more important than all other predictor
variables.

``` r
rf$importance %>% as.data.frame() %>%
  tibble::rownames_to_column("variable") %>%
  ggplot(aes(x=reorder(variable,-MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat='identity')
```

![](classification_mushrooms_edible_poisonous_files/figure-gfm/plot%20rf%20importance-1.png)<!-- -->

### Excluding odor as a predictor variable

What if we excluded odor as a predictor variable?

``` r
tree <- rpart(class~., data=train_set[,-which(colnames(train_set) == 'odor')], method="class")
plot(tree)
text(tree)
```

![](classification_mushrooms_edible_poisonous_files/figure-gfm/decision%20tree%20excluding%20odor-1.png)<!-- -->

Now the decision tree has much more nodes which shows how dominant odor
was in predicting edibility.

``` r
tree_predict <- predict(tree, test_set)
# find which class has the highest probability
max_prob_tree <- colnames(tree_predict)[max.col(tree_predict,ties.method="first")]
roc_tree = roc(test_set$class, factor(max_prob_tree, ordered = TRUE))
auc_tree = auc(roc_tree)
auc_tree
```

    ## Area under the curve: 0.9874

After excluding odor as a predictor variable the model still has a very
high accuracy of almost 99%.

### Classifying with colour as the sole predictor

So what if we wanted to predict if a mushroom was edible based solely on
it’s colour? Personally I would find this useful if I saw a mushroom in
a field. I could just use the colour as an indicator of it was poisonous
and wouldn’t have to consider other mushroom characteristics.

``` r
tree <- rpart(class~., data=train_set[,c('class', 'spore.print.color')], method="class")
tree_predict <- predict(tree, test_set)
# find which class has the highest probability
max_prob_tree <- colnames(tree_predict)[max.col(tree_predict,ties.method="first")]
roc_tree = roc(test_set$class, factor(max_prob_tree, ordered = TRUE))
auc_tree = auc(roc_tree)
auc_tree
```

    ## Area under the curve: 0.8645

Now the accuracy has dropped to 86%, let’s explore other techniques to
check if they can do better than a decision tree.

``` r
rf <- randomForest(class~., data=train_set[,c('class', 'spore.print.color')])
predictions_rf = predict(rf, test_set)

roc_rf = roc(test_set$class, factor(predictions_rf, ordered = TRUE))
auc_rf = auc(roc_rf)
auc_rf
```

    ## Area under the curve: 0.8645

The random forest is giving the same accuracy as the decision tree - as
there is just 1 predictor variable the random forest is essentially the
same as a decision tree.

Can simple logistic regression do as well as the decision tree and
random forest?
