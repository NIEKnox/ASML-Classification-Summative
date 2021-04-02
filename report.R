'
Summative assignment code
Z code: Z0158114
'
# installing packages

packages <- c("skimr", "tidyverse", "ggplot2", "dplyr", "mlr3verse", "data.table", 
              "rsample", "DataExplorer", "GGally", "scales", "recipes", "keras")

install.packages(packages)

# Loading all necessary packages
library(skimr)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(mlr3verse)
library(data.table)
library(rsample)
library(DataExplorer)
library(GGally)
library(scales)
library(recipes)
library(keras)

# Reading the data and initial look
heart.data <- read.csv("heart_failure.csv")

skim(heart.data)

DataExplorer::plot_boxplot(heart.data, by = "fatal_mi")

fatal.col <- heart.data$fatal_mi
names(heart.data)

pairs(~ age + creatinine_phosphokinase + ejection_fraction + platelets + serum_creatinine + serum_sodium, 
      data=heart.data, col=alpha(fatal.col+2, 0.5), pch = 19)
fatal.nonlist <- unlist(heart.data$fatal_mi)




# Separating the data into test, train and validate for Deep Learning

heart.split1 <- initial_split(heart.data, 0.5)
heart.split2 <- initial_split(testing(heart.split1), 0.5)

heart.train <- training(heart.split1) 
heart.validate <- training(heart.split2)
heart.test <- testing(heart.split2)


# Logistic Regression

# # Reformat target data appropriately
# 
# heart.data$fatal_mi <- plyr::mapvalues(heart.data$fatal_mi, from=c(0, 1), to = c("nonfatal", "fatal"))


set.seed(1234) # set seed to ensure reproducibility

names(heart.data)


# Define task

# Defining levels for fatal col

heart.data$fatal_mi <- factor(heart.data$fatal_mi)

heart.refresh <- function(){
  heart.backend <- as_data_backend(heart.data)
  heart.task <- TaskClassif$new(id = "heart_fatal", 
                                backend = heart.backend, 
                                target = "fatal_mi", 
                                positive = "1")
  heart.task
}

heart.backend <- as_data_backend(heart.data)
heart.task <- TaskClassif$new(id = "heart_fatal", 
                              backend = heart.backend, 
                              target = "fatal_mi", 
                              positive = "1")

# Cross validation
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart.task)

# Define a collection of base learners
lrn.baseline <- lrn("classif.featureless", predict_type = "prob")
lrn.cart     <- lrn("classif.rpart", predict_type = "prob")
lrn.cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn.ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn.xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn.log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Define a super learner
lrnsp.log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Factors coding pipeline
pl.factor <- po("encode")

# Now define the full pipeline

# Learners require no extra modification to input  
?po
lrn.spr <- gunion(list(
    po("learner_cv", lrn.baseline),
    po("learner_cv", lrn.cart),
    po("learner_cv", lrn.cart_cp),
    po("learner_cv", lrn.ranger),
    po("learner_cv", lrn.log_reg),
    po("learner_cv", lrn.xgboost)
  ))%>>%
    po("featureunion") %>>%
    po(lrnsp.log_reg)

# Pipeline
lrn.spr$plot()

# Fit the base learners and super learner and evaluate
res.spr <- resample(heart.task, lrn.spr, cv5, store_models = TRUE)
res.spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

heart.refresh()

res.cart <- resample(heart.task, lrn.cart, cv5, store_models = TRUE)
res.cart$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

heart.refresh()

res.cart_cp <- resample(heart.task, lrn.cart_cp, cv5, store_models = TRUE)
res.cart_cp$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))

heart.refresh()

res.ranger <- resample(heart.task, lrn.ranger, cv5, store_models = TRUE)
res.ranger$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))

heart.refresh()

res.log_reg <- resample(heart.task, lrn.log_reg, cv5, store_models = TRUE)
res.log_reg$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))

heart.refresh()

res.xgboost <- resample(heart.task, lrn.xgboost, cv5, store_models = TRUE)
res.xgboost$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))




# Deep learning
# Preprocessing

cake <- recipe(fatal_mi ~ ., data = heart.data) %>%
  step_meanimpute(all_numeric()) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
  step_unknown(all_nominal(), -all_outcomes()) %>% #
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  prep(training = heart.train) 

heart.train.final <- bake(cake, new_data = heart.train) # apply preprocessing to training data
heart.validate.final <- bake(cake, new_data = heart.validate) # apply preprocessing to validation data
heart.test.final <- bake(cake, new_data = heart.test) # apply preprocessing to testing data

heart.train.x <- heart.train.final %>%
  select(-starts_with("fatal")) %>%
  as.matrix()
heart.train.y <- heart.train.final %>%
  select(fatal_mi) %>%
  as.matrix()

heart.validate.x <- heart.validate.final %>%
  select(-starts_with("fatal")) %>%
  as.matrix()
heart.validate.y <- heart.validate.final %>%
  select(fatal_mi) %>%
  as.matrix()

heart.test.x <- heart.test.final %>%
  select(-starts_with("fatal")) %>%
  as.matrix()
heart.test.y <- heart.test.final %>%
  select(fatal_mi) %>%
  as.matrix()

# Fixing normalising of y vals for the net

heart.train.y <- heart.train.y - min(heart.train.y)
heart.train.y <- heart.train.y/max(heart.train.y)

heart.validate.y <- heart.validate.y - min(heart.validate.y)
heart.validate.y <- heart.validate.y/max(heart.validate.y)

heart.test.y <- heart.test.y - min(heart.test.y)
heart.test.y <- heart.test.y/max(heart.test.y)




# Constructing neural net

# For ease of recompiling it when needed

deepnet.function <- function(dropout.rate){
  heart.net <- keras_model_sequential() %>%
    layer_dense(units = 60, activation = "relu",
                input_shape = c(ncol(heart.train.x))) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout.rate) %>%
    layer_dense(units = 30, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout.rate) %>%
    layer_dense(units = 30, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout.rate) %>%
    layer_dense(units = 30, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout.rate) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  heart.net %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )
  
  return(heart.net)
  }

# training for different dropouts

dropout.rates <- c(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
dictids <- as.character(dropout.rates)

test.probs <- c()
test.ress <- c()

test.accs <- c()
test.rocs <- c()

heart.history.training <- {}
heart.history.validation <- {}

epochnum <- 50

for (i in (1:8)){
  thisnet <- deepnet.function(dropout.rates[i])
  
  history <- thisnet %>% fit(
    heart.train.x, heart.train.y,
    epochs = epochnum,
    validation_data = list(heart.validate.x, heart.validate.y),
  )
  
  
  training <- slice(as.data.frame(history)[2], 51:100)
  validation <- slice(as.data.frame(history)[2], 151:200)
  
  heart.history.training[dictids[i]] <- training
  heart.history.validation[dictids[i]] <- validation
  
  heart.test.prob <- thisnet %>% predict_proba(heart.test.x)
  
  heart.test.res <- thisnet %>% predict_classes(heart.test.x)
  
  thisnet.accuracy <- yardstick::accuracy_vec(as.factor(heart.test.y),
                          as.factor(heart.test.res))
  test.accs <- c(test.accs, thisnet.accuracy)
  thisnet.rocauc <- yardstick::roc_auc_vec(factor(heart.test.y, levels = c("1","0")),
                         c(heart.test.prob))
  test.rocs <- c(test.rocs, thisnet.rocauc)
}


# Plotting the data

heart.histrain.df <- as_tibble(heart.history.training)
heart.histval.df <- as_tibble(heart.history.validation)

col1 <- 3
col2 <- 4
pointype <- 19

par(mfrow=c(4, 2))

plot(heart.histrain.df$'0.2', main='Drop out = 0.2', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.2', col=col2, pch=pointype)
legend("bottomright", legend=c("Training", "Validation"), fill=c(col1, col2))

plot(heart.histrain.df$'0.25', main='Drop out = 0.25', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.25', col=col2, pch=pointype)

plot(heart.histrain.df$'0.3', main='Drop out = 0.3', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.3', col=col2, pch=pointype)

plot(heart.histrain.df$'0.35', main='Drop out = 0.35', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.35', col=col2, pch=pointype)

plot(heart.histrain.df$'0.4', main='Drop out = 0.4', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.4', col=col2, pch=pointype)

plot(heart.histrain.df$'0.45', main='Drop out = 0.45', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.45', col=col2, pch=pointype)

plot(heart.histrain.df$'0.5', main='Drop out = 0.5', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.5', col=col2, pch=pointype)

plot(heart.histrain.df$'0.55', main='Drop out = 0.55', xlab = 'Epoch', ylab = 'Accuracy', col = col1, pch=pointype)
points(heart.histval.df$'0.55', col=col2, pch=pointype)

plot(y = test.accs, x = dropout.rates, main="Accuracy for each drop out rate", ylab="Accuracy", xlab="Dropout Rate", ylim=c(0.72, 0.85), pch=19, col=col2, type='l')
lines(x = dropout.rates, y = test.rocs, pch=19, col=col1)
legend("topright", legend=c("Training", "Validation"), fill=c(col1, col2))

dropout.rate <- 0.35

unitnum <- 30

# Using the best dropout rate
best.net <- keras_model_sequential() %>%
  layer_dense(units = 60, activation = "relu",
            input_shape = c(ncol(heart.train.x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = dropout.rate) %>%
  layer_dense(units = unitnum, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = dropout.rate) %>%
  layer_dense(units = unitnum, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = dropout.rate) %>%
  layer_dense(units = unitnum, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = dropout.rate) %>%
  layer_dense(units = 1, activation = "sigmoid")

best.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy"))

bestnet.history <- best.net %>% fit(
  heart.train.x, heart.train.y,
  epochs = 100,
  validation_data = list(heart.validate.x, heart.validate.y),
)

bestnet.prob <- best.net %>% predict_proba(heart.test.x)

# To get the raw classes (assuming 0.5 cutoff):
bestnet.res <- best.net %>% predict_classes(heart.test.x)

# Confusion matrix/accuracy/AUC metrics
# (recall, in Lab03 we got accuracy ~0.80 and AUC ~0.84 from the super learner,
# and around accuracy ~0.76 and AUC ~0.74 from best other models)
table(bestnet.res, heart.test.y)
yardstick::accuracy_vec(as.factor(heart.test.y),
                        as.factor(bestnet.res))
yardstick::roc_auc_vec(factor(heart.test.y, levels = c("1","0")),
                       c(bestnet.prob))





