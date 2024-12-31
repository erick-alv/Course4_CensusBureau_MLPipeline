# Model Card

## Model Details
This project uses a logistic regression using the default hyperparameters in scikit-learn version 1.3.2.

## Intended Use
The purpose of this model is to predict if the salary of a person is less equal or greater than 50K.

## Training Data
The training data belongs to the [Census Income](https://archive.ics.uci.edu/dataset/20/census+income) dataset of the UC Irvine.

## Evaluation Data
For the evaluation a test split was extracted from the training data using the function 'train_test_split' of scikit-learn version 1.3.2. The split size is 20% of the data.

## Metrics
The overall performance of the model on the whole test split is the following.

precision: 0.71

recall: 0.27

F1: 0.39

For a more detailed overview of the performance on different splits run the [performance test script](starter/performance_test.py).

## Ethical Considerations
There is a class imbalance for many splits of the data. One of them being the native-country of the features. This might cause bias in different applications.

## Caveats and Recommendations
With respect to the aspects mentioned before. Is important to consider that the results of the model come from an imbalanced dataset and measures to prevent biases caused by it should be taken. For example for the split where the education = 10th the model shows a poor performance (precision=0.2).
