import tensorflow as tf
import pandas as pd
# wine quality ai 
# linear regression
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
# code from https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=sQ9iJrSbBTZB

dftrain = pd.read_csv('.\\train.csv', delimiter=';') # training data
dfeval = pd.read_csv('.\\eval.csv', delimiter=';') # testing data
print(dfeval)
print(dftrain)
y_train = dftrain.pop('quality')
y_eval = dfeval.pop('quality')

#CATEGORICAL_COLUMNS = []
NUMERIC_COLUMNS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

feature_columns = []
#for feature_name in CATEGORICAL_COLUMNS:
#  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
#  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))



def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=11)


linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

print(result)  # the result variable is simply a dict of stats about our model

