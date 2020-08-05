# Fraud Detection

This repository is for one course project, where I played around fraud loan datasets with machine learning algorithms, and further with basic neural networks.

## Project Description

The project is designed to farmilirize people with data cleanning, the implementation of basic machine learning algorithms, and neural networks. It is based on a fraud loan dataset, which is highly unbalanced.

I did the predition both by machine learning algorithms and neural networks. And apparently, it turned out that neural networks performs much better in predicting the fraud loans in terms whatever metrics being used, such as AUROC score or the self-defined losses.

## Dataset

The dataset is availabal on Google Drive through the following link: <https://drive.google.com/drive/folders/1WOo7QX8BZGE5E2EsvDSByXuzeMrwrIfP?usp=sharing>

Overall, the dataset is highly unbalanced, with less than 5% of loans being fraud. I will describe each dataset here in this section.

Original Dataset

- newtrain_transaction_200000.csv: trainning set with true label
- newtrain_identity.csv: an complement to the trainning set, which saves all the identity information about the loans
- newtest_transaction_withoutlabel.csv: test set without the label
- newtest_identity.csv: an complement to the test set, containning all the indentity information about the loans
- newsample_submission.csv: this dataset includes the loan ID and fraud probability in the same order as 'new_test_transaction_withoutlabel.csv'. The fraud probablity is for replacement with the prediction.

Dataset during trainning

The following datasets are generated during the trainning stage after I did all the data cleaning. The features are all numeric so that you can practice any machine learning algorithms directly with these datasets.

- train_processed.csv: processed trainning set with true label
- submission_processed.csv: processed test set without label

## Code

- dataset_exploration.ipynb: this notebook contains code for data exploration
- notebooks ended with 'my_code': they are code written by myself
- notebooks ended with 'good_example': they are some good examples shared by students in the class
- notebooks ended with 'ml'/'nn': ml refers to machine learning and nn refers to neural networks
