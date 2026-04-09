# Prompts Used for the SPC4004 Code Generation Project

## Prompt 1 - Initial Code (iteration_1_initial.py)
"Write Python code to predict credit card default using a decision tree 
classifier. Load the dataset from a CSV file called 'credit_default.csv'. 
The target column is called 'default.payment.next.month'. Train a decision 
tree on the data and print the accuracy."

## Prompt 2 - Adding Evaluation Metrics (iteration_2_eval_metrics.py)
"The code only prints accuracy. Can you modify it to also print a full 
classification report showing precision, recall and F1 score. Additionally, 
display a confusion matrix as a heatmap using seaborn."

## Prompt 3 - (iteration_3_depth.py)
"Now modify this code to test decision tree depths from 2 to 15. For each 
depth, record the precision, recall and F1 score on the test set for the 
default class (class 1). Plot all three metrics against tree depth on a 
single graph and print the best depth based on F1 score."

## Prompt 4 - (iteration_4_final.py)
"Can you write a final version of the code that includes: loading the 
dataset from credit_default.csv and dropping the ID column, an 80/20 
train/test split, a decision tree with max_depth=3 and class_weight='balanced',
printing accuracy and a full classification report with target names No 
Default and Default, a confusion matrix heatmap saved as a PNG, and a feature 
importance bar chart showing the top 10 features saved as a PNG."
