import pandas as pd

Reddit_data = pd.read_csv('posts_comments_reddit.csv')
Reddit_data['BERT_class'] = [str(value).title() for value in Reddit_data['BERT_class']]



# Function to calculate accuracy
def calculate_accuracy(df, predictions):
    correct = sum(df[predictions] == df['manuel_class'])
    total = len(df[predictions])
    accuracy = correct / total
    return accuracy



VADER_Accuracy = calculate_accuracy(Reddit_data,'VADER_class')
print (f'VADER_class Accuracy = {VADER_Accuracy}')

BERT_Accuracy = calculate_accuracy(Reddit_data,'BERT_class')
print (f'BERT_class Accuracy = {BERT_Accuracy}')

print ('*' + '=' * 50 + '*')
# Function to calculate precision Positive
def calculate_precision_positive(df, predictions):
    positive_predicted = df[df[predictions] == 'Positive']
    true_positive = positive_predicted[positive_predicted['manuel_class'] == 'Positive']
    false_positive = positive_predicted[positive_predicted['manuel_class'] != 'Positive']
    precision = len(true_positive) / (len(true_positive) + len(false_positive))
    return precision

vader_class = calculate_precision_positive(Reddit_data,'VADER_class')
print(f"VADER_class precision positive class: {vader_class:.2f}")

BERT_class = calculate_precision_positive(Reddit_data,'BERT_class')
print(f"BERT_class precision positive class: {BERT_class:.2f}")

print ('*' + '=' * 50 + '*')
# Function to calculate recall Positive score
def calculate_recall_positive(df, predictions):
    true_positive = df[(df[predictions] == 'Positive') & (df['manuel_class'] == 'Positive')]
    false_negative = df[(df[predictions] != 'Positive') & (df['manuel_class'] == 'Positive')]
    recall_score = len(true_positive) / (len(true_positive) + len(false_negative))
    return recall_score

vader_class = calculate_recall_positive(Reddit_data,'VADER_class')
print(f"VADER_class recall score positive class: {vader_class:.4f}")

BERT_class = calculate_recall_positive(Reddit_data,'BERT_class')
print(f"BERT_class recall score positive class: {BERT_class:.4f}")

print ('*' + '=' * 50 + '*')
# Function to calculate precision Negative
def calculate_precision_negative(df, predictions):
    true_negative = df[(df[predictions] == 'Negative') & (df['manuel_class'] == 'Negative')]
    false_positive = df[(df[predictions] == 'Negative') & (df['manuel_class'] != 'Negative')]
    precision = len(true_negative) / (len(true_negative) + len(false_positive))
    return precision

vader_class = calculate_precision_negative(Reddit_data,'VADER_class')
print(f"VADER_class precision negative class: {vader_class:.4f}")

BERT_class = calculate_precision_negative(Reddit_data,'BERT_class')
print(f"BERT_class precision negative class: {BERT_class:.4f}")

print ('*' + '=' * 50 + '*')
# Function to calculate recall Negative score
def calculate_recall_negative(df, predictions):
    true_negative = df[(df[predictions] == 'Negative') & (df['manuel_class'] == 'Negative')]
    false_negative = df[(df[predictions] != 'Negative') & (df['manuel_class'] == 'Negative')]
    recall = len(true_negative) / (len(true_negative) + len(false_negative))
    return recall

vader_class = calculate_recall_negative(Reddit_data,'VADER_class')
print(f"vader_class recall score negative class: {vader_class:.4f}")

BERT_class = calculate_recall_negative(Reddit_data,'BERT_class')
print(f"BERT_class recall score negative class: {BERT_class:.4f}")
