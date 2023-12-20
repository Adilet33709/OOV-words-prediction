# Regression-model-for-OOV-words-prediction
This repository contains a deep learning regression model that tackles the OOV (out of vocabulary) problem for automatically generated vocabulary lists from specific corpora. Gold score for the regression model is a log probability of each word in the vocabulary list in training corpora and features are fasttext embeddings and frequency in SUBTLEX. The updated list outperforms the initial list by 10% according to the time comprehension metric built by John&Uvaliyev that can be accessed from here [https://aclanthology.org/2023.mwe-1.12/]. 

## Before your start
1. Download the fasttext embeddings from https://fasttext.cc/ and put it in a folder where the code is located
2. You can experiment with your own training/test set. Put automatically generated vocabulary words in the first column and log probability in the second column in training_set.xlsx. Put OOV words in the test_set.xlsx file.
3. To measure the improvement according to the text comprehension metric you can use the code at https://github.com/Adilet33709/Automatic-generation-of-vocabulary-lists-with-MWE-expressions. 




