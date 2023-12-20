# Regression-model-for-OOV-words-prediction
This repository contains a deep learning regression model that tackles the OOV (out of vocabulary) problem for automatically generated vocabulary lists from specific corpora. Gold score for the regression model is a log probability of each word in the vocabulary list in training corpora and features are fasttext embeddings and frequency in SUBTLEX [1]. The updated list outperforms the initial list by 10% according to the time comprehension metric built by John&Uvaliyev [2] 

## Before your start
1. Download the fasttext embeddings from https://fasttext.cc/ and put it in a folder where the code is located
2. You can experiment with your own training/test set. Put automatically generated vocabulary words in the first column and log probability in the second column in training_set.xlsx. Put OOV words in the test_set.xlsx file.
3. To measure the improvement according to the text comprehension metric you can use the code at https://github.com/Adilet33709/Automatic-generation-of-vocabulary-lists-with-MWE-expressions. 

## References
1. Cai, Q., & Brysbaert, M. (2010). SUBTLEX-CH: Chinese word and character frequencies based on film subtitles. PloS one, 5(6), e10729.
2. Lee, J. S., & Uvaliyev, A. (2023, May). Automatic Generation of Vocabulary Lists with Multiword Expressions. In Proceedings of the 19th Workshop on Multiword Expressions (MWE 2023) (pp. 81-86).

