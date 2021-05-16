# Data 606 - Capstone Project - UMBC

# Text Summarization

Text summarization is the technique for generating a concise and precise summary of voluminous texts while focusing on the sections that convey useful information, and without losing the overall meaning. Automatic text summarization aims to transform lengthy documents into shortened versions, something which could be difficult and costly to undertake if done manually. Machine learning algorithms can be trained to comprehend documents and identify the sections that convey important facts and information before producing the required summarized texts. 

# Objective

This project aims to provide a text analysis of a given document and summarize (Abstractive summarization) it to a fraction of its length. This is an important and complicated task in natural language processing. For the scope of this project, we will be using Encoder-Decoder architecture with attention mechanism, which is used for natural language processing problems that generate variable length output sequences. 

# EDA
We need to clean our dataset before we perform an Exploratory Data Analysis, on the data. Data cleaning is performed by the following manner:
Removal of punctuations, numbers, special symbols and stop-words. Though they are responsible for maintaining the context in a paragraph, we remove these items because they do not imply any meaning by themselves. 
Conversion of text to lower-case and tokenizing it. I have used NLTK (Natural Language Toolkit)  module to tokenize the text.
Lemmatize the tokens and perform Topic Modeling, Word Cloud and Name-Entity recognition.

# Model

PEGASUS: Pre-Training with Extracted Gap Sentences for Abstractive Summarization

Pegasus works on a pre-trained self-supervised objective called 'gap-sentence' generation for Transformer encoder-decoder models to improve fine-tuning models on abstractive summarization. In PEGASUS pre-training several whole sentences are removed from the documents and model is tasked with recovering them. An example input for pre-training is a document with missing sentences, while the output consists of the missing sentences concatenated together. 
The advantage of this self-supervision is that you can create as many examples as there are documents, without any human annotation, which is often the bottleneck in purely supervised systems.

# Deployment

Optimizer

Adafactor is a stochastic optimization algorithm based on Adam that reduces memory usage while retaining the empirical benefits of adaptivity by maintaining factored representation of the squared gradient accumulator across training steps. By tracking moving averages of the row and column sums of the squared gradients for matrix-valued variables  they were able to reconstruct a low-rank approximation of the exponentially smoothed accumulator at each training step that is optimal with respect to the generalized Kullback-Leibler divergence. For this project I have set the hyperparameters to default values proposed in the research paper.

Loss - Cross Entropy

The CrossEntropyLoss in pytorch combines LogSoftmax and NLLLoss in one single class. It is useful when training a classification problem with C classes. If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set. The input is expected to contain raw, unnormalized scores for each class. input has to be a Tensor of size either (minibatch, C) or (minibatch,C,d1,d2,...,dK) with K≥1 for the K-dimensional case (described later).This criterion expects a class index in the range [0,C−1] as the target for each value of a 1D tensor of size minibatch; if ignore_index is specified, this criterion also accepts this class index (this index may not necessarily be in the class range).

Scoring Metric - BLEU

The BLEU algorithm compares consecutive phrases of the automatic translation with the consecutive phrases it finds in the reference translation, and counts the number of matches, in a weighted fashion. These matches are position independent. A higher match degree indicates a higher degree of similarity with the reference translation, and higher score. Intelligibility and grammatical correctness are not taken into account. BLEU’s strength is that it correlates well with human judgment by averaging out individual sentence judgment errors over a test corpus, rather than attempting to devise the exact human judgment for every sentence. 

