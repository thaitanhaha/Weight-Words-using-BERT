# Weight words using BERT

My task is to assign weights to words in a sentence on an economic topic.

## Dataset

I first filtered the Vietnamese [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) using the ChatGPT-4 model, then manually filtered it again to obtain 2002 documents related to economic topics. The English documents are retrieved by obtaining the corresponding English version of each Vietnamese document.

The dataset include:
- id
- url: wikipedia URL of the document
- title: title of the document
- text: full text of the document

The code for filtering with ChatGPT-4 and retrieving English documents are included above.

My dataset can be found at [Economic Documents](https://www.kaggle.com/datasets/nguyenthaitan/economic-documents).

## Method

First, I have to label the words with weights. I use `TF-IDF` to get top 15% or 25% importance words of each document. Then I split each document into sentences. 

### Weight with 0-1 label

In each sentence, if a word is among the most important words, it is labeled as 1. If the word is a stopword, it is labeled as 0. Otherwise, it is labeled as 0.

For Vietnamese, I used `PhobertTokenizerFast`, `RobertaForTokenClassification` with pretrained `vinai/phobert-base-v2`.

For English, I used `AutoTokenizer`, `BertForTokenClassification` with pretrained `bert-base-uncased`.

### Weight with distribution

In each sentence, if a word is among the most important words, it is labeled as its TF-IDF score. If the word is a stopword, it is labeled as 0. Otherwise, it is labeled as 0. Then in each sentence, applying `softmax` for non-zero labels, we get something like a distribution. 

The `forward` processes input through a BERT model to generate embeddings, applies a linear transformation to calculate word importance scores, and then masks out padding tokens. The resulting word weights are converted into a probability distribution using a `softmax` function and returned as the final output.

In the `backward`, we calculate the loss between two distributions, which is why the `KL-divergence` is used.

For Vietnamese, I used `PhobertTokenizerFast`, `AutoModel` with pretrained `vinai/phobert-base-v2`.

For English, I used `AutoTokenizer`, `AutoModel` with pretrained `bert-base-uncased`.

## 

More can be found at [English Kaggle notebook](https://www.kaggle.com/code/nguyenthaitan/weight-words) and [Vietnamese Kaggle notebook](https://www.kaggle.com/code/nguyenthaitan/weight-words-vietnamese).




