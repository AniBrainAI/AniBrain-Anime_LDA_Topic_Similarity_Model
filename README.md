# AniBrain-Anime_LDA_Topic_Similarity_Model
LDA Topic Model for Anime Synopsis Similarity

## What is topic modelling?
Topic models learn topics—typically represented as sets of important words—automatically from unlabelled documents in an unsupervised way.

## What is LDA? How does it work?
Latent Dirichlet Allocation(LDA) is a popular algorithm for topic modeling.

"LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.

Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.

When I say topic, what is it actually and how it is represented?

A topic is nothing but a collection of dominant keywords that are typical representatives. Just by looking at the keywords, you can identify what the topic is all about." [[1]](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#4whatdoesldado)

## How do I use LDA to make recommendations?
After preprocessing my dataset and determining the number of topics to use (through viewing coherence scores and playing with different models), I create a matrix of the topical distriution for every document in my corpus. I use the Jensen Shannon divergence method to measure the similarity distributions and choose those that are most (or least if using a different measure) similar.

## What are the names of the topics found?
Topic No. | Topic Name
--- | ---
0 | Music
1 | Shonen Action
2 | Supernatural
3 | Romance
4 | Human-Demon
5 | Hentai
6 | Sci-fi
7 | Crime/Mystery
8 | School
9 | War
10 | Kids
11 | Family
12 | Magic
13 | Adventure
14 | Sports

## What are the word distributions of the topics?
Go inside the Jupyter Notebook and you'll find out :) 
*p.s. It's near the bottom of the notebook*

## How can I use this?
The model is easily accessible through [docker] (https://hub.docker.com/r/koji98/anibrain_anime_lda_topic_similarity_model). The model is wrapped is wrapped in a 
REST API so it can easily be as a microservice.

## References 
[1] [Topic Modeling with Gensim (Python) by  Selva Prabhakaran](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#4whatdoesldado)
