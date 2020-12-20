# AniBrain-Anime_LDA_Topic_Similarity_Model
LDA Topic Model for Anime Topic Distribution Similarity

## What is topic modelling?
Topic models learn topics—typically represented as sets of important words—automatically from unlabelled documents in an unsupervised way.

## What is LDA? How does it work?
"Latent Dirichlet Allocation(LDA) is a popular algorithm for topic modeling.

LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.

Once you provide the algorithm with the number of topics, all it does is rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.

*When I say **topic**, what is it actually and how it is represented?*

A **topic** is nothing but a collection of dominant keywords that are typical representatives. Just by looking at the keywords, you can identify what the topic is all about." [[1]](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#4whatdoesldado)

## How do I use LDA to make recommendations?
After preprocessing my dataset and determining the number of topics to use (through viewing coherence scores and playing with different models), I create a matrix of the topical distributions for every anime the model was trained on. I use the Jensen Shannon divergence method to measure the similarity of their topical distributions and choose those that are most (or least if using a different measure) similar.
<br/>
<br/>
The documents were constructed by merging the synopsis of an anime with their genres and rating. I did this because I realized the synopses alone weren't strong enough to make *good* recommendations. Imagine there's a kids show which talks about ninjas *(A)* and a different violent, murder-filled, bloody show about ninjas *(B)*. I saw cases where *(A)* could be a recommendation of *(B)* due to them both being heavily ninja focused. The synopses I worked with weren't as long as I'd like, making it easy for two very different shows to seem similar based on purely that alone. Adding in genres and ratings into my bag of words helped the model learn to associate shows with similar genres/ratins together. By doing this, the model's coherence scores dropped but it learned to make *better* recommendations (in my opinion).

## What are the names of the topics found?
These are the names I gave each topic based on the word distribution (found below).
<br/>
<br/>
Topic No. | Topic Name
--- | ---
0 | R-17+ Mystery
1 | War
2 | Magical World
3 | Sports
4 | Action Sci-Fi
5 | Comedy
6 | Slice of Life
7 | Romance
8 | Adventure
9 | Kids
10 | Action Historical
11 | Supernatural
12 | Music
13 | Sci-Fi Mecha
14 | Fantasy World
15 | School
16 | Hentai
17 | Family

## What are the word distributions of the topics?
These are the top 10 words for every topic.
<br/>
<br/>
**0:** 0.034*"mystery" + 0.024*"r___violence__profanity" + 0.020*"supernatural" + 0.017*"horror" + 0.015*"psychological" + 0.015*"find" + 0.013*"death" + 0.012*"mysterious" + 0.012*"drama" + 0.012*"action"
<br/><br/>
**1:** 0.030*"war" + 0.027*"military" + 0.025*"action" + 0.014*"r___violence__profanity" + 0.012*"force" + 0.012*"drama" + 0.011*"empire" + 0.011*"battle" + 0.011*"nation" + 0.010*"country"
<br/><br/>
**2:** 0.058*"magic" + 0.030*"world" + 0.030*"fantasy" + 0.023*"princess" + 0.017*"girl" + 0.017*"comedy" + 0.017*"magical" + 0.016*"witch" + 0.016*"shoujo" + 0.015*"kingdom"
<br/><br/>
**3:** 0.049*"game" + 0.031*"team" + 0.027*"sports" + 0.021*"shounen" + 0.020*"pg__teens__or_older" + 0.017*"player" + 0.015*"play" + 0.013*"win" + 0.012*"action" + 0.012*"world"
<br/><br/>
**4:** 0.028*"action" + 0.021*"scifi" + 0.020*"police" + 0.015*"city" + 0.015*"world" + 0.014*"comedy" + 0.012*"pg__teens__or_older" + 0.011*"group" + 0.010*"adventure" + 0.010*"work"
<br/><br/>
**5:** 0.051*"comedy" + 0.039*"girl" + 0.032*"school" + 0.027*"romance" + 0.025*"ecchi" + 0.023*"r__mild_nudity" + 0.014*"harem" + 0.014*"pg__teens__or_older" + 0.013*"day" + 0.012*"life"
<br/><br/>
**6:** 0.017*"pg__teens__or_older" + 0.016*"day" + 0.016*"girl" + 0.016*"begin" + 0.015*"find" + 0.014*"life" + 0.014*"drama" + 0.014*"time" + 0.012*"friend" + 0.011*"live"
<br/><br/>
**7:** 0.027*"pg__teens__or_older" + 0.022*"romance" + 0.021*"life" + 0.020*"love" + 0.018*"drama" + 0.017*"slice_of_life" + 0.017*"comedy" + 0.017*"friend" + 0.015*"girl" + 0.014*"school"
<br/><br/>
**8:** 0.045*"adventure" + 0.030*"island" + 0.023*"find" + 0.022*"comedy" + 0.020*"fantasy" + 0.017*"friend" + 0.017*"pg__children" + 0.013*"shounen" + 0.012*"kids" + 0.012*"dragon"
<br/><br/>
**9:** 0.039*"g__all_ages" + 0.032*"kids" + 0.027*"adventure" + 0.019*"friend" + 0.019*"child" + 0.017*"comedy" + 0.017*"fantasy" + 0.015*"pg__children" + 0.014*"live" + 0.013*"world"
<br/><br/>
**10:** 0.026*"action" + 0.023*"historical" + 0.020*"samurai" + 0.020*"adventure" + 0.018*"man" + 0.016*"martial_arts" + 0.015*"clan" + 0.015*"ninja" + 0.014*"kill" + 0.012*"shounen"
<br/><br/>
**11:** 0.045*"human" + 0.033*"supernatural" + 0.025*"vampire" + 0.021*"monster" + 0.020*"demon" + 0.016*"world" + 0.014*"horror" + 0.014*"action" + 0.014*"r___violence__profanity" + 0.014*"demons"
<br/><br/>
**12:** 0.042*"music" + 0.020*"comedy" + 0.017*"g__all_ages" + 0.015*"idol" + 0.014*"pg__teens__or_older" + 0.012*"japanese" + 0.010*"band" + 0.010*"work" + 0.009*"girl" + 0.009*"include"
<br/><br/>
**13:** 0.046*"scifi" + 0.033*"earth" + 0.029*"mecha" + 0.027*"space" + 0.024*"action" + 0.022*"planet" + 0.020*"pg__teens__or_older" + 0.017*"adventure" + 0.017*"robot" + 0.015*"alien"
<br/><br/>
**14:** 0.032*"world" + 0.030*"fantasy" + 0.024*"adventure" + 0.023*"action" + 0.022*"power" + 0.017*"magic" + 0.016*"pg__teens__or_older" + 0.014*"demon" + 0.013*"battle" + 0.012*"hero"
<br/><br/>
**15:** 0.098*"school" + 0.043*"student" + 0.032*"club" + 0.023*"girl" + 0.022*"pg__teens__or_older" + 0.019*"class" + 0.018*"high_school" + 0.018*"comedy" + 0.016*"member" + 0.012*"teacher"
<br/><br/>
**16:** 0.066*"hentai" + 0.064*"rx__hentai" + 0.021*"woman" + 0.020*"girl" + 0.019*"man" + 0.017*"day" + 0.015*"sex" + 0.013*"sexual" + 0.012*"love" + 0.012*"work"
<br/><br/>
**17:** 0.048*"father" + 0.044*"family" + 0.033*"mother" + 0.024*"live" + 0.021*"drama" + 0.020*"life" + 0.014*"daughter" + 0.014*"day" + 0.013*"g__all_ages" + 0.012*"son"

## How can I use this?
The model is easily accessible through [docker](https://hub.docker.com/r/koji98/anibrain_anime_lda_topic_similarity_model). The model is wrapped in a 
REST API made with [FastAPI](https://fastapi.tiangolo.com/) so it can easily be accessed as a microservice.

## References 
[1] [Topic Modeling with Gensim (Python) by  Selva Prabhakaran](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#4whatdoesldado)
