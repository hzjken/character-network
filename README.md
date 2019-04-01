# Character Network
A project on using network graph, NLP techniques (entity recognition, sentiment analysis) to analyse the relationships among characters in a novel. The project takes novel Harry Potter as an example and outputs reasonable results.
  
![harry potter](https://user-images.githubusercontent.com/30411828/55331547-15f2c400-54c6-11e9-9c76-2f779acc83e9.png)

## Why this project?
The relationship among characters in a novel tells an important part of the story. Great novels usually have large and complex character relationship network due to the number of characters and story complexity. If there is a method to automatically **analyse complex character networks** through computer programs, which **takes in novel text** only and is able to **generate a comprehensive visualized network graph** as output, what a good news will it be for lazy readers (like myself) that like skipping parts while don't want to lose important information on the plots!

With such idea in mind, this project came live! In the following parts, I will explain the techniques and implementation details of this automatic character network project, and evaluate its performance on the **Harry Potter** series.

## Key Technqiues in Implementation
**1. Name Entity Recognition**<br>
To identify all the character names showed up in the novel, which will be used in the later processes for co-occurrence counting and sentiment score calculation.

**2. Sentiment Analysis**<br>
To calculate the sentiment score of a certain text context (a sentence, a fixed length of words, a paragraph etc.), which is the base of character sentiment relationship.

**3. Matrix Multiplication**<br>
To vectorise and speed up the calculation of character co-occurrence and sentiment relationship, implemented with numpy.

**4. Network Visualization**<br>
To generate the network graph from data in co-occurrence and sentiment matrix, implemented with networkX.

**5. PySpark Distributed Computing (Optional)**<br>
To parallelize the computation in the procedures of name entity recognition and sentiment analysis for higher speed. The repo provides the code implementation of both a normal version and Pyspark-distributed version. 

## Output Results
![harry potter](https://user-images.githubusercontent.com/30411828/47213848-cebcbf00-d3ce-11e8-905e-0d0701a4c5b5.gif)

not ended
