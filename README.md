# Character Network
A project on using network graph, NLP techniques (entity recognition, sentiment analysis) to analyse the relationships among characters in a novel. The project takes novel Harry Potter as an example and outputs reasonable results.
  
![harry potter](https://user-images.githubusercontent.com/30411828/55331547-15f2c400-54c6-11e9-9c76-2f779acc83e9.png)

## Why this project?
The relationship among characters in a novel tells an important part of the story. Great novels usually have large and complex character relationship network due to the number of characters and story complexity. If there is a method to automatically **analyse complex character networks** through computer programs, which **takes in novel text** only and is able to **generate a comprehensive visualized network graph** as output, what a good news will it be for lazy readers (like myself) that like skipping parts while don't want to lose important information on the plots!

With such idea in mind, this project came live! In the following parts, I will explain the techniques and implementation details of this automatic character network project, and evaluate its performance on the **Harry Potter** series.

## Final Output
Before we go deep into the tedious implementation details, let's first have a look on the **fancy part**!
![harry potter](https://user-images.githubusercontent.com/30411828/47213848-cebcbf00-d3ce-11e8-905e-0d0701a4c5b5.gif)
Above is the sentiment graph outputs of the novel series, each **node** represents a character in the novel and each **edge** represents the relationship between the two characters it's connected to. 

In terms of node, the **node size** represents the importance of a character, which is measured by the number of occurrence of its name in the novel. With no surprise, Harry, Ron and Hermione take up the top 3 characters as the graph shows. 

In terms of edge, each edge has a different **color**, from bright (yellow) to dark (purple), which represents the **sentiment relationship** between two characters, or in a more human-understandable way, **hostile or friendly** relationship. The **brighter** the color of the edge, the more **friendly** the relationship is; The **darker** the color, the more **hostile** it is. Just with a general look, you can easily find out that Harry possesses most of the bright connections with other characters while Voldemort does the opposite, nearly all dark connections with others, which makes sense as they are the two opposite poles in the novel. 

Besides, the graph's edges change along with the story series proceeds. This is because we split the whole novel series by episode itself to generate one set of edge parameters resepectively for each episode, so that we can see the relationship changes among characters as story proceeds. 

Personally I do find this graph quite reasonable as the relationships it shows correspond with some plots that I, a person who has watched the whole series of HP movies 3 times (LOL), remembered.  You could also have a detailed check on the correctness of the graph  here if you are also an enthusiastic HP fan. If not, let's carry on to the technical part so that you can apply it on your favorites! 

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

## Steps
**Data Preparation**<br>
Before we start processing and analysing the novels, we need to prepare the **novel** and **common words** files. The **novel** files contain novel text of the whole story, which would be split into sentences for later computation. **Common words** file contains the commonly used 4000+ English words, which can be downloaded easily elsewhere. The purpose of this file is to reduce the errors in the procedure of name entity recognition by removing non-name words that appear in it.



not ended
