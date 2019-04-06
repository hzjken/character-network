# Character Network
A project on using network graph, NLP techniques (entity recognition, sentiment analysis) to analyse the relationships among characters in a novel. The project takes novel Harry Potter as an example and outputs reasonable results.
  
![harry potter](https://user-images.githubusercontent.com/30411828/55331547-15f2c400-54c6-11e9-9c76-2f779acc83e9.png)

## Why this project?
The relationship among characters in a novel tells an important part of the story. Great novels usually have large and complex character relationship network due to the number of characters and story complexity. If there is a method to automatically **analyse complex character networks** through computer programs, which **takes in novel text** only and is able to **generate a comprehensive visualized network graph** as output, what a good news will it be for lazy readers (like myself) that like skipping parts while don't want to lose important information on the plots!

With such idea in mind, this project came live! In the following parts, I will explain the techniques and implementation details of this automatic character network project, and evaluate its performance on the **Harry Potter** series.

## Fancy Part!
Before we go deep into the tedious implementation details, let's first have a look on the fancy part!
![ezgif-5-e98e242d28ba](https://user-images.githubusercontent.com/30411828/55665266-d5e96380-586e-11e9-89af-c1a5da88d46f.gif)
Above is the sentiment graph outputs of the novel series, each ***node*** represents a character in the novel and each ***edge*** represents the relationship between the two characters it's connected to. 

In terms of node, the ***node size*** represents the importance of a character, which is measured by the number of occurrence of its name in the novel. With no surprise, Harry, Ron and Hermione take up the top 3 characters as the graph shows. 

In terms of edge, each edge has a different ***color***, from bright (yellow) to dark (purple), which represents the ***sentiment relationship*** between two characters, or in a more human-understandable way, **hostile or friendly** relationship. The **brighter** the color of the edge, the more **friendly** the relationship is; The **darker** the color, the more **hostile** it is. Just with a general look, you can easily find out that Harry possesses most of the bright connections with other characters while Voldemort does the opposite, nearly all dark connections with others, which makes sense as they are the two opposite poles in the novel. 

Besides, the graph's edges change along with the story series proceeds. This is because we split the whole novel series by episode itself to generate one set of edge parameters resepectively for each episode, so that we can see the relationship changes among characters as story proceeds. 

Personally I do find this graph quite reasonable as the relationships it shows correspond with some plots that I, a person who has watched the whole series of HP movies 3 times (LOL), remembered.  You could also have a detailed check on the correctness of the [**graphs**](https://github.com/hzjken/character-network/tree/master/graphs) if you are also an enthusiastic HP fan. If not, let's carry on to the technical part so that you can apply it on your favorites! 

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
**Data Preparation**

Before we start processing and analysing the novels, we need to prepare the **novel** and **common words** files. The [**novel**](https://github.com/hzjken/character-network/tree/master/novels) files contain novel text of the whole story, which would be split into sentences for later computation. [**Common words**](https://github.com/hzjken/character-network/blob/master/common_words.txt) file contains the commonly used 4000+ English words, which can be downloaded easily elsewhere. The purpose of this file is to reduce the errors in the procedure of name entity recognition by removing non-name words that appear in it.

**Name Entity Recognition**

With no prior knowledge into the novel, programs need to figure out the characters in the novel by name entity recognition (NER). In this project, we use the pretrained ***Spacy NER*** classifier. Because the initiation of a [`Spacy NLP`](https://spacy.io/) class takes up loads of memory, we will run the NER process **by sentence** instead of whole novel, where ***PySpark distribution*** can be embedded. For each sentence, we identify the name entities and do a series of processings. One important processing is to split the name into single words if it consists of more than one words, e.g. "Harry Potter". The point is to count the occurrence of a character more accurately, as "Harry" and "Harry Potter" refers to the same character in the novel but word "Harry" shows up more often and "Harry" will be counted where "Harry Potter" is counted. After all the single name words are created, we will filter out the names that show up in **common words**, as some common words might be counted wrongly. Then, we aggregate the names from each sentence, and do a second filter to remove names whose number of occurrence is lower than a **user-defined threshold**, to get rid of some unfrequent recognition mistakes.

**Character Importance**

From the preliminary character name list we get from last step, we can calculate each character’s character importance, or more specifically, the occurrence frequency. This task can be done easily with the Sickit-Learn text processing function [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). After that, we can select the top characters of interest based on their importances. The `top_names` function outputs the top 20 characters and their frequencies as default, but in this Harry Potter example ,we set the number to be 25 to capture a larger network.

**Co-occurence Matrix**

In our project, we pick the simplist definition of co-occurrence that a co-occurrence is observed if two character names show up in the same one sentence. (There might be some other definitions based on paragraphs, number of words, several sentences etc.)  To calculate
co-occurrence, we first need a binary ***occurrence matrix***, that gives information on whether a name occurs in each sentence, again with function `CountVectorizer`. Then, the ***co-occurrence matrix*** equals the dot product of occurrence matrix and its transpose. As co-occurrence is mutually interactive, we will find that the co-occurrence matrix is repeated (symmetric) along the diagonal, so we ***triangularize*** it and set ***diagonal elements*** to be zeros as well.

<p align="center"><img width="220" alt="formula1" src="https://user-images.githubusercontent.com/30411828/55671792-42dc1800-58c6-11e9-973b-d66c7a726f77.png"></p>
<p align="center"><img width="600" alt="formula5" src="https://user-images.githubusercontent.com/30411828/55672777-783a3300-58d1-11e9-93fe-909d78ed01c2.png"></p>

**Sentiment Matrix**

While the co-occurrence matrix above gives information on the co-occurrence or **interaction intensity**,  the ***sentiment matrix*** gives information on the **sentiment intimacy** among characters. The larger the value, the more positive relationship (friends, lovers etc.) between two characters, the lower the value (could be negative), the more negative relationship (enemies, rivals etc.) between two characters. In fact, the calculation process of sentiment matrix is similar to the above one, just that we need to introduce two more concepts.

* **Context Sentiment Score**<br>
Context sentiment score is the sentiment score of each sentence in the novel. In this project, we assumes that the sentiment score of a context (sentence) implies the relationship between two characters that co-occur in the context. For example, if a context has more happy words like “love”, “smile”, “good” etc., which lead to higher sentiment score, we assume that the characters involved in
such context are highly probable to have a positive relationship and vice versa. In implementation, the sentiment score of each sentence is given by NLP library [`Afinn`](https://github.com/fnielsen/afinn) and all the scores are stored together as a **1-D array** for the convenience of ***vectorization*** later.

* **Sentiment Alignment Rate**<br>
Different authors and different types of novels might have different ways of narratives, which will lead to different levels of emotion description. For example, a horror novel should probably have more negative emotion descriptions than a fairy tail. The different description styles will lead to a skewness of the sentiment scores on the overall character network. In extreme cases, the relationships might be all positive or all negative. Therefore, ***sentiment alignment rate*** is introduced to align the sentiment scores and reduce skewness. It means that the sentiment score between two characters will change by one unit of sentiment alignment rate every time a co-occurrence between them is observed. The rate equals negative two times the **mean sentiment score** of non-zero-score sentences in the whole novel. The mean score gave us an intuitive sense of the sentiment skewness. The negative sign in front adjust the sentiments to an opposite direction of the skewness. Because co-occurrence does not happen in every sentence, which will reduce the adjustment effect on sentiments, we multiply the rate by 2 to offset it to some extent.

Having these two concepts, the formula for calculating the sentiment matrix could be written as below, where ***θ<sub>align</sub>*** represents the sentiment alignment rate, ***V<sub>sentiment</sub>*** represents the sentiment array, ***V<sup> i</sup><sub>sentiment</sub>*** represents the **i** th element of it,  and ***N*** represents the number of elements. Besides, triangularization and diagonal processings are still required.

<p align="center"><img width="400" alt="formula2" src="https://user-images.githubusercontent.com/30411828/55671901-6bb0dd00-58c7-11e9-9563-d70359835cd8.png"></p>
<p align="center"><img width="500" alt="formula3" src="https://user-images.githubusercontent.com/30411828/55671903-6e133700-58c7-11e9-8739-3df8cab2e746.png"></p>
<p align="center"><img width="560" alt="formula6" src="https://user-images.githubusercontent.com/30411828/55672693-6efc9680-58d0-11e9-9683-6029e7bd84e5.png"></p>

## Graph Parameters and Plot

After we have the two matrices, we can now transform them into the graph parameters and then plot the fancy graph out! In this process, the matrices are first **normalized** so as to make the magnitude consistent across different novels while keeping the diversity among characters in one novel. By the way, do notice that the formulas for transforming graph parameters (in functions `matrix_to_edge_list` and `plot_graph`) have **no actual meanings**, just to make the plot look nicer and more aligned by passing proper parameters.
![aa](https://user-images.githubusercontent.com/30411828/55665312-b141bb80-586f-11e9-9751-274f6e5359c9.png)
## Done!
After all the steps above, the work is done! You can now check the generated [**.png files**](https://github.com/hzjken/character-network/tree/master/graphs) in the folder to see the plots.

## Some More Thoughts

1. Based on my personal test, the running of the whole process roughl takes **15 mins** to finish, which is relatively slow for one novel. That’s also the reason why I provide a simple [**Pyspark distributed version**](https://github.com/hzjken/character-network/blob/master/characterNetwork-distributed.py) to parallel and accelerate the process. But I am sure there must be faster solutions out there.

2. This methodology captures some extent of information on the character relationships, but it’s not perfect. There are many parts that
could be improved. For instance, in this example we haven’t considered the use of **personal pronoun**. Therefore, many places where
a character’s name is replaced by “He” or “She” will not be captured, causing a loss of information. It will be even worse if a
novel is written in **first perspective**, where the protagonist’s name is mostly replaced by “I” and “me”. Besides, the whole process can be regarded as ***unsupervised learning***, which from its nature is considered not very accurate. A more rigorous project might
consider re-training the NER and sentiment score classifier or providing more labeled data for machines to learn on character relationships.

3. Though this methodology works reasonably good (from my perspective) for Harry Potter novel series, I am not sure about its performance on other novels as I haven’t had time to validate. If you like this methodology, feel free to apply it on your favorite novels and share with me the results. Of course, **stars** are even more welcomed! &nbsp; **:P**


