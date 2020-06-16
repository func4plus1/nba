import pandas as pd
import annoy 
from annoy import AnnoyIndex
import streamlit as st
import numpy as np 
nba = pd.read_csv('nba.csv') 

df = pd.read_csv('nba.csv')

df = df.drop(columns=['G','MP', 'Y', 'Team'])

df.reset_index(inplace=True)

df = df.drop(columns=('index'))

df = df.T

'''
# Clustering 
##
[Clustering](https://machinelearningmastery.com/clustering-algorithms-with-python/) is a foundational [Machine Learning](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/) technique underlying many advances in [Artificial Intelligence](https://www.csee.umbc.edu/courses/471/papers/turing.pdf). In combination with other approaches, clustering has provided a base upon which computational intellgience has been able to surpass human level performance on certain tasks. 

Due to the simplicity and clearly explainable results of it's models, we find clustering not only in research, but more importantly, at the heart of successful ML commerical applications solving real world problems. 

![Spam](https://www.spam.com/wp-content/uploads/2019/09/image-product_spam-classic-12oz-420x420.png)

[K Nearest Neighbor](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/) is often recognized as one of the most popular forms of clustering for production applications. In K Nearest Neighbor, data is interpreted into K number of like groups using simple math that is still powerful enough to recognize patterns in huge scale and within  production relevant time. 

## 

![Nearest Neighbor](https://dashee87.github.io/images/em_only.gif)')

While Nearest Neighbor, and it's cousin clustering approaches often provide top of the line performance (while also excelling in explainability and maintainability) there are limitations.

> ["As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially."](https://www.kdnuggets.com/2017/04/must-know-curse-dimensionality.html)

As data grows, it becomes increasingly difficult to fit nearest neighbor models into available memory space and demands on processing power begin to outstrip what is feasible for production applications. 


![Nearest Neighbor meets data it cant match](https://i1.wp.com/badbooksgoodtimes.com/wp-content/uploads/2016/01/kryptonite.gif)


Named [*"The Curse of Dimensonality"*](https://www.nature.com/articles/s41592-018-0019-x), Nearest Neighbor Kryptonite is data that increases simultaneously in complexity and size.  



# NBA Team Data 2020-2010 (per 100)
##

Where clustering goes wrong is when we attempt to increase the number of categories we are examining while also increasing the number of examples. When increasing dimensions we need to increase input data exponentially. 

To demonstrate data size and complexity, below is a dataframe containing  all NBA team offensive data for the last decade. This totals 331 teams over 21 statistical categories. 
#

'''

st.write(nba)

'''
## 

If we were to increase our dataset to every team that has ever played basketball, while also increasing the number of statistics we believed contributed to a "neighborhood", we would eventually run out of computation memory and/or computational power.

If we desired to include team data for every year moving forward our model would not be able to set a limit on the 'Y axis' of the above data. As the NBA has 30 new team examples each season - the Y axis must be allowed to grow towards infinity. 

![Axis](https://www.eduplace.com/math/mathsteps/artwork/4_coord_what1.gif) ![To Infinity and Beyond](https://media.tenor.com/images/467a0553795c6c013edf2402b9869ec7/tenor.gif)

Instead it is the 'X axis' that must be limited. The X-axis of the above NBA data can be thought of as the "Dimensions" or "Features." In this dataset there are only 21 Dimensions but if we were to try to grow complexity into the 1000's or millions, our data space would be untenable.

Rather than growing dimensions to infinity, we must instead discover the Dimensions that provide insight into our data and thus allow it to be clustered into nearest neighbors. This activity is called ["Dimensionality Reduction"](https://en.wikipedia.org/wiki/Dimensionality_reduction) and it is a core principal of much Machine Learning used in production applications.

If we can discover which are the *only* dimensions we need to create distinct and accurate neighbors, we arrive at a model that is capable of running in production quickly, cheaply and accurately.

Though there are many mathmatical techniques for dimensionality reduction, as well as state-of-the-art ML techniques for "feature extraction", it is useful to see how a domain expert might observe data patterns to extract meaning.

Below we look at a single statistic from the NBA data, 3 Point Attempts (3PA), grouping the teams by year.
##
'''

st.vega_lite_chart(nba,{
    'width': 600,
    'height': 400,
    'mark':{'type':'circle', 'tooltip':True},
    'encoding':{
        'color': {'field': 'Team', 'type': 'nominal', 'legend': None},
        'y': {'field': '3PA', 'type': 'quantitative', 'title': '3 Point Attempts'},
        'x': {'field': 'Y', 'type': 'nominal', 'title': 'Year'},
    },
})

'''
An NBA fan can look at this chart and instantly recognize the shift in the way the game has been played over the last ten years. This chart shows us that there has been a year-over year increase in number of 3PA all teams in the league have made. 

This season, the team attempting the least number of 3's appears to have still have attempted double the number of 3's of the lowest 3PA team of 2010 (Memphis Grizzlies). A team in 2020 managing their 3 point shooting the way a team did just 10 years ago, would be entirely out of line with their peers and the market in general. 

Another interesting insight is that every team taking the most 3PA made the playoffs each year of the decade (* next to team name indicates made playoffs). Unexpectedly however, teams at the other end of the spectrum, outliers attempting the least number of 3's, seperating themselves from the pack, also made the playoffs quite often. It could be said that teams with a strong identity around how they managed their 3PA were the most successful during the regular season.    

# Approximate Nearest Neighbor

 But what if we didnt want to discover insight from a single dimension but all dimensions? How could we run such a model given the memory/performance limitaitons of nearest neighbor? 

Approximate Nearest Neighbor models allow us to overcome production deficiencies in cluster models by reducing accuracy of neighbor search but increasing speed and data size.

Studies have repeatedly shown that the most performant Approximate Nearest Neighbor model is [Annoy](https://github.com/spotify/annoy), an open source clustering library. One such study stated:

> We also recommend Annoy, due to its excellent search performance, and robustness to the datasets.  Additionally, a nice property of Annoy is that, compared with proximity graph based approaches, it can provide better trade-off between search performance and index size/construction time. This is because, one can reduce the number of trees without hurting the search performance substantially.

Made in 2015 in a weekend hackathon, much of Spotify's playlist reccomendation success has been in building a CI/CD production pipeline and slowly building up ML capacity upon Annoy's solid foundation. Through optimized memory management and intelligent data indexing Annoy has been capable of achieving clustering in production once thought impossible. Within the last decade, it has been thought that having even 10 dimensions triggered the "Curse."   

Below we run Annoy on the 21 Dimensions of NBA data over 331 examples and get 30 nearest neighbor analysis in the amount of time it took to load this page:
'''

f = 21
t = AnnoyIndex(f, 'angular') 

for i in range(330):
  v = df[i]
  t.add_item(i,v)

t.build(10)
t.save('test.ann')

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file

for i in range(30):
    list = [] 
    list.append(u.get_nns_by_item(i,12))
    st.write(nba.iloc[list[0]])

'''
#
While 331 examples is a tiny dataset, used for demonstration purposes, Annoy is performant at far larger data.  Annoy scales better than any other clustering library, especially  well suited to providing higher precision without losing speed as Spotify's recommendation engine can attest.

![Annoy comparisons](https://github.com/erikbern/ann-benchmarks/raw/master/results/glove-100-angular.png)

If we expland Nvidia's [PLASTER framework](https://blogs.nvidia.com/blog/2018/07/11/how-to-measure-deep-learning-performance/) to include the evaluation of Machine Learning models in general, we see where Annoy really shines.

Nvidia describes PLASTER as:

> **Programmability**: Models must be coded, trained and then optimized for a specific runtime inference environment. 

>**Latency**: The time between requesting something and receiving a response is critical to the quality of a service. With most human-facing software systems, not just AI, the time is often measured in milliseconds.

> **Accuracy**: Models need to make the right predictions. Options to address data volumes have been either to transmit the full information with long delays or to sample the data and reconstruct it using techniques that can lead to inaccurate reconstruction and diagnostics.

> **Size of model**: The computing approach needs to be support and efficiently process large... models.

> **Throughput**: Hyperscale data centers require massive investments of capital. Justifying a return on this requires understanding how many inferences can be delivered within the latency targets. Or, put another way, how many applications and users can be supported by the data center.

> **Energy efficiency**: Power consumption can quickly increase the costs of delivering a service, driving a need to focus on energy efficiency in devices and systems.

> **Rate of learning**: Models are dynamic, involving training, deployment and retraining. Understanding how and how fast models can be trained, re-trained and deployed as new data arrives helps define success.

If we look at the entirety of the code that ran our 30 nearest neighbor comparisons over 21 dimensions instantaneously we can understand why Annoy is such an exemplary approach to clustering for production application.
'''

code = '''f = 21
t = AnnoyIndex(f, 'angular') 

for i in range(330):
  v = df[i]
  t.add_item(i,v)

t.build(10)
t.save('test.ann')

u = AnnoyIndex(f, 'angular')
u.load('test.ann') 

for i in range(30):
    u.get_nns_by_item(i,12)
'''

st.code(code, language='python')

'''
Compared to other Machine Learning approaches, often 1000's of lines of code, we get the power of almost state-of-the-art clustering in 15 lines of code. Code that has been battle-tested by years of stability in production is worth it's weight in gold. If your data can fit within Annoy, what you see is what you get - 15 lines of code built upon a library that has been battle tested by years of stability in production. From a PLASTER mindset this is worth it's weight in gold:

* **Programmability** - as you can see - super programmable
* **Latency** - the statistics show that Annoy has the lowest latency
* **Accuracy** - good enough is what we are looking for in-production. Annoy gives us good enough accuracy while excelling in all the other metrics of ML performance
* **Size of the Model** - small model, big insight
* **Throughput** - super high throughput, excellent indexing approach
* **Energy Efficiency** - amount of cycles required far less than other approaches
* **Rate of Learning** - Give it new data - it gives you new, more accurate answers (to a certain point)

Annoy excels in the metrics of production ML systems, but does Nearest Neighbor clustering generate insight? 
What is the so what? of this model? By looking at the NBA data it is amazing the deep knoweldge Annoy nearest neighbor can uncover about NBA teams in an instant.



Looking at just the few below examples it is important to remember we have given this clustering nothing but team data. Yet it infers wisdom about teams that seems incomprehensible given the input. A human knowing nothing about basketball could not possibly derive such insightful comparison of teams.
'''

st.title(' 2020 L.A. Lakers Neighbors')

nba.iloc[[14,65,17,78,63, 37,45,120,8,66,127,32]]
''' 
#
That the 2019 Champion Toronto Raptors, 2016 Warriors that lost the finals in game 7 and won 73 games in the regular season, and 2016 Champion Cleveland Cavs make this list is a sign you want to do whatever this seasons L.A. Lakers do. 

Remember we have given this clustering nothing but team data and yet it infers wisdom about teams that seems incomprehensible given the input. A human knowing nothing about basketball could not possibly derive such insightful comparison

We can tell our nearest neighbor search is giving us accurate information as it returns as nearest neighbor, two teams that previously featured Lebron James (2016 Cavs, 2019 Lakers) and one that featured current Lakers star Anthony Davis (2019 Pelicans). That the Nearest Neighbor search can identify this with only team data, demonstrates the immense power this algorithm has. 
'''

st.title('2020 Oklahoma City Thunder Neighbors')

nba.iloc[[20,95,68,97,16,35,33,126,157,14,117,151]]
'''
The accuracy of what the model is capable of finding is quite astounding: Of the 11 nearest neighbor teams, four of them are the 2015, 2016, 2017, in which Thunder Star Chris Paul was the starting Point Guard and the 2018 Clippers, attempting to continue to play the same style without Chris Paul. 

Not having previous Thunder teams as nearest neighbor is also a proof of insight, as those teams featured Russell Westbrook and Paul George, players no longer on the team. 

Comparables for the Thunder, include mostly playoff teams which bodes well for their first round Playoff prospects.
'''
st.title('2020 Houston Rockets Neighbors')
nba.iloc[[2,61,40,91,0,11,50,13,10,19,93,44]]
'''
James Harden
'''
'''
## Appendix 

### How is Nearest Neighbor calculated via Euclidian Distance?
####

$$d(p,q) = \sqrt{(q_1-p_1)^2 + (q_2-p_2)^2 + \cdots + (q_n-p_n)^2}$$

or 

$$\sqrt{\sum_{i=1}^n (q_i-p_i)^2}$$

---

###

Team | 3PA | Blocks
---| --- | ---
Toronto Raptors | 4 | 0
Brooklyn Nets | 6 | 6

###

$$d(p,q) =  \sqrt{(4-6)^2 + (0-6)^2}$$

$$\sqrt{(-2)^2 + (-6)^2}$$

$$\sqrt{(40)}$$

The Euclidian Distance Between the Raptors and Nets =    6.3

----

Team | 3PA | Blocks
---| --- | ---
Toronto Raptors | 4 | 0
Miami Heat | 3 | 2

###

$$d(p,q) = \sqrt{(4-3)^2 + (0-2)^2}$$

$$\sqrt{(1)^2 + (-2)^2}$$

$$\sqrt{(5)}$$

The Euclidian Distance Between the Raptors and Heat = 2.2

---

Team | 3PA | Blocks | Distance
---| --- | --- | --- |
Toronto Raptors | 4 | 0 | 0
Brooklyn Nets | 3 | 2 | 6.3

###

Team | 3PA | Blocks| Distance
---| --- | --- | --- |
Toronto Raptors | 4 | 0 | 0
Miami Heat | 3 | 2 | 2.2

###
The Miami Heat are the Toronto Raptors Nearest Neighbor

'''
