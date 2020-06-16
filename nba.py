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
Clustering is a foundational Machine Learning technique underlying many advances in Artificial Intelligence. In combination with other approaches, clustering has provided a base upon which computational intellgience has been able to surpass human level performance on certain tasks. 

Due to the simplicity and clearly explainable results of it's models, we find clustering not only in research, but more importantly, at the heart of successful ML commerical applications solving real world problems. 

![Spam](https://www.spam.com/wp-content/uploads/2019/09/image-product_spam-classic-12oz-420x420.png)

K Nearest Neighbor is often recognized as one of the most popular forms of clustering for production applications. In K Nearest Neighbor, data is interpreted into K number of like groups using simple math that is still powerful enough to recognize patterns in huge scale and within  production relevant time. 

## 

![Nearest Neighbor](https://dashee87.github.io/images/em_only.gif)')

While Nearest Neighbor, and it's cousin clustering approaches often provide top of the line performance (while also excelling in explainability and maintainability) there are limitations.

As data grows, it becomes increasingly difficult to fit nearest neighbor models into available memory space and demands on processing power begin to outstrip what is feasible for production applications. Named *"The Curse of Dimensonality"*, Nearest Neighbor Kryptonite is data that increases simultaneously in complexity and size.  

#

![Nearest Neighbor meets data it cant match](https://i1.wp.com/badbooksgoodtimes.com/wp-content/uploads/2016/01/kryptonite.gif)

#

To demonstrate data size and complexity, below is a dataframe containing  all NBA team offensive data for the last decade. This totals 331 teams over 21 statistical categories. 
#

# NBA Team Data 2020-2010 (per 100)
##

'''

st.write(nba)

'''
## 
Where clustering runs into production problem is when we attempt to increase the number of categories we are examining while also increasing the number of examples. If we were to increase our dataset to every team that has ever played basketball, while also increasing the number of statistics we believed contributed to a "neighborhood", we would eventually run out of computation memory and/or computational power.

If we desired to include team data for every year moving forward our model would not be able to set a limit on the 'Y axis' of the above data. As the NBA has 30 new team examples each season - the Y axis must be allowed to grow towards infinity. 

![Axis](https://www.eduplace.com/math/mathsteps/artwork/4_coord_what1.gif)

Instead it is the 'X axis' that must be limited. The X-axis of the above NBA data can be thought of as the "Dimensions" or "Features." In this dataset there are only 21 Dimensions but if we were to try to grow this data into the 1000's or millions, our data space would be untenable.

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

Another interesting insight is that each team taking the most 3PA made the playoffs each year of the decade (* next to team name indicates made playoffs). Unexpectedly however, teams at the other end of the spectrum, outliers attempting the least number of 3's, seperating themselves from the pack, also made the playoffs quite often. It could be said that teams with a strong identity around how they managed their 3PA were the most successful during the regular season.    

# Approximate Nearest Neighbor

 But what if we didnt want to discover insight from a single dimension but all dimensions? How could we run such a model given the memory/performance limitaitons of nearest neighbor? 

Approximate Nearest Neighbor models allow us to overcome production deficiencies in cluster models by reducing accuracy of neighbor search but increasing speed and data size.

Studies have repeatedly shown that the most performant Approximate Nearest Neighbor model is Annoy, an open source clustering library. One such study stated:

> We also recommend Annoy, due to its excellent search performance, and robustness to the datasets.  Additionally, a nice property of Annoy is that, comparedwith proximity graph based approaches, it can provide better trade-off between search performance and index size/construction time. This is because, one can reduce the number of trees without hurting the searchperformance substantially.

Made in 2015 in a weekend hackathon, much of Spotify's playlist reccomendation success has been in building a CI/CD production pipeline and slowly building up ML capacity upon Annoy's solid foundation. 
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

st.title(' 2020 L.A. Lakers Neighbors')

nba.iloc[[14,65,17,78,63, 37,45,120,8,66,127,32]]
''' 
#
That the 2019 Champion Toronto Raptors, 2016 Warriors that lost the finals in game 7 and won 73 games in the regular season, and 2016 Champion Cleveland Cavs make this list is a sign you want to do whatever this seasons L.A. Lakers do. 

We can tell our nearest neighbor search is giving us accurate information as it returns as nearest neighbor, two teams that previously featured Lebron James (2016 Cavs, 2019 Lakers) and one that featured current Lakers star Anthony Davis (2019 Pelicans). That the Nearest Neighbor search can identify this with only team data, demonstrates the immense power this algorithm has. 
'''

st.title('2020 Oklahoma City Thunder Neighbors')

nba.iloc[[20,95,68,97,16,35,33,126,157,14,117,151]]
'''
The accuracy of what the model is capable of finding is quite astounding: Of the 11 nearest neighbor teams, four of them are the 2015, 2016, 2017, in which Thunder Star Chris Paul was the starting Point Guard and the 2018 Clippers, attempting to continue to play the same style without Chris Paul. Comparables for the Thunder, include mostly playoff teams which bodes well for their first round Playoff prospects.
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
u.get_nns_by_item(i,12)
'''
st.code(code, language='python')

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
