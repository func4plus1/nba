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

K-Means Nearest Neighbor is often recognized as one of the most popular form of clustering for production application. Data is interpreted into K number of like groups allowing discovery of patterns within huge scale in production relevant time. 

## 

![Nearest Neighbor](https://dashee87.github.io/images/em_only.gif)')

While Nearest Neighbor and it's cousin clustering approaches often provide top of the line performance, while also excelling in explainability and maintainability, there are limitations.

As data grows, it becomes increasingly difficult to fit nearest neighbor models into memory space and processing power beings to falter in production. Named *"The Curse of Dimensonality"*, the fatal flaw is data that increases  simultaneously in complexity and size.  


To demonstrate data size and complexity, below is a dataframe containing  all NBA team offensive data for the last decade. This totals 331 teams over 21 statistical categories. 

# NBA Team Data 2020-2010 (per 100)
##

'''

st.write(nba)

'''
## 
Where clustering runs into production problem is when we attempt to increase the number of categories we are examining while also increasing the number of teams in the dataset. If we were to increase our dataset to every team that has ever played basketball, while also increasing the number of statistics we believed contributed to a "neighborhood", we would eventually run out of computation memory and/or computational power does not currently exist that would be able to solve the equation in reasonable time.

If we desired to include team data for every year moving forward our model would not be able to set a limit on the 'Y axis' of the above data. As the NBA has 30 new team examples each season - the Y axis must be allowed to grow towards infinity. 

![Axis](https://www.eduplace.com/math/mathsteps/artwork/4_coord_what1.gif)

Instead it is the 'X axis' that must be limited. The X-axis of the above NBA data can be thought of as the "Dimensions" or "Features." In this dataset there are only 21 Dimensions but if we were to try to grow this data into the 1000's or millions, our data space would be untenable.

Rather than growing dimensions to infinity, we must instead discover the Dimensions that provide insight into our data and thus allow it to be clustered into nearest neighbors. This activity is called "Dimensionality Reduction" and it is a core principal of much Machine Learning used in production applications.

If we can discover which are the only dimensions we need to create distinct and accurate neighbors, we discover a model that is capable of running in production quickly, cheaply and accurately.

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

