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

As data grows, it becomes increasingly difficult to fit nearest neighbor models into memory space and processing power beings to falter in production. Named *"The Curse of Dimensonality"*, a fatal flaw for Nearest Neighbor is data growing simultaneously in increased complexity and 
universal application of clustering for every problem is not possible. The limiations of memory space and processing power make it impossible to cluster everywhere. 
'''

'''
# NBA Team Data 2020-2010 (per 100)
##



Here we have the team data for all NBA teams for the last decade. This totals 331 teams over 21 statistical categories
##
'''

st.write(nba)

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

