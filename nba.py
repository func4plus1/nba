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

st.markdown('## NBA Team Data 2020 - 2010')
st.text('Per 100 Possesions')

st.write(nba)

option = st.selectbox(
    'Which team  do you like best?',
     nba['Team'])

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
st.write(u.get_nns_by_item(0, 12)) # will find the 1000 nearest neighbors


for i in range(30):
   st.write( u.get_nns_by_item(i,12))

