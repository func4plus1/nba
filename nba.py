import pandas as pd
import annoy 
from annoy import AnnoyIndex

df = pd.read_csv('nba.csv')

df = df.drop(columns=['G','MP', 'Y', 'Team'])

df.reset_index(inplace=True)

df = df.drop(columns=('index'))

df = df.T


f = 21
t = AnnoyIndex(f, 'angular') 

for i in range(330):
  v = df[i]
  t.add_item(i,v)

t.build(10)
t.save('test.ann')

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 12)) # will find the 1000 nearest neighbors
