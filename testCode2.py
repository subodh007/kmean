import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./custdata.csv',sep=',')

#Make a copy of DF
df_tr = df

#Transsform the timeOfDay to dummies
#df_tr = pd.get_dummies(df_tr, columns=['timeOfDay'])

#Standardize
clmns = ['customer_id', 'basket_id','item_id']
df_tr_std = stats.zscore(df_tr[clmns])

#Cluster the data
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr_std)
labels = kmeans.labels_

#Glue back to originaal data
df_tr['clusters'] = labels

#Add the column into our list
clmns.extend(['clusters'])

#Lets analyze the clusters
print df_tr[clmns].groupby(['clusters']).mean()

#Scatter plot of Wattage and Duration
sns.lmplot('customer_id', 'item_id', 
           data=df_tr, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('customer_id vs item_id')
plt.xlabel('customer_id')
plt.ylabel('item_id')
plt.show()