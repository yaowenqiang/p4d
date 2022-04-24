# Libraries

+ NumPy
+ SciKit-Learn
+ SciPy
+ MatplotLib
+ Pandas
+ Plotly
+ Seaborn
+ PySpark

# Conda environments

> https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html
> conda create --name myenv biopython
> activate myenv
> deactivate myenv
> conda create --name myenv numpy
> conda create --name myenv numpy python=2.7
> conda create --name myenv numpy python=2.7 anaconda
> condas info --envs
> 
# NumPy

> conda install numpy

> my_list = [1,2,3]
> import numpy as np
> np.array(my_list)
> my_mat = [[1,2,3], [4,56],[7,8.9]]

>> np.aranage(0, 10)
> np.zeros(10)
> np.zeros(12,3)
> np.linspace(12,5, 100)
> np.eye(4)
>np.random.rand(5)
>np.random.rand(5, 5)
>np.random.ranmd(2)
>np.random.ranmd(5,5)
>np.random.ranmdint(5,100)
>np.random.ranmdint(5,100, 10)
> arr = np.arange(25)
> ranarr = np.random.randint(0, 50, 10)
> arr.reshap(5,5)
> randarr.max()
> randarr.min()
> randarr.argxmax()
> randarr.argxmin()
> randarr.shap
> randarr.dhap
> arr[0:5] = 100
> arr_copy = arr.copy()
> arr_2d = [[1,1,2],[4,5,6], [7,8.9]]
> arr_2d[0][0]
> arr_2d[0,10]
> arr_2d[:2,11:]
> boolean_arr = arr > 5
> arr[boolean_arr]
> arr[arr>5]

> arr + arr
> arr - arr
> arr * arr
> arr / arr
> arr + 100
> arr ** 2

# numpy Universal functions(ufunc)

> np.sqrt(arr)
> np.exp(arr)
> np.max(arr)
> np.sin(arr)
> np.cos(arr)
> np.log(arr)

# pandas
> pip install pandas


## series

> import numpy as np
> import pandas as pd
>  my_data  = [10,20,30]
> labels = [a', 'b', 'c']
> pd.Series(data = my_data)
> pd.Series(data = my_data, index=labels)
> pd.Series(my_data, labels)

## dataFrames

> from numpy.random import randn
> np.random.seed(101)
> df = pd.DataFrame(randn(5,4), ['A', 'B', 'C', 'D', 'E'], ['W','X', 'Y', 'Z'])
> df['W']
> df.W
> df[['W','X']]
> df['new'] = df['W'] + df['Y']

> df.drop('new', axis=1)
> df.drop('new', axis=1, inplace=True)
> df.drop('E',axis=0)

## Select rows

> df.loc['A']
> df.iloc[1]
> df.loc['B', 'Y']
> df.loc[['A', 'B'], 'Y']
> booldf = > df > 0
> df[booldf]
> >df[df > 0]
> df['W'] > 0
> df[df]['W'] > 0]
> df[df]['W'] > 0]['X']
> df[(df['W']> 0) & (df['Y'] > 1)]
> df[(df['W']> 0) | (df['Y'] > 1)]
> df.reset_index()
> newind = 'CA NY WY OR CO'.split()
> df['stats'] = newwind
> df.set_index('stats')
> df.set_index('stats', inplace=True)
> 

## multi index

> outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
> inside = [1,2,3,1,2,3]
> hier_index = list(zip(outside ,inside))
> hier_index = pd.MultiIndex.from_tuples(hier_index)
> df = pd.DataFrame(randn(6,2), hier_index, ['A', 'B'])
> df.loc['G1']
> df.loc['G1'].loc[1]
> df.index.names = ['Group', 'Num']
> df.loc['G2'].loc[2]['B']
> df.xs('G1')
> df.xs(1, levle='Num)


## Data input and ouput

+ CSV
+ HTML
+ sql
+ Excel

> conda install sqlalchemy
> conda install lxml
> conda install html5lib
> conda install BeautifulSoup4
> conda install xlrd

> import pandas as pd
> df = pd.read_csv('example.csv')
> df.to_csv('new', index=False)
> pd.read_excel('my_excel.elsx', sheetname='Sheet1')
> df.to_excel('new.xlsx', sheet_name='Sheet2')

> data = pd.read_html('url')
> data[0].head()

>> from sqlalchemy import create_engine
> engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)

> df.read_sql('my__table', engine)

## Missing Data

> d = {'A': [1,2, np.nan], 'B': [5,np.nan, np.nan], 'C':[1,2,3]}
> df = pd.DataFrame(d)
> df.dropna()
> df.dropna(axis=1)
> df.dropna(axis=1, thresh=2)
> df.fillna(value='FILL VALUE')
> df['A'].fillna(value=df['A'].mean())


## Data Group

'''
data = {
        'Company': ['GOOG','GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
        'Persion': ['Sam', 'Charlie', 'Amy', 'vanessa', 'Carl', 'Sarah'], 
        'Sales': [200, 120, 340, 124, 243, 350]
}

df = pd.DataFrame(data)

byCom = df.groupby('Company')
> byCom.mean()
> byCom.std()
> byCom.sum()
> byCom.count()
> byCom.max()
> byCom.min()

df.groupby('Company').sum().loc['FB']
df.groupby('Company').describe()
df.groupby('Company').describe().transpose()
df.groupby('Company').describe().transpose()['FB']
''''''

## Merging, Joinng, and Concatenating

> df1 = pd.DataFrame({
	"A": ['A0', 'A1', 'A2','A3'],	
	"B": ['B0', 'B1', 'B2','B3'],	
	"C": ['C0', 'C1', 'C2','C3'],	
	"D": ['D0', 'D1', 'D2','D3'],	
	}, index=[0,1,2,3])
> df2 = pd.DataFrame({
	"A": ['A4', 'A5', 'A6','A7'],	
	"B": ['B4', 'B5', 'B6','B7'],	
	"C": ['C4', 'C5', 'C6','C7'],	
	"D": ['D4', 'D5', 'D6','D7'],	
	}, index=[4,5,6,7])
> df3 = pd.DataFrame({
	"A": ['A8', 'A9', 'A10','A11'],	
	"B": ['B8', 'B9', 'B10','B11'],	
	"C": ['C8', 'C9', 'C10','C11'],	
	"D": ['D8', 'D9', 'D10','D11'],	
	}, index=[8,9,10,11])
> pd.concat([df1, df2, df3])
> pd.concat([df1, df2, df3],axis=1)


	left = pd.DataFrame({
	"KEY": ['K0', 'K1', 'K2','K3'],
	"A": ['A8', 'A9', 'A10','A11'],
	"B": ['B8', 'B9', 'B10','B11'],
	})
	right = pd.DataFrame({
	"KEY": ['K0', 'K1', 'K2','K3'],
	"C": ['C4', 'C5', 'C6','C7'],
	"D": ['D4', 'D5', 'D6','D7'],
	})

> pd.merge(left, right, how='inner', on='key')
> 
> lef.join(right)



## Operations

> df.head
> df['col2'].unique()
> df['col2'].nunique()
> df['col2'].value_counts()
> df[df['col2']>2]
> df[(df['col2']>2) | (df['col2']<10]

> def times2(x):
	return x *2

> df['col1'].apply(times2)
> df['col1'].apply(len)
> df['col1'].apply(lambda x: x*2)
> df.drop('col1', axis=1)
> df.drop('col1', axis=1, inplace=True)
> df.columns
> df.index
> df.sort_values('col2')
> df.sort_values(by ='col2')
> df.isnull()


''''''
     data = {
        'A':['foo','foo','foo','bar','bar','bar'],
        'B':['one','one','one','two','two','two'],
        'C':['x','x','x','y','y','y'],
        'D':[1,3,2,5,4,1]
        
df = pd.DataFrame(data)
df.pivot_table(values='d',index=['A', 'B'], columns=['C'])

''''''


# Matplotlib

> pip install matplotlib

'''
	import matplotlib.pyplot as plt
    %matplotlib inline
    plt.show()

> import numpy as np

x = lispace(0, 5, 11)
y = x ** 2
plt.plot(x, y,)
plt.xlable('x label')
plt.ylable('y label')
plt.title('title)
plt.plot(x, y, 'r-',)
'''

