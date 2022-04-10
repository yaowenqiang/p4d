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

