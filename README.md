# entropy-mdlp

Minimum description length principle algorithm in python, for optimal binning of continuous variables. 
A fork off of https://github.com/maxpumperla/entropy-mdlp, but essentially a rewrite in numba to address 
the poor speed. This package is a port of the respective R package of the same name.


Install with `pip install https://github.com/el-hult/entropy-mdlp`

Run code like
```python
import entropymdlp as mdlp
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
x_binned = np.digitize(x, mdlp.cut_points(x, y))
print(x_binned)
```

The tests are simply run as `python test.py`.

The algorithm is due to `Fayyad, Usama, and Keki Irani. "Multi-interval discretization of continuous-valued attributes for classification learning." (1993).`