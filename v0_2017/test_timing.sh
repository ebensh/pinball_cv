set -ex

#setup='import numpy as np; import cv2; x = np.random.random_sample((10, 1080, 608))'
#setup='import numpy as np; import cv2; x = cv2.convertScaleAbs(np.random.random_sample((10, 1080, 608)))'
setup='import numpy as np; import cv2; x = cv2.convertScaleAbs(np.random.random_sample((10, 320, 240)))'
echo $setup

# http://blog.hackerearth.com/descriptive-statistics-with-Python-NumPy

# Minimum/maximum (fast)
python -m timeit -n 100 -s "$setup" 'np.amin(x, axis=0)'

# Mean (Pretty quick)
python -m timeit -n 100 -s "$setup" 'np.mean(x, axis=0)'

# Variance (Not very fast)
python -m timeit -n 100 -s "$setup" 'np.var(x, axis=0)'

# Median (SLOWWW)
python -m timeit -n 100 -s "$setup" 'np.median(x, axis=0)'
