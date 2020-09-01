from joblib import Parallel, delayed
import time, math


def my_fun(i):
    """ We define a simple function here.
    """
    time.sleep(1)
    return math.sqrt(i**2)

num = 10
start = time.time()
for i in range(num):
    my_fun(i)
    end = time.time()
    print('{:.4f} s'.format(end-start))




start = time.time()
# n_jobs is the number of parallel jobs
r = Parallel(n_jobs=2)([delayed(my_fun)(0), delayed(my_fun)(1), delayed(my_fun)(2)])
end = time.time()
print('{:.4f} s'.format(end-start))
print(r)
