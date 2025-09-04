import time
import os
import psutil

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def memory_profile(func):
    def wrapper(*args, **kwargs):

        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))

        return result
    return wrapper


def time_profile(func):
    """
    Wrapper function for runtime check
    """
    
    def wrapper(*args, **kwargs):
        nonlocal total
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        total += duration
        print(f"Execution time: {duration}   Total: {total}")
        return result

    total = 0
    return wrapper

#test functions. Don't use it 
@time_profile
def _func():
    x = [1] * (10 ** 7)
    y = [2] * (4 * 10 ** 2)
    del x
    
@memory_profile
def _func():
    x = [1] * (10 ** 7)
    y = [2] * (4 * 10 ** 2)
    del x
    