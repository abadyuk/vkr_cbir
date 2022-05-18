import time
import datetime
import hashlib
import numpy as np
import torch


def show_progress(func, iterable, **kwargs):
    results = []
    times = []
    total = len(iterable)
    for i, item in enumerate(iterable):
        start = time.time()
        results.extend(func(item, **kwargs))
        times.append(time.time() - start)
        avg = np.mean(times)
        eta = avg * total - avg * (i + 1)
        eta = datetime.timedelta(seconds=eta)
        print("Progress %d/%d - ETA: %s" % (i + 1, total, eta), end="\r")
    return results


def get_image_id(array):
    return hashlib.sha1(array).hexdigest()


