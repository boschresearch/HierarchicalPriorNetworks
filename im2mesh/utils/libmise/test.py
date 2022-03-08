# This source code is from Autonomous Vision - Occupancy Networks
#   (https://github.com/autonomousvision/occupancy_networks)
# Copyright 2019 Lars Mescheder, Michael Oechsle, Michael Niemeyer, Andreas Geiger, Sebastian Nowozin
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
import numpy as np
from mise import MISE
import time

t0 = time.time()
extractor = MISE(1, 2, 0.)

p = extractor.query()
i = 0

while p.shape[0] != 0:
    print(i)
    print(p)
    v = 2 * (p.sum(axis=-1) > 2).astype(np.float64) - 1
    extractor.update(p, v)
    p = extractor.query()
    i += 1
    if (i >= 8):
        break

print(extractor.to_dense())
# p, v = extractor.get_points()
# print(p)
# print(v)
print('Total time: %f' % (time.time() - t0))
