from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

a = np.random.randint(1,10,size=(1,3,3,1))
aa = tf.constant([[1,1],[1,1]], dtype=tf.float32)

print(aa)