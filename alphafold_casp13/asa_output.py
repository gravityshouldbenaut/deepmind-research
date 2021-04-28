# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for predicting Accessible Surface Area."""

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


class ASAOutputLayer(object):
  """An output layer to predict Accessible Surface Area."""

  def __init__(self, name='asa'):
    self.name = name

  def compute_asa_output(self, activations):
    """Just compute the logits and outputs given activations."""
    asa_logits = tf.contrib.layers.linear(
        activations, 1,
        weights_initializer=tf.random_uniform_initializer(-0.01, 0.01), #generates tensors with a uniform distribution from -0.01 to 0.01 https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer
        scope='ASALogits') #essentially creates a neural net wherein the activations are the inputs, there is 1 output, the weights are randomized from -0.01 to 0.01, and a scope name is assigned
      #same neural net is ran below, looking for the positive results after applications of weights 
      
    self.asa_output = tf.nn.relu(asa_logits, name='ASA_output_relu') #computes the rectified linear https://www.tensorflow.org/api_docs/python/tf/nn/relu, which is only the positive part of the tensor https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    return asa_logits
