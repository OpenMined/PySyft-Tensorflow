# PySyft-TensorFlow
TensorFlow bindings for [PySyft](https://github.com/openmined/pysyft).

PySyft is a Python framework for secure, private deep learning.  PySyft-TensorFlow brings
secure, private deep learning to [TensorFlow](https://tensorflow.org).

[![Build Status](https://travis-ci.org/OpenMined/PySyft-TensorFlow.svg?branch=master)](https://travis-ci.org/OpenMined/PySyft-TensorFlow)
[![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft)

## Installation

PySyft-TensorFlow is available on pip

```
pip install syft-tensorflow
```

NOTE: We aren't yet on a proper release schedule. Until then, we recommend building the code from source. The master branch is intended to be kept in line with [this branch](https://github.com/dropoutlabs/PySyft/tree/dev) on the [DropoutLabs](https://github.com/dropoutlabs/PySyft) fork of PySyft. If you have any trouble, please open an issue or reach out on Slack via the #team_tensorflow or #team_pysyft channels.

## Usage

See the [PySyft tutorials](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials)
if you are unfamiliar with any Syft paradigms.

```python
import tensorflow as tf
import syft

hook = sy.TensorFlowHook(tf)
# Simulates a remote worker (ie another computer)
remote = sy.VirtualWorker(hook, id="remote")

# Send data to the other worker
x = tf.constant(5).send(remote)
y = tf.constant(10).send(remote)

z = x * y

print(z.get())
# => 50
```

## Developing PySyft-TensorFlow

See [CONTRIBUTING](./CONTRIBUTING.md).

## Project Support

PySyft-Tensorflow was contributed by and continues to be maintained by the team at [Dropout Labs](https://dropoutlabs.com).

Please reach out to contact@dropoutlabs.com for support.

[<img src="https://raw.githubusercontent.com/dropoutlabs/files/master/dropout-labs-logo-white-2500.png" alt="Dropout Labs" width="200"/>](https://dropoutlabs.com/)
|--------------------------------------------------------------|
