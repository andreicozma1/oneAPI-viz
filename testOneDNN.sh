#!/bin/bash

python -c "from tensorflow.python.util import _pywrap_util_port; print('oneDNN optimizations enabled:', _pywrap_util_port.IsMklEnabled())"