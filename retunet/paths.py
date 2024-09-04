#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

RetUNet_raw = os.environ.get('RetUNet_raw')
RetUNet_preprocessed = os.environ.get('RetUNet_preprocessed')
RetUNet_results = os.environ.get('RetUNet_results')

if RetUNet_raw is None:
    print("RetUNet_raw is not defined and RetU-Net can only be used on data for which preprocessed files "
          "are already present on your system. RetU-Net cannot be used for experiment planning and preprocessing like "
          "this.")

if RetUNet_preprocessed is None:
    print("RetUNet_preprocessed is not defined and RetU-Net can not be used for preprocessing "
          "or training.")

if RetUNet_results is None:
    print("RetUNet_results is not defined and RetU-Net cannot be used for training or "
          "inference.")
