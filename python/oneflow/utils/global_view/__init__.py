"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow
from oneflow.utils.global_view.to_global import to_global
from oneflow.utils.global_view.to_local import to_local
from oneflow._oneflow_internal.global_view import global_mode, current_is_enabled, current_sbp, current_placement

__all__ = [
    "to_global",
    "to_local",
    "global_mode",
    "current_is_enabled",
    "current_sbp",
    "current_placement",
]
