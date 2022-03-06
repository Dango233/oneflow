/*
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
*/
#include <pybind11/pybind11.h>
#include <string>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/api/python/session/session_api.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("IsSessionInited", &IsSessionInited);
  m.def("InitLazyGlobalSession", &InitLazyGlobalSession);
  m.def("InitEagerGlobalSession", &InitEagerGlobalSession);
  m.def("DestroyLazyGlobalSession", &DestroyLazyGlobalSession);

  m.def("StartLazyGlobalSession", &StartLazyGlobalSession);
  m.def("StopLazyGlobalSession", &StopLazyGlobalSession);

  // multi-client lazy global session context
  m.def("CreateMultiClientSessionContext", &CreateMultiClientSessionContext);
  m.def("InitMultiClientSessionContext", &InitMultiClientSessionContext);
  m.def("MultiClientSessionContextUpdateResource", &MultiClientSessionContextUpdateResource);
  m.def("TryDestroyMultiClientSessionContext", &TryDestroyMultiClientSessionContext);

  using namespace oneflow;
  py::class_<MultiClientSessionContext, std::shared_ptr<MultiClientSessionContext>>(
      m, "SessionContext")
      .def(py::init())
      .def("try_init",
           [](MultiClientSessionContext& session, const std::string& config_proto_str) {
             return session.TryInit(config_proto_str).GetOrThrow();
           })
      .def("update_resource",
           [](MultiClientSessionContext& session, const std::string& reso_proto_str) {
             return session.UpdateResource(reso_proto_str).GetOrThrow();
           });

  m.def("NewSessionId", &NewSessionId);
  py::class_<LogicalConfigProtoContext>(m, "LogicalConfigProtoContext")
      .def(py::init<const std::string&>());
}
