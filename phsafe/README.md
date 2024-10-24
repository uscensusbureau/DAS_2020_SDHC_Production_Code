SPDX-License-Identifier: Apache-2.0
Copyright 2024 Tumult Labs

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

# PHSafe
This repository contains PHSafe and its supporting Tumult-developed libraries. For instructions on running PHSafe, see its [README](phsafe/README.md).

## Contents

In the repository there are four folders, each of which contains a component of the release:

- **Core 0.12.0**: A Python library for performing differentially private computations. The design of Tumult Core is based on the design proposed in the [OpenDP White Paper](https://projects.iq.harvard.edu/files/opendp/files/opendp_programming_framework_11may2020_1_01.pdf), and can automatically verify the privacy properties of algorithms constructed from Tumult Core components. Tumult Core is scalable, includes a wide variety of components to handle various query types, and supports multiple privacy definitions. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/core/v0.12/.
- **Analytics 0.8.3**: A Python library for privately answering statistical queries on tabular data, implemented using Tumult Core. It is built on PySpark, allowing it to scale to large datasets. Its privacy guarantees are based on differential privacy, a technique that perturbs statistics to provably protect the data of individuals in the original dataset. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/analytics/v0.8/.
- **Common 0.8.7**: A Python library with utilities for reading and validating data. Code in Common is designed not to be specific to Census applications.
- **PHSafe 3.0.0**: The main program of this release. It produces the Supplemental Demographic and Housing Characteristics File (S-DHC).

PHSafe also requires a CEF reader module for reading data from Census' file formats. The CEF reader is implemented separately, and is therefore not included in this release. We do include a built-in CSV reader for PHSafe, which can be used for PHSafe if the input files are in a CSV form.

For details, consult each library's `README` within its respective subfolder. To see which new features have been added since the previous versions, consult their respective `CHANGELOG`s.

