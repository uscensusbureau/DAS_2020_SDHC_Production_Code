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

## Access to the Deliverable

The source code and documentation for this deliverable can be accessed by executing the following command at the command line (or entering the URL into the clone window of a client, e.g., Github Desktop):

```
git clone https://decennial-census:juD7SnKzbfasAAT_tUqy@gitlab.com/tumult-labs/PHSafe-release.git
```

In the URL above, `juD7SnKzbfasAAT_tUqy` is a Gitlab deploy token associated with the username `decennial-census`.  This grants read access to this repository.

## Contents

In the repository there are four folders, each of which contains a component of the release:

- **Core 0.12.0**: A Python library for performing differentially private computations. The design of Tumult Core is based on the design proposed in the [OpenDP White Paper](https://projects.iq.harvard.edu/files/opendp/files/opendp_programming_framework_11may2020_1_01.pdf), and can automatically verify the privacy properties of algorithms constructed from Tumult Core components. Tumult Core is scalable, includes a wide variety of components to handle various query types, and supports multiple privacy definitions. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/core/v0.12/.
- **Analytics 0.8.3**: A Python library for privately answering statistical queries on tabular data, implemented using Tumult Core. It is built on PySpark, allowing it to scale to large datasets. Its privacy guarantees are based on differential privacy, a technique that perturbs statistics to provably protect the data of individuals in the original dataset. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/analytics/v0.8/.
- **Common 0.8.7**: A Python library with utilities for reading and validating data. Code in Common is designed not to be specific to Census applications.
- **PHSafe 3.0.0**: The main program of this release. It produces the Supplemental Demographic and Housing Characteristics File (S-DHC).

PHSafe also requires a CEF reader module for reading data from Census' file formats. The CEF reader is implemented separately, and is therefore not included in this release. We do include a built-in CSV reader for PHSafe, which can be used for PHSafe if the input files are in a CSV form.

For details, consult each library's `README` within its respective subfolder. To see which new features have been added since the previous versions, consult their respective `CHANGELOG`s.

## Sample Input Files

This release also comes with a set of synthetic data files that can be used to test PHSafe. The ZIP file containing the sample files is hosted on Amazon Simple Storage Service (Amazon S3). Please note that the download link will be valid until 12:00 pm Eastern on 2024-04-09.

**To download the ZIP file from the browser:**

 Please click this [Amazon S3 URL](https://s3.us-east-1.amazonaws.com/tumult.data.census/phsafe-3.0.0/phsafe-300m-input.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA25LEV2NNTS4WZ777%2F20240402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240402T161530Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=4db6e57160e0527d44241ce22dbe0e0ac98d5768632ca0b8b4d9a1bddcdb21c9) to download the ZIP file.


**To download the ZIP file from the command line:**

- Execute the following on the command line:

```bash
curl "https://s3.us-east-1.amazonaws.com/tumult.data.census/phsafe-3.0.0/phsafe-300m-input.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA25LEV2NNTS4WZ777%2F20240402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240402T161530Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=4db6e57160e0527d44241ce22dbe0e0ac98d5768632ca0b8b4d9a1bddcdb21c9" -L -o phsafe-300m-input.zip
```

- To unzip the zip file:

```bash
unzip phsafe-300m-input.zip -d <path to directory>/phsafe-300m-input
```

The downloaded file `phsafe-300m-input.zip` will contain the following input files that work with the CSV reader:

- `geo.csv`: A representation of a custom geography lookup table derived from the CEF Unit file and the GRFC that is input to DAS.
- `units.csv`: A representation of custom unit records derived from the CEF Unit file that is input to DAS.
- `persons.csv`: A representation of custom person records derived from the CEF Person file that is input to DAS.
- `config_puredp.json`: PHSafe config file containing input parameters for running PHSafe under PureDP (e.g. `privacy_budget`, `tau`, `state_filter`, `reader`, `privacy_defn`).
- `config_zcdp.json`: PHSafe config file containing input parameters for running PHSafe under Rho zCDP (e.g. `privacy_budget`, `tau`, `state_filter`, `reader`, `privacy_defn`).


See [PHSafe Spec Doc](phsafe/PHSafe_Documentation.pdf) for a description of each file.
