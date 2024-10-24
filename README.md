This repository contains source code for the PH-SAFE disclosure
avoidance application. PH-SAFE was used by the Census Bureau for the
protection of individual 2020 Census responses in the tabulation and
publication of the Supplemental Demographic and Housing
Characteristics File (S-DHC). Previous source code releases have
included the code for earlier data releases focused on demographic and
housing characteristics respectively.

PH-SAFE combines Census survey results of individuals and households
and infuses them with noise to create *privacy-protected
tabulations*. The algorithm links information about individuals and
households and then truncates the resulting table to exclude
especially large households. This is done because information about
these households is highly disclosive. Noise is then infused into the
innermost detail cells of the truncated table. The PH-SAFE code itself
can be found in the `phsafe` directory of this repository. PH-SAFE was
built on Tumult's "Analytics" and "Core" platforms, whose source is
found in the `tumult` subdirectory and makes use of customized CEF
(Census Edited File) readers implemented by MITRE and included in the
`mitre` subdirectory. All of these components are implemented in
Python and the latest version of the platforms can be found at
[[https://tmlt.dev/]].

The resulting protected table is then processed by a variety of
statistical models to remove certain illogical results from the
noise-infusion, such as negative counts or infinite ratios (whose
demoninators are 0). The post-processing code can be found in the
`SDHC_Model_Based_Estimates` subdirectory and is written in R.

In the interests of both transparency and scientific advancement, the
Census Bureau committed to releasing any source code used in creation
of products protected by formal privacy guarantees. In the case of the
the Detailed Demographic & Housing Characteristics publications, this
includes code developed under contract by Tumult Software (tmlt.io)
and MITRE corporation. Tumult's underlying platform is evolving and
the code in the repository is a snapshot of the code used for the
production of S-DHC

The bureau has already separately released the internally developed
software for the Top Down Algorithm (TDA) used in production of the
2020 Redistricting and the 2020 Demographic & Housing Characteristics
products.

This software for this repository is divided into five subdirectories:
* `configs` contains the specific configuration files used for the
  production S-DHC runs, including privacy loss budget (PLB) allocations
  and the rules for adaptive table generation. These configurations reflect
  decisions by the Bureau's DSEP (Data Stewardship Executive Policy) committee
  based on experiments conducted by Census Bureau staff.
* `phsafe` contains the source code for the application itself as used
   to generate the protected microdata used in production.
* `mitre/cef_readers` contains code by MITRE to read the Census input
  files used by the SafeTab applications.
* `tumult` contains the Tumult Analytics platform. This is divided
   into `common`, `analytics`, and `core` directories. The `core` directory
   also includes a pre-packaged Python *wheel* for the core library.
* `ctools` contains Python utility libraries developed the the Census
  Bureau's DAS team and used by the MITRE CEF readers.
* `SDHC_Model_Based_Estimates` contains the R modeling code used for
  postprocessing the privacy-protected join table produced by PH-SAFE.
