# required packages
required.packages = c(
    'cowplot',
    'data.table',
    'ggplot2',
    'here', 
    'jsonlite', 
    'msm',
    'tidyr', 
    'tigris', 
    'tmvmixnorm',
    'xtable',
    'readr',
    'mvtnorm'
)

# list of currently installed packages
packages = installed.packages()[,"Package"]

# list of packages needed 
missing.packages = required.packages[!(required.packages %in% packages)]

# install missing packages
if(length(missing.packages)) install.packages(missing.packages, repos=${DAS_R_REPO_URL})
