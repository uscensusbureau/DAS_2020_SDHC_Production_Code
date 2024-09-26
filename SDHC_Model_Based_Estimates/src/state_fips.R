suppressPackageStartupMessages({
  library(here)
  library(tidyr)
})
here::i_am('src/state_fips.R')

data(fips_codes, package='tigris')
ST_FIPS = fips_codes %>%
  dplyr::filter(!(state %in% c("AS", "GU", "MP", "PR", "UM", "VI"))) %>%
  dplyr::distinct(state_code) %>%
  as.vector() %>%
  unlist()

saveRDS(ST_FIPS, file=here('data', 'ST_FIPS.rds'))
