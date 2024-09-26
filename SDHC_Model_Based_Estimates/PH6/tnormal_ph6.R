##############################################################################
# Table PH6 (formerly P40): Family Type and Age For Own Children Under 18 Years
#
# Summary: This script fits models to state-level noisy measurements with 
# different levels of noise added. The PH6 table shell is as follows.
#
# Universe:  Own children under 18 years
# Cell Name
# 1    Total:
# 2    In married couple families:
# 3    Under 4 years
# 4    4 and 5 years
# 5    6 to 11 years
# 6    12 to 17 years
# 7    In cohabiting couple families:
# 8    Under 4 years
# 9    4 and 5 years
# 10   6 to 11 years
# 11   12 to 17 years
# 12   In male householder, no spouse or partner present family
# 13   Under 4 years
# 14   4 and 5 years
# 15   6 to 11 years
# 16   12 to 17 years
# 17   In female householder, no spouse or partner present family
# 18   Under 4 years
# 19   4 and 5 years
# 20   6 to 11 years
# 21   12 to 17 years
#
# Usage: Rscript tnormal_ph6.R
# Return: Model cell count table written to config file location 
#
# Author: Ryan Janicki (CSRM)
# Support: Nate Cady (MITRE) & Nathan Welch (MITRE)
# Last Updated: 10 July 2023
##############################################################################

here::i_am('PH6/tnormal_ph6.R')
suppressPackageStartupMessages({
  library(here)
  library(jsonlite)
  library(msm)
  library(tidyr)
  library(tmvmixnorm)
})
options(tibble.width = Inf)
source('fitting_tools.R')

# Load configuration
ph_table = 'PH6'
config_global = fromJSON(here('config_global.json'))
config_local = fromJSON(here(ph_table, "config.json"))
config = c(config_global, config_local)

set.seed(config$seed)

nm_dir = config$nm_dir

# internal cells
src_cells = list(3:6, 8:11, 13:16, 18:21, c(3:6, 8:11, 13:16, 18:21))
# marginal cells
dest_cells = list(2, 7, 12, 17, 1)

# format the geographic identifiers
width = 2

fit_model <- function(data, geo=NULL){
  cat("\n\nCALCULATING", geo, "\n")
  # truncation level
  ALPHA = config$alpha
  NUM_ITER = config$iteration_num
  f_l = function(x){quantile(x, probs = ALPHA)}
  f_u = function(x){quantile(x, probs = 1 - ALPHA)}
  lastp = 0
  a = 1
  geoids = unique(data$GEOID)
  iteration_codes = unique(data$ITERATION_CODE)
  N = length(geoids) * length(iteration_codes) * length(src_cells)
  for (idx_geo in geoids) {
    for (idx_it in iteration_codes) {
      for (idx_s in seq_along(src_cells)) {
        # internal cells
        wI = which(data$GEOID == idx_geo &
                   data$ITERATION_CODE == idx_it &
                   data$DATA_CELL %in% src_cells[[idx_s]])

        # outer cell
        wO = which(data$GEOID == idx_geo &
                   data$ITERATION_CODE == idx_it &
                   data$DATA_CELL %in% dest_cells[[idx_s]])

        # posterior mean (point estimates)
        Z = data$COUNT_NOISY[wI]
        D = data$VARIANCE[wI]
        alph = -Z/sqrt(D)
        den = 1 - pnorm(alph)
        data$PRED[wI] = round(Z + dnorm(alph)*sqrt(D)/den)
        data$PRED[wO] = sum(data$PRED[wI])

        # credible interval
        tmp = matrix(nrow = NUM_ITER, ncol = length(src_cells[[idx_s]]))

        for (idx_y in seq_along(Z)) {
          tmp[, idx_y] = rtnorm(NUM_ITER, mean = Z[idx_y], sqrt(D[idx_y]), lower = 0)
          data$PRED_LO[wI[idx_y]] = floor(f_l(tmp[, idx_y]))
          data$PRED_HI[wI[idx_y]] = ceiling(f_u(tmp[, idx_y]))
        }

        agg = apply(tmp, 1, sum)
        data$PRED_LO[wO] = floor(f_l(agg))
        data$PRED_HI[wO] = ceiling(f_u(agg))

        p = round(a/N * 100)
        if(a==1 | ((p %% 5)==0 & p!=lastp)) cat("\rProgress:", p, "%")
        a = a + 1
        lastp = p
      }
    }
  }
  return(data)
}


main <- function() {
  cat("\nStart processing US dataset\n")
  region_types = get_region_types("us", "simple")
  for (i in seq_along(region_types)) {
    data = fit_simple_nm(FALSE, "us", region_types[i])
    fit_and_save_data(data, region_types[i], fit_model, ph_table)
  }

  cat("\n\nStart processing PUERTO RICO dataset\n")
  pr_region_types = get_region_types("pr", "simple")
  for (i in seq_along(pr_region_types)) {
    data = fit_simple_nm(FALSE, "pr", pr_region_types[i])
    fit_and_save_data(data, pr_region_types[i], fit_model, ph_table, TRUE)
  }

  cat("\n\nCompleted all geographies for table", ph_table, "\n")
}


main()
