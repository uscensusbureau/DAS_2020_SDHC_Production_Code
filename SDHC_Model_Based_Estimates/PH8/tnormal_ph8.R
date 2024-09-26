##############################################################################
# Table PH8 (formerly H12): Average household size of occupied housing units
# by tenure.
#
# Summary: This script fits models to state-level noisy measurements with
# different levels of noise added. The PH1 table shell is as follows.
#
# Universe:  Occupied housing units
# Cell name
#  1:    Total
#  2:    Owner occupied
#  3:    Renter occupied
#
# Usage: Rscript tnormal_ph8.R
# Return: Model cell count table written to config file location
#
# Author: Ryan Janicki (CSRM)
# Support: Nate Cady (MITRE) & Nathan Welch (MITRE)
# Last Updated: 10 July 2023
##############################################################################

here::i_am('PH8/tnormal_ph8.R')
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
ph_table = 'PH8'
config_global = fromJSON(here('config_global.json'))
config_local = fromJSON(here(ph_table, "config.json"))
config = c(config_global, config_local)

set.seed(config$seed)

nm_dir = config$nm_dir

width = 2

fit_model <- function(data, geo=NULL){
  cat("\n\nCALCULATING", geo, "\n")
  ALPHA = config$alpha
  f_l = function(x){quantile(x, probs=ALPHA)}
  f_u = function(x){quantile(x, probs=1 - ALPHA)}
  TRUN = config$trun
  D = matrix(c(0, 1, 1, -1, -1, TRUN), nrow=3, byrow=TRUE)
  N = nrow(data)
  lastp = 0
  for (idx in 1:N) {
    NUM = data$COUNT_NOISY_NUM[idx]
    DEN = data$COUNT_NOISY_DENOM[idx]
    NUM_VAR = data$VARIANCE_NUM[idx]
    DEN_VAR = data$VARIANCE_DENOM[idx]
    l_bd = matrix(c(1, 0, 0), ncol=1)
    u_bd = matrix(c(abs(DEN) + 6*sqrt(DEN_VAR),
                  abs(NUM - DEN) + 6*sqrt(NUM_VAR + DEN_VAR),
                  abs(TRUN * DEN - NUM) + 6*sqrt(TRUN^2 * DEN_VAR + NUM_VAR)),
                  ncol = 1)
    SIG = diag(c(NUM_VAR, DEN_VAR))
    # ensure initial value is slightly off the boundary
    Dstar = rbind(D, -D)
    l_err = rep(0.5, dim(D)[1])
    u_err = -rep(0.5, dim(D)[1])
    bdstar = c(l_bd + l_err, -u_bd + u_err)
    init_val = quadprog::solve.QP(Dmat=diag(2), dvec=c(NUM, DEN),
                                  t(Dstar), bdstar)$solution
    tmp = try(rtmvn(10000, Mean=c(NUM, DEN), Sigma=SIG, D=D,
                    lower=l_bd, upper=u_bd, int=init_val))
    max_attempts = 100
    for (i in 1:max_attempts) {
      if (inherits(tmp, 'try-error')) {
      tmp = try(rtmvn(10000, Mean=c(NUM, DEN), Sigma=SIG, D=D,
                      lower=l_bd, upper=u_bd))
      }
      else {
        if (i != 1) { # We only care if first attempt failed
          print(paste0("rtmvn successful on attempt # ", i))
      }
        break
      }
      if (i %% 10 == 0) {
        print(paste0("rtmvn failing, attempt number: ", i))
      }
    }



    tmp_ratio = tmp[, 1] / tmp[, 2]
    data$RATIO_PRED[idx] = mean(tmp_ratio)
    data$RATIO_PRED_LO[idx] = f_l(tmp_ratio)
    data$RATIO_PRED_HI[idx] = f_u(tmp_ratio)

    p = round(idx/N * 100)
    if(idx==1 | ((p %% 5)==0 & p!=lastp)) cat("\rProgress:", p, "%")
    lastp = p
  }
  return(data)
}


main <- function() {
  cat("\nStart processing US dataset\n")
  region_types = get_region_types("us", "num")
  for (i in seq_along(region_types)) {
    data = fit_split_nm(TRUE, "us", region_types[i])
    fit_and_save_data(data, region_types[i], fit_model, ph_table)
  }

  cat("\n\nStart processing PUERTO RICO dataset\n")
  pr_region_types = get_region_types("pr", "num")
  for (i in seq_along(pr_region_types)) {
    data = fit_split_nm(TRUE, "pr", pr_region_types[i])
    fit_and_save_data(data, pr_region_types[i], fit_model, ph_table, TRUE)
  }

  cat("\n\nCompleted all geographies for table", ph_table, "\n")
}


main()
