##############################################################################
# Table PH5 (formerly P37): Average family size by age
#
# Summary: This script fits models to state-level noisy measurements with 
# different levels of noise added. The PH1 table shell is as follows.
#
# Universe:  Families
# Cell name
#  1:    Total
#  2:    Under 18 years
#  3:    18 years and over
#
# Usage: Rscript tnormal_ph5.R
# Return: Model cell count table written to config file location 
#
# Author: ${DEVELOPER} 
# Support: ${DEVELOPER} 
# Last Updated: 10 July 2023
##############################################################################

here::i_am('PH5/tnormal_ph5.R')
suppressPackageStartupMessages({
  library(here)
  library(jsonlite)
  library(msm)
  library(tidyr)
  library(tmvmixnorm)
})
options(tibble.width = Inf)
options(error = function() traceback(3))
source('fitting_tools.R')

# Load configuration
ph_table = 'PH5'
config_global = fromJSON(here('config_global.json'))
config_local = fromJSON(here(ph_table, "config.json"))
config = c(config_global, config_local)

set.seed(config$seed)

nm_dir = config$nm_dir

width = 2

fit_model <- function(data, geo=NULL){
    cat("\n\nCALCULATING", geo, "\n")
    ALPHA = config$alpha
    f_l = function(x){quantile(x, probs=ALPHA, na.rm=TRUE)}
    f_u = function(x){quantile(x, probs=1 - ALPHA, na.rm=TRUE)}
    TRUN = config$trun
    # the response vector will be NUM2, NUM3, DENOM
    D = matrix(c(1, 0, 0,
                 0, 1, 0,
                 0, 0, 1,
                 1, 1, -2,
                 -1, -1, TRUN), nrow=5, byrow=TRUE)
    a = 1
    lastp=0
    geoids = unique(data$GEOID) %>% sort()
    iteration_codes = unique(data$ITERATION_CODE) %>% sort()
    N = length(geoids) * length(iteration_codes)
    for (idx_geo in geoids) {
      for (idx_it in iteration_codes) {
        w1 = which(data$GEOID == idx_geo & data$ITERATION_CODE == idx_it &
             data$DATA_CELL == 1)
        w2 = which(data$GEOID == idx_geo & data$ITERATION_CODE == idx_it &
             data$DATA_CELL == 2)
        w3 = which(data$GEOID == idx_geo & data$ITERATION_CODE == idx_it &
             data$DATA_CELL == 3)
        NUM2 = data$COUNT_NOISY_NUM[w2]
        NUM3 = data$COUNT_NOISY_NUM[w3]
        DEN = data$COUNT_NOISY_DENOM[w1]
        NUM2_VAR = data$VARIANCE_NUM[w2]
        NUM3_VAR = data$VARIANCE_NUM[w3]
        DEN_VAR = data$VARIANCE_DENOM[w1]
        SIG = diag(c(NUM2_VAR, NUM3_VAR, DEN_VAR))
        l_bd = c(0, 0, 1, 0, 0)
        u_bd = c(NUM2 + 6 * sqrt(NUM2_VAR),
                 NUM3 + 6 * sqrt(NUM3_VAR),
                 DEN + 6 * sqrt(DEN_VAR),
                 NUM2 + NUM3 - 2 * DEN + 6 * sqrt(NUM2_VAR + NUM3_VAR + (4 * DEN_VAR)),
                 TRUN * DEN - NUM2 - NUM3 + 6 * sqrt(NUM2_VAR + NUM3_VAR +
                                                   TRUN ^ 2 * DEN_VAR))
        # ensure initial value is slightly off the boundary
        Dstar = rbind(D, -D)
        l_err = rep(0.5, dim(D)[1])
        u_err = -rep(0.5, dim(D)[1])
        bdstar = c(l_bd + l_err, -u_bd + u_err)
        init_val = quadprog::solve.QP(Dmat=diag(3), dvec=c(NUM2, NUM3, DEN),
                                      t(Dstar), bdstar)$solution
        tmp = try(rtmvn(10000, Mean = c(NUM2, NUM3, DEN), Sigma=SIG,
                        D=D, lower=l_bd, upper=u_bd, int=init_val))
        max_attempts = 100
        for (i in 1:max_attempts) {
          if (inherits(tmp, 'try-error')) {
            tmp = try(rtmvn(10000, Mean=c(NUM2, NUM3, DEN),
                            Sigma=SIG, D=D, lower=l_bd, upper=u_bd))
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

        data$PRED_NUM[w1] = mean(tmp[, 1] + tmp[, 2], na.rm=TRUE)
        data$PRED_NUM[w2] = mean(tmp[, 1], na.rm=TRUE)
        data$PRED_NUM[w3] = mean(tmp[, 2], na.rm=TRUE)
        data$PRED_DENOM[w1] = data$PRED_DENOM[w2] = data$PRED_DENOM[w3] =
                              mean(tmp[, 3])

        ratio1 = (tmp[, 1] + tmp[, 2]) / tmp[, 3]
        ratio2 = tmp[, 1] / tmp[, 3]
        ratio3 = tmp[, 2] / tmp[, 3]

        data$RATIO_PRED[w1] = mean(ratio1, na.rm=TRUE)
        data$RATIO_PRED_LO[w1] = f_l(ratio1)
        data$RATIO_PRED_HI[w1] = f_u(ratio1)

        data$RATIO_PRED[w2] = mean(ratio2, na.rm=TRUE)
        data$RATIO_PRED_LO[w2] = f_l(ratio2)
        data$RATIO_PRED_HI[w2] = f_u(ratio2)

        data$RATIO_PRED[w3] = mean(ratio3, na.rm=TRUE)
        data$RATIO_PRED_LO[w3] = f_l(ratio3)
        data$RATIO_PRED_HI[w3] = f_u(ratio3)

        a = a + 1
        p = min(round(a/N * 100), 100)
        if(a==2 | ((p %% 5)==0 & p!=lastp)) cat("\rProgress:", p, "%")
        lastp = p
    }
  }
  return(data)
}


main <- function() {
  cat("\nStart processing US dataset\n")
  region_types = get_region_types("us", "num")
  for (i in seq_along(region_types)) {
    data = fit_split_nm(FALSE, "us", region_types[i])
    fit_and_save_data(data, region_types[i], fit_model, ph_table)
  }

  cat("\n\nStart processing PUERTO RICO dataset\n")
  pr_region_types = get_region_types("pr", "num")
  for (i in seq_along(pr_region_types)) {
    data = fit_split_nm(FALSE, "pr", pr_region_types[i])
    fit_and_save_data(data, pr_region_types[i], fit_model, ph_table, TRUE)
  }

  cat("\n\nCompleted all geographies for table", ph_table, "\n")
}


main()
