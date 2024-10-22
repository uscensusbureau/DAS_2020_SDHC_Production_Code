##############################################################################
# Summary: This script summarizies model-based estimate performance for all
# tables.
#
# Usage: Rscript summarize_results.R
# Return: Model cell count table written to out_{count, ratio}.tex
#
# Author: ${DEVELOPER} 
# Support: ${DEVELOPER} 
# Last Updated: 10 July 2023
##############################################################################

here::i_am('summary/summarize_results.R')
suppressPackageStartupMessages({
  library(here)
  library(jsonlite)
  library(tidyr)
  library(xtable)
  library(data.table)
})
options(tibble.width = Inf)
options(error=traceback)
options(show.error.locations = TRUE)

# summarize results of Experiment 5

config = fromJSON(here('config_global.json'))

alpha = config$alpha
trun = config$trun
# count tables
count_tables = paste0("PH", c(2:4, 6:7))
Ncount = length(count_tables)

# directory to save summarized results
save_dir = here('summary', 'results')

metrics = c("min","max","p_neg","LEN")

out = matrix(0, nrow = Ncount*2, ncol = length(metrics))
colnames(out) = metrics
rownames(out)[seq(2, Ncount*2, 2)] = paste0(count_tables, "_PRED")
rownames(out)[seq(1, Ncount*2, 2)] = paste0(count_tables, "_NM")

#summarize output:  count tables
for (tbl in count_tables) {
  # load state results file for table tbl
  file_nm = here(tbl, 'estimates', paste0(tbl, "_state_trunc_normal.csv"))
  dat = read.csv(file_nm, sep="|")

  # calculate raw noisy measurement performance results
  out[paste0(tbl, "_NM"), "min"] = min(dat$COUNT_NOISY)
  out[paste0(tbl, "_NM"), "max"] = max(dat$COUNT_NOISY)
  out[paste0(tbl, "_NM"), "p_neg"] = 
    length(which(dat$COUNT_NOISY<=0)) / dim(dat)[1]
  out[paste0(tbl, "_NM"), "LEN"] = median(dat$MOE_NOISY*2)

  # calculate model-based performance results
  out[paste0(tbl, "_PRED"), "min"] = min(dat$PRED)
  out[paste0(tbl, "_PRED"), "max"] = max(dat$PRED)
  out[paste0(tbl, "_PRED"), "p_neg"] = 
    length(which(dat$PRED<= 0)) / dim(dat)[1]
  out[paste0(tbl, "_PRED"), "LEN"] = median(dat$PRED_HI - dat$PRED_LO)
}
out_count = out



#summarize output:  ratio tables

ratio_tables = paste0("PH", c(1, 5, 8))
Nratio = length(ratio_tables)

metrics = c("min","max","p_bad","LEN")

out = matrix(0, nrow = Nratio*2, ncol = length(metrics))
colnames(out) = metrics
rownames(out)[seq(2, Nratio*2, 2)] = paste0(ratio_tables, "_PRED")
rownames(out)[seq(1, Nratio*2, 2)] = paste0(ratio_tables, "_NM")

for (tbl in ratio_tables) {
  # load state results file for table tbl
  file_nm = here(tbl, 'estimates', paste0(tbl, "_state_trunc_normal.csv"))
  dat = read.csv(file_nm, sep="|")

  x_bd = 1
  if (tbl == "PH5") x_bd = 2

  out[paste0(tbl, "_NM"), "min"] = min(dat$RATIO_NOISY)
  out[paste0(tbl, "_NM"), "max"] = max(dat$RATIO_NOISY)
  out[paste0(tbl, "_NM"), "p_bad"] = 
    length(
      which(
        (dat$DATA_CELL == 1 & (dat$RATIO_NOISY < x_bd | dat$RATIO_NOISY > trun)) | ( dat$DATA_CELL != 1 & dat$RATIO_NOISY < 0 )
      ) ) / dim(dat)[1]

  out[paste0(tbl, "_PRED"), "min"] = min(dat$RATIO_PRED)
  out[paste0(tbl, "_PRED"), "max"] = max(dat$RATIO_PRED)
  out[paste0(tbl, "_PRED"), "p_bad"] = 
    length(
      which(
        (dat$DATA_CELL == 1 & (dat$RATIO_PRED < x_bd | dat$RATIO_PRED > trun)
        ) | (dat$DATA_CELL != 1 & dat$RATIO_PRED < 0)
      )
    ) / dim(dat)[1]
  out[paste0(tbl, "_PRED"), "LEN"] = mean(dat$RATIO_PRED_HI - dat$RATIO_PRED_LO)
}   
out_ratio = out

out_count[, "p_neg"] = 100*out_count[, "p_neg"]
out_ratio[, "p_bad"] = 100*out_ratio[, "p_bad"]

print(xtable(out_count), booktabs=TRUE, 
  file=here("summary", "out_count.tex"))
print(xtable(out_ratio), booktabs=TRUE,
  file = here("summary","out_ratio.tex"))

count = as.data.table(out_count, keep.rownames=TRUE) %>%
  melt(id.var='rn', value.name='harden') %>%
  setnames(old='rn', new='table') %>%
  .[,':='(table=tolower(table), variable=tolower(variable))]
fwrite(count, file=here('summary', 'out_count.csv'))

ratio = as.data.table(out_ratio, keep.rownames=TRUE) %>%
  melt(id.var='rn', value.name='harden') %>%
  setnames(old='rn', new='table') %>%
  .[,':='(table=tolower(table), variable=tolower(variable))]

fwrite(ratio, file=here('summary', 'out_ratio.csv'))

