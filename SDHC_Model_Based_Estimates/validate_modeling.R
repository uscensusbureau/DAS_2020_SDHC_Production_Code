suppressPackageStartupMessages({
library(dplyr)
library(here)
library(jsonlite)
})


config = fromJSON(here('config_global.json'))
trun = config$trun


collect_file_paths <- function(nums) {
  file_paths = list()
  for (i in nums) {
    folder = paste0("PH", i, "/estimates")
    filenames = list.files(folder)
    file_paths = c(file_paths, file.path(folder, filenames))
  }
  return(file_paths)
}


validate_single_model <- function(file) {
  data = read.table(file, sep="|", header=TRUE)
  if (!all(data$PRED%%1==0 & data$PRED > 0)) {
    stop("ERROR: Not all predictions are positive integers.")
  }

  # validate column order for PH 2, 3, 4, 6, and 7
  vector_single <- c("GEOID","REGION_TYPE","ITERATION_CODE",
  "DATA_CELL","COUNT_NOISY","VARIANCE","MOE_NOISY","PRED",
  "PRED_LO","PRED_HI")
  if(!identical(colnames(data),vector_single)) {
  stop("ERROR: Columns for single models are not in correct order.")
  }

  stdev = sqrt(data$VARIANCE)
  pred_diff = data$PRED - data$COUNT_NOISY
  if (any(abs(pred_diff) > (6 * stdev))) {
    stop("ERROR: Prediction diff is > 6 * stdev")
  }
}


validate_split_model_15 <- function(file) {
  data = read.table(file, sep="|", header=TRUE)
  if (any(0 > data$RATIO_PRED | data$RATIO_PRED > trun)) {
    stop("ERROR: Ratio either < 0 or > trun")
  }

  # for PH1 and 5, num comes before denom
  vector_15 <- c("GEOID","REGION_TYPE","ITERATION_CODE",
  "DATA_CELL","COUNT_NOISY_NUM","VARIANCE_NUM","MOE_NOISY_NUM",
    "COUNT_NOISY_DENOM","VARIANCE_DENOM","MOE_NOISY_DENOM",
    "RATIO_NOISY","RATIO_PRED","PRED_NUM","PRED_DENOM",
    "RATIO_PRED_LO","RATIO_PRED_HI")

  # validate column order for 1, 5,
  if(!identical(colnames(data),vector_15)) {
    stop("ERROR: Columns for split models 1 or 5 are not in correct order.")
  }
}

validate_split_model_8 <- function(file) {
  data = read.table(file, sep="|", header=TRUE)
  if (any(1 > data$RATIO_PRED | data$RATIO_PRED > trun)) {
    stop("ERROR: Ratio either < 0 or > trun")
  }

  # for PH8, there are two less columns and order is different with denom coming before num
  vector_8 <- c("GEOID","REGION_TYPE","ITERATION_CODE","DATA_CELL",
    "COUNT_NOISY_DENOM","VARIANCE_DENOM","MOE_NOISY_DENOM",
    "COUNT_NOISY_NUM","VARIANCE_NUM","MOE_NOISY_NUM","RATIO_NOISY",
    "RATIO_PRED","RATIO_PRED_LO","RATIO_PRED_HI")

  # validate column order for 8
  if(!identical(colnames(data),vector_8)) {
    stop("ERROR: Columns for split models 1 or 5 are not in correct order.")
  }

}

main <- function() {
  single_nums = list(2, 3, 4, 6, 7)
  split_15_nums = list(1,5)
  split_8_nums = list(8)
  single_paths = collect_file_paths(single_nums)
  split_15_paths = collect_file_paths(split_15_nums)
  split_8_paths = collect_file_paths(split_8_nums)
  for(file in single_paths) {
    validate_single_model(file)
  }
  for(file in split_15_paths) {
    validate_split_model_15(file)
  }
  for(file in split_8_paths) {
    validate_split_model_8(file)
  }
}


main()
