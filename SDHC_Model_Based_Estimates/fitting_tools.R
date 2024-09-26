
here::i_am('fitting_tools.R')
suppressPackageStartupMessages({
  library(here)
  library(tidyr)
  library(tmvmixnorm)
  library(jsonlite)
})
options(tibble.width = Inf)

config = fromJSON(here('config_global.json'))
exclude_region = config$exclude_levels

fit_and_save_data <- function(data, geo, fit_model, ph_table, pr=FALSE) {
  # For using the specific PH fitting function then saving the results
  data_pred = fit_model(data, geo=geo)
  if(pr) {
    prefix1 = 'PR_'
  } else {
    prefix1 = ""
  }

  if(toupper(geo) == 'USA') {
    prefix2 = 'nation'
  } else {
    prefix2 = tolower(geo)
  }
  fname = paste0(ph_table, "_", prefix1, prefix2, "_trunc_normal.csv")
  write.table(data_pred, here(ph_table, 'estimates', fname), row.names = FALSE,
              sep = "|")
}


get_nm_filename <- function(nm_dir, geotype, ph_table, num) {
  if(num == "simple") {
    split = ""
  } else if(num == "num") {
    split = "_num"
  } else {
    split = "_denom"
  }

  geotype = tolower(geotype)
  if(geotype != "pr") {
    geotype = "us"
  }
  filename = paste0(nm_dir, geotype, "/", ph_table, split, "/",
                    tolower(ph_table), split, ".csv")
  return(filename)
}

get_region_types <- function(geotype, datatype) {
  nm_file = get_nm_filename(nm_dir, geotype, ph_table, datatype)
  df <- data.table::fread(file = nm_file, select = "REGION_TYPE", sep = "|", header = TRUE)
  df <- unique(df)
  uniq_region_type <- as.vector(unlist(df))
  if(length(exclude_region)) {
    region_type <- setdiff(uniq_region_type, exclude_region)
  } else {
    region_type <- uniq_region_type
  }

  return(region_type)
}

fit_simple_nm <- function(inline_iter, geotype, geo_level) {
  nm_file = get_nm_filename(nm_dir, geotype, ph_table, "simple")
  nm = read.table(nm_file, sep = "|", header = TRUE) %>%
    dplyr::filter(REGION_TYPE == geo_level) %>%
    dplyr::rename(COUNT_NOISY = COUNT) %>%
    dplyr::rename(DATA_CELL = paste0(toupper(ph_table), "_DATA_CELL")) %>%
    dplyr::mutate(MOE_NOISY = qnorm(0.95)*sqrt(VARIANCE)) %>%
    dplyr::select(-c(NOISE_DISTRIBUTION))

  # aggregate internal state cells to marginal cells
  agg_table = NULL
  if(inline_iter == TRUE) {
    for (idx_a in seq_along(src_cells)) {
      tmp = nm %>%
        dplyr::filter(DATA_CELL %in% src_cells[[idx_a]]) %>%
        dplyr::group_by(REGION_ID, ITERATION_CODE, REGION_TYPE) %>%
        dplyr::summarise(COUNT_NOISY = sum(COUNT_NOISY),
                         VARIANCE = sum(VARIANCE), .groups='keep') %>%
        dplyr::mutate(DATA_CELL = dest_cells[[idx_a]]) %>%
        dplyr::mutate(MOE_NOISY = qnorm(0.95)*sqrt(VARIANCE))
      agg_table = rbind(agg_table, tmp)
    }
    nm = rbind(agg_table[, colnames(nm)], nm)
  }
  else {
    for (idx_a in seq_along(src_cells)) {
      tmp = nm %>%
        dplyr::filter(DATA_CELL %in% src_cells[[idx_a]]) %>%
        dplyr::group_by(REGION_ID, REGION_TYPE) %>%
        dplyr::summarise(COUNT_NOISY = sum(COUNT_NOISY),
                         VARIANCE = sum(VARIANCE), .groups='keep') %>%
        dplyr::mutate(DATA_CELL = dest_cells[[idx_a]]) %>%
        dplyr::mutate(MOE_NOISY = qnorm(0.95)*sqrt(VARIANCE))
      agg_table = rbind(agg_table, tmp)
    }
  nm = rbind(agg_table[, colnames(nm)], nm) %>%
    dplyr::mutate(ITERATION_CODE = "*")
  nm = nm[, c(1:2, 7, 3:6)]
  }

  data = nm %>%
    dplyr::rename(GEOID = REGION_ID) %>%
    dplyr::mutate(GEOID = stringr::str_pad(GEOID, width=width, side="left",
                          pad = "0")) %>%
    dplyr::mutate(PRED = NA) %>%
    dplyr::mutate(PRED_LO = NA) %>%
    dplyr::mutate(PRED_HI = NA)

  # the logical constraints that need to be imposed are
  # Y >= 0
  return(data)
}


fit_split_nm <- function(inline_data_cell, geotype, geo_level) {
  nm_num_file = get_nm_filename(nm_dir, geotype, ph_table, "num")
  nm_num = read.table(nm_num_file, sep = "|", header = TRUE) %>%
    dplyr::filter(REGION_TYPE == geo_level) %>%
    dplyr::rename(COUNT_NOISY_NUM = COUNT) %>%
    dplyr::rename(DATA_CELL = paste0(toupper(ph_table), "_NUM_DATA_CELL")) %>%
    dplyr::rename(VARIANCE_NUM = VARIANCE) %>%
    dplyr::mutate(MOE_NOISY_NUM = qnorm(0.95)*sqrt(VARIANCE_NUM)) %>%
    dplyr::select(-c(NOISE_DISTRIBUTION))

  # aggregate cells 2 and 3
  tmp = nm_num %>%
    dplyr::select(-MOE_NOISY_NUM) %>%
    dplyr::group_by(REGION_ID, ITERATION_CODE, REGION_TYPE) %>%
    dplyr::summarise(COUNT_NOISY_NUM = sum(COUNT_NOISY_NUM),
                     VARIANCE_NUM = sum(VARIANCE_NUM)) %>%
    dplyr::mutate(DATA_CELL = 1) %>%
    dplyr::mutate(MOE_NOISY_NUM = qnorm(0.95)*sqrt(VARIANCE_NUM))

  nm_num = rbind(nm_num, tmp[, colnames(nm_num)])


  nm_denom_file = get_nm_filename(nm_dir, geotype, ph_table, "denom")
  nm_denom = read.table(nm_denom_file, sep = "|", header = TRUE) %>%
    dplyr::filter(REGION_TYPE == geo_level) %>%
    dplyr::rename(VARIANCE_DENOM = VARIANCE) %>%
    dplyr::rename(COUNT_NOISY_DENOM = COUNT) %>%
    dplyr::rename(DATA_CELL = paste0(toupper(ph_table), "_DENOM_DATA_CELL")) %>%
    dplyr::mutate(MOE_NOISY_DENOM = qnorm(0.95)*sqrt(VARIANCE_DENOM)) %>%
    dplyr::select(-c(NOISE_DISTRIBUTION))


  if(inline_data_cell == TRUE) {
    # aggregate cells 2 and 3
    tmp = nm_denom %>%
      dplyr::select(-MOE_NOISY_DENOM) %>%
      dplyr::group_by(REGION_ID, ITERATION_CODE, REGION_TYPE) %>%
      dplyr::summarise(COUNT_NOISY_DENOM = sum(COUNT_NOISY_DENOM),
                       VARIANCE_DENOM = sum(VARIANCE_DENOM), .groups='keep') %>%
      dplyr::mutate(DATA_CELL = 1) %>%
      dplyr::mutate(MOE_NOISY_DENOM = qnorm(0.95)*sqrt(VARIANCE_DENOM))

    nm_denom = rbind(nm_denom, tmp[, colnames(nm_denom)])
    nm = nm_denom %>%
      dplyr::left_join(nm_num,
                       by = c("REGION_ID" = "REGION_ID",
                              "REGION_TYPE" = "REGION_TYPE",
                              "ITERATION_CODE" = "ITERATION_CODE",
                              "DATA_CELL" = "DATA_CELL")) %>%
      dplyr::mutate(RATIO_NOISY = COUNT_NOISY_NUM / COUNT_NOISY_DENOM)
    data = nm %>%
    dplyr::rename(GEOID = REGION_ID) %>%
    dplyr::mutate(GEOID = stringr::str_pad(GEOID, width=width, side="left",
                                           pad = "0")) %>%
    dplyr::mutate(RATIO_PRED = NA) %>%
    dplyr::mutate(RATIO_PRED_LO = NA) %>%
    dplyr::mutate(RATIO_PRED_HI = NA)
  }
  else {
    nm = nm_num %>%
      dplyr::left_join(nm_denom,
                       by = c("REGION_ID" = "REGION_ID",
                              "REGION_TYPE" = "REGION_TYPE",
                              "ITERATION_CODE" = "ITERATION_CODE")) %>%
      dplyr::mutate(RATIO_NOISY = COUNT_NOISY_NUM / COUNT_NOISY_DENOM) %>%
      dplyr::rename(DATA_CELL = DATA_CELL.x) %>%
      dplyr::select(-DATA_CELL.y)
    data = nm %>%
      dplyr::rename(GEOID = REGION_ID) %>%
      dplyr::mutate(GEOID = stringr::str_pad(GEOID, width=width,
                                             side="left", pad="0")) %>%
      dplyr::mutate(RATIO_PRED = NA) %>%
      dplyr::mutate(PRED_NUM = NA) %>%
      dplyr::mutate(PRED_DENOM = NA) %>%
      dplyr::mutate(RATIO_PRED_LO = NA) %>%
      dplyr::mutate(RATIO_PRED_HI = NA)
  }

  return(data)
}
