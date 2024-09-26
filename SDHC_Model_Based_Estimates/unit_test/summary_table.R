here::i_am('unit_test/summary_table.R')
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
  library(here)
  library(jsonlite)
  library(magrittr)
})
theme_set(theme_cowplot())

ratio_harden = fread(file=here('summary', 'out_ratio.csv'))
ratio_csrm = 
    rbind(
        'ph1_pred' = c(0.17, 6.30, 0.00, 0.24, 90.20, 0.26),
        'ph5_pred' = c(0.50, 6.20, 0.00, 0.20, 89.28, 0.34),
        'ph8_pred' = c(1.44, 6.07, 0.00, 0.23, 90.65, 0.25)
    ) %>% 
    as.data.table(keep.rownames=TRUE) %>%
    set_colnames(c('table', 'min', 'max', 'p_bad', 'rmse', 'cov', 'len')) %>%
    melt(id.vars='table', value.name='csrm')

ratio = merge(ratio_csrm, ratio_harden, by=c('table', 'variable'))
ggplot(data=ratio) + 
    geom_abline(intercept=0, slope=1, linetype='dashed', col='grey') +
    geom_point(aes(x=harden, y=csrm))

count_harden = fread(file=here('summary', 'out_count.csv'))
count_csrm =
    rbind(
        'ph2_pred' = c(1478, 36320972, 0.00, 157.90, 91.10, 402.95, 107.62, 88.24, 399.66 ),
        'ph3_pred' = c(4, 9033149, 0.00, 96.52, 91.35, 55.97, 14.40, 94.12, 38.23),
        'ph4_pred' = c(48, 29788113, 0.00, 139.36, 91.18, 402.17, 139.21, 88.31, 398.73),
        'ph6_pred' = c(179, 7830059, 0.00, 203.57, 88.70, 401.13, 125.16, 85.19, 399.22 ),
        'ph7_pred' = c(23, 36320844, 0.00, 92.25, 89.95, 137.83, 44.46, 91.18, 135.24 )
    )  %>% 
    as.data.table(keep.rownames=TRUE) %>%
    set_colnames(c('table', 'min', 'max', 'p_neg', 'rmse', 'cov', 'len', 
                    'rmse_s', 'cov_s', 'len_s')) %>%
    melt(id.vars='table', value.name='csrm')

count = merge(count_csrm, count_harden, by=c('table', 'variable'))
ggplot(data=count) + 
    geom_abline(intercept=0, slope=1, linetype='dashed', col='grey') +
    geom_point(aes(x=harden, y=csrm))
