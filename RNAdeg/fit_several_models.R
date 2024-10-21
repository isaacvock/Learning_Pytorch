### PURPOSE OF THIS SCRIPT
## Fit several models to RNAdeg data, namely:
# 1) Multi-linear regression
# 2) XGBoost
# 3) Random forest
# 4) Something else? Fitting NN in Pytorch, but maybe I can also do something
# with MLPs in R?


# Load dependencies ------------------------------------------------------------

library(caret)
library(xgboost)
library(dplyr)
library(readr)
library(ggplot2)
library(MASS)

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}


# Make common processed feature table ------------------------------------------

feature_table <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_dataset.csv")

colnames(feature_table)

feature_table_processed <- feature_table  %>%
  dplyr::filter(avg_lkd_se < 0.25) %>% # 90%
  dplyr::mutate(NMD_both = EZbakR_nmd & (nmd == "yes"),
                log10_avg_TPM = log10(avg_TPM),
                log10_avg_reads = log10(avg_reads),
                log10_length = log10(effective_length),
                log10_numexons = log10(num_exons),
                log10_3primeUTR = log10(`3'UTR_length` + 1),
                log10_5primeUTR = log10(fiveprimeUTR_lngth + 1)) %>%
  dplyr::select(seqnames, old_gene_id, transcript_id, NMD_both,
                log10_3primeUTR, log10_avg_TPM, log10_avg_reads,
                log_ksyn, log_kdeg, log10_5primeUTR, log10_length,
                log10_numexons, avg_lkd_se) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(NMD_both = as.numeric(NMD_both)) %>%
  dplyr::mutate(NMD_both = (NMD_both - mean(NMD_both)) / sd(NMD_both),
                log10_avg_TPM = (log10_avg_TPM - mean(log10_avg_TPM)) / sd(log10_avg_TPM),
                log10_avg_reads = (log10_avg_reads - mean(log10_avg_reads)) / sd(log10_avg_reads),
                log10_length = (log10_length - mean(log10_length))/sd(log10_length),
                log10_numexons = (log10_numexons - mean(log10_numexons)) / sd(log10_numexons),
                log10_3primeUTR = (log10_3primeUTR - mean(log10_3primeUTR)) / sd(log10_3primeUTR),
                log10_5primeUTR = (log10_5primeUTR - mean(log10_5primeUTR)) / sd(log10_5primeUTR),
                log_kdeg = (log_kdeg - mean(log_kdeg))/sd(log_kdeg),
                log_ksyn = (log_ksyn - mean(log_ksyn))/sd(log_ksyn))


write_csv(feature_table_processed,
          "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_feature_table.csv")

# Multi-linear regression ------------------------------------------------------

# Load data
ft <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_feature_table.csv")

# Train-test split
ft_train <- ft %>%
  filter(!(seqnames %in% c("chr1", "chr22")))

ft_test <- ft %>%
  filter((seqnames %in% c("chr1", "chr22")))


# Fit model
lmfit <- lm(log_kdeg ~ NMD_both + log10_3primeUTR +
              log_ksyn + log10_5primeUTR + log10_length +
              log10_numexons - 1,
            data = ft_train)


##### MODEL PERFORMANCE #####

# Assess fit
test_pred <- predict(lmfit, ft_test)

test_rmse <- sqrt(mean((test_pred - ft_test$log_kdeg)^2))

print(paste0("Testing RMSE is: ", test_rmse))

plot(ft_test$log_kdeg,
     test_pred)


train_pred <- predict(lmfit, ft_train)

train_rmse <- sqrt(mean((train_pred - ft_train$log_kdeg)^2))

print(paste0("Training RMSE is: ", test_rmse))
# 0.783

plot(ft_train$log_kdeg,
     train_pred)

cor(train_pred,
    ft_train$log_kdeg)
  # 0.74

##### PLOT ACCURACY #####


lmfit_df <- tibble(
  prediction = test_pred,
  truth = ft_test$log_kdeg
)


lmfit_train_df <- tibble(
  prediction = train_pred,
  truth = ft_train$log_kdeg
)

gtest_lm <- lmfit_df %>%
  mutate(
    density = get_density(
      x = prediction,
      y = truth,
      n = 200
    )
  ) %>%
  ggplot(
    aes(x = prediction,
        y = truth,
        color = density)
  ) +
  geom_point() +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(kdeg) prediction") +
  ylab("log(kdeg) truth") +
  geom_abline(slope = 1,
              intercept = 0,
              color = 'darkred',
              linetype = 'dotted',
              linewidth = 0.75)

gtrain_lm <- lmfit_train_df %>%
  mutate(
    density = get_density(
      x = prediction,
      y = truth,
      n = 200
    )
  ) %>%
  ggplot(
    aes(x = prediction,
        y = truth,
        color = density)
  ) +
  geom_point() +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(kdeg) prediction") +
  ylab("log(kdeg) truth") +
  geom_abline(slope = 1,
              intercept = 0,
              color = 'darkred',
              linetype = 'dotted',
              linewidth = 0.75)


gtrain_lm


setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/DC_IWV_2024_10_08/Figures/")
ggsave(filename = "LinModel_test_accuracy.png",
       plot = gtest_lm,
       width = 4,
       height = 3)
ggsave(filename = "LinModel_train_accuracy.png",
       plot = gtrain_lm,
       width = 4,
       height = 3)
ggsave(filename = "LinModel_coefficients.png",
       plot = gcoef,
       width = 4,
       height = 4)


##### PLOT FEATURE IMPORTANCE #####

### Coefficients

lm_coef <- coefficients(lmfit)
coef_df <- tibble(
  value = lm_coef,
  feature = names(lm_coef)
)

gcoef <- coef_df %>%
  as_tibble() %>%
  mutate(feature = case_when(
    feature == "log10_3primeUTR" ~ "3prime UTR length",
    feature == "log_ksyn" ~ "ksyn",
    feature == "log10_numexons" ~ "# of exons",
    feature == "log10_5primeUTR" ~ "5prime UTR length",
    feature == "NMD_both" ~ "NMD target",
    feature == "log10_length" ~ "Length"
  )) %>%
  mutate(feature = factor(feature,
                          levels = feature[order(-value)])) %>%
  ggplot(aes(x = feature,
             y = value)) +
  geom_bar(stat = "identity") +
  theme_classic() +
  xlab("Feature") +
  ylab("Coefficient") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

gcoef



setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/DC_IWV_2024_10_08/Figures/")
ggsave(filename = "XGBoost_test_accuracy.png",
       plot = gtest,
       width = 4,
       height = 3)

# XGBoost ----------------------------------------------------------------------


# Load data
ft <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_feature_table.csv")

features_to_use <- c("NMD_both",
                     "log10_3primeUTR",
                     "log_ksyn",
                     "log10_5primeUTR",
                     "log10_length",
                     "log10_numexons")


# Train-test split
ft_train <- ft %>%
  filter(!(seqnames %in% c("chr1", "chr22")))

ft_test <- ft %>%
  filter((seqnames %in% c("chr1", "chr22")))


ft_train_hyper <- ft_train %>%
  filter(seqnames != "chr2")
ft_test_hyper <- ft_train %>%
  filter(seqnames == "chr2")


train_list <- list(
  data = ft_train %>% dplyr::select(!!features_to_use),
  label = ft_train %>% dplyr::select(log_kdeg)
)


test_list <- list(
  data = ft_test %>% dplyr::select(!!features_to_use),
  label = ft_test %>% dplyr::select(log_kdeg)
)


train_hyper_list <- list(
  data = ft_train_hyper %>% dplyr::select(!!features_to_use),
  label = ft_train_hyper %>% dplyr::select(log_kdeg)
)

test_hyper_list <- list(
  data = ft_test_hyper %>% dplyr::select(!!features_to_use),
  label = ft_test_hyper %>% dplyr::select(log_kdeg)
)


### Find best XGBoost parameters if necessary

# depths <- c(3, 5, 7, 10, 15)
# rounds <- c(5, 10, 15, 20, 25)
# etas <- c(0.1, 0.3, 0.5, 0.7, 0.9)
# subsamps <- c(0.3, 0.5, 0.7, 0.9, 1)
#
# nfits <- length(depths) *
#   length(rounds) *
#   length(etas) *
#   length(subsamps)
#
# rmse_df <- tibble()
# count <- 0
# for(d in depths){
#   for(r in rounds){
#     for(e in etas){
#      for(s in subsamps){
#        bst <- xgboost(
#          data = as.matrix(train_hyper_list$data),
#          label = as.matrix(train_hyper_list$label),
#          max.depth = d,
#          nthread = 8,
#          nrounds = r,
#          eta = e,
#          subsample = s,
#          objective = "reg:squarederror",
#          verbose = 0
#        )
#
#        pred <- predict(bst,
#                        as.matrix(test_hyper_list$data))
#
#        rmse <- sqrt(mean((pred - test_hyper_list$label$log_kdeg)^2))
#
#        rmse_df <- rmse_df %>%
#          bind_rows(tibble(
#            depth = d,
#            nrounds = r,
#            eta = e,
#            subsample = s,
#            RMSE = rmse
#          ))
#
#        count <- count + 1
#        print(paste0((count*100) / nfits, "% done"))
#
#      }
#     }
#   }
# }


# Best parameters from finetuning on testing set:
  # depth = 7
  # rounds = 25
  # eta = 0.1
  # subsample = 0.7

# Best parameters from finetuning on validation set:
  # depth = 7
  # rounds = 25
  # eta = 0.1
  # subsample = 0.7

# Pretty cool that best parameters are stable across
# valdiation sets

### Fit XGBoost
bst <- xgboost(
  data = as.matrix(train_list$data),
  label = as.matrix(train_list$label),
  max.depth = 7,
  nthread = 8,
  nrounds = 25,
  subsample = 0.7,
  eta = 0.1,
  objective = "reg:squarederror"
)

train_pred <- predict(bst,
                      as.matrix(train_list$data))


pred <- predict(bst,
                as.matrix(test_list$data))

rmse <- sqrt(mean((pred - test_list$label$log_kdeg)^2))

print(paste("Test RMSE = ", rmse))
  # 0.733

cor(pred, test_list$label$log_kdeg)
  # 0.674


pred_df <- tibble(
  prediction = pred,
  truth = test_list$label$log_kdeg
)

pred_df_train <- tibble(
  prediction = train_pred,
  truth = train_list$label$log_kdeg
)


gtest <- pred_df %>%
  mutate(
    density = get_density(
      x = prediction,
      y = truth,
      n = 200
    )
  ) %>%
  ggplot(
    aes(x = prediction,
        y = truth,
        color = density)
  ) +
  geom_point() +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(kdeg) prediction") +
  ylab("log(kdeg) truth") +
  geom_abline(slope = 1,
              intercept = 0,
              color = 'darkred',
              linetype = 'dotted',
              linewidth = 0.75)


gtrain <- pred_df_train %>%
  mutate(
    density = get_density(
      x = prediction,
      y = truth,
      n = 200
    )
  ) %>%
  ggplot(
    aes(x = prediction,
        y = truth,
        color = density)
  ) +
  geom_point(size = 0.75) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(kdeg) prediction") +
  ylab("log(kdeg) truth") +
  geom_abline(slope = 1,
              intercept = 0,
              color = 'darkred',
              linetype = 'dotted',
              linewidth = 0.75)



importance_matrix <- xgb.importance(model = bst)

ggain <- importance_matrix %>%
  as_tibble() %>%
  mutate(Feature = case_when(
    Feature == "log10_3primeUTR" ~ "3prime UTR length",
    Feature == "log_ksyn" ~ "ksyn",
    Feature == "log10_numexons" ~ "# of exons",
    Feature == "log10_5primeUTR" ~ "5prime UTR length",
    Feature == "NMD_both" ~ "NMD target",
    Feature == "log10_length" ~ "Length"
  )) %>%
  mutate(Feature = factor(Feature,
                       levels = Feature[order(-Gain)])) %>%
  ggplot(aes(x = Feature,
             y = Gain)) +
  geom_bar(stat = "identity") +
  theme_classic() +
  xlab("Feature") +
  ylab("Gain") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


### Overfitting example for fun

bst_o <- xgboost(
  data = as.matrix(train_list$data),
  label = as.matrix(train_list$label),
  max.depth = 10,
  nthread = 8,
  nrounds = 100,
  subsample = 1,
  eta = 0.7,
  objective = "reg:squarederror"
)

train_pred_o <- predict(bst_o,
                      as.matrix(train_list$data))


pred_o <- predict(bst_o,
                as.matrix(test_list$data))



pred_df_o <- tibble(
  prediction = pred_o,
  truth = test_list$label$log_kdeg
)

pred_df_train_o <- tibble(
  prediction = train_pred_o,
  truth = train_list$label$log_kdeg
)


gtest_o <- pred_df_o %>%
  mutate(
    density = get_density(
      x = prediction,
      y = truth,
      n = 200
    )
  ) %>%
  ggplot(
    aes(x = prediction,
        y = truth,
        color = density)
  ) +
  geom_point() +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(kdeg) prediction") +
  ylab("log(kdeg) truth") +
  geom_abline(slope = 1,
              intercept = 0,
              color = 'darkred',
              linetype = 'dotted',
              linewidth = 0.75)


gtrain_o <- pred_df_train_o %>%
  mutate(
    density = get_density(
      x = prediction,
      y = truth,
      n = 200
    )
  ) %>%
  ggplot(
    aes(x = prediction,
        y = truth,
        color = density)
  ) +
  geom_point(size = 0.75) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(kdeg) prediction") +
  ylab("log(kdeg) truth") +
  geom_abline(slope = 1,
              intercept = 0,
              color = 'darkred',
              linetype = 'dotted',
              linewidth = 0.75)

gtrain_o
gtest_o


### Save plots
setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/DC_IWV_2024_10_08/Figures/")
ggsave(filename = "XGBoost_test_accuracy.png",
       plot = gtest,
       width = 4,
       height = 3)
ggsave(filename = "XGBoost_train_accuracy.png",
       plot = gtrain,
       width = 4,
       height = 3)
ggsave(filename = "XGBoost_feature_importance.png",
       plot = ggain,
       width = 4,
       height = 4)
ggsave(filename = "XGBoost_overfit_test_accuracy.png",
       plot = gtest_o,
       width = 4,
       height = 3)
ggsave(filename = "XGBoost_overfit_train_accuracy.png",
       plot = gtrain_o,
       width = 4,
       height = 3)
