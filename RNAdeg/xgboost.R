### PURPOSE OF THIS SCRIPT
## Fit XGBoost model to isoform degradation information

# Load dependencies ------------------------------------------------------------

library(caret)
library(xgboost)
library(dplyr)
library(readr)
library(ggplot2)


# Fit XGBoost model ------------------------------------------------------------

### Load data and split into train and test
RNAdeg_data <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_dataset.csv")

train <- RNAdeg_data %>%
  dplyr::filter(!(seqnames %in% c("chr1", "chr22"))) %>%
  dplyr::select(-seqnames,-transcript_id, -gene_id,
                -gene_name, -old_gene_id, -strand,
                -cds) %>%
  dplyr::mutate(
    nmd = ifelse(
      nmd == "no", 0,
      1
    )
  )

train_list <- list(
  data = train %>% dplyr::select(-log_kdeg,
                                 -log_ksyn,
                                 -avg_reads,
                                 -kdeg,
                                 -ksyn),
  label = train %>% dplyr::select(log_kdeg)
)


test <- RNAdeg_data %>%
  dplyr::filter(seqnames %in% c("chr1", "chr22")) %>%
  dplyr::select(-seqnames,-transcript_id, -gene_id,
                -gene_name, -old_gene_id, -strand,
                -cds) %>%
  dplyr::mutate(
    nmd = ifelse(
      nmd == "no", 0,
      1
    )
  )


test_list <- list(
  data = test %>% dplyr::select(-log_kdeg,
                                 -log_ksyn,
                                 -avg_reads,
                                 -kdeg,
                                 -ksyn),
  label = test %>% dplyr::select(log_kdeg)
)



### Fit XGBoost
bst <- xgboost(
  data = as.matrix(train_list$data),
  label = as.matrix(train_list$label),
  max.depth = 10,
  nthread = 8,
  nrounds = 100,
  objective = "reg:squarederror"
)

train_pred <- predict(bst,
                as.matrix(train_list$data))


pred <- predict(bst,
                as.matrix(test_list$data))

rmse <- sqrt(mean((pred - test_list$label$log_kdeg)^2))

print(paste("Test RMSE = ", rmse))

plot(test_list$label$log_kdeg,
     pred)
abline(0,1)

plot(train_list$label$log_kdeg,
     train_pred)
abline(0,1)

