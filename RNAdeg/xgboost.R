### PURPOSE OF THIS SCRIPT
## Fit XGBoost model to isoform degradation information

# Load dependencies ------------------------------------------------------------

library(caret)
library(xgboost)
library(dplyr)
library(readr)
library(ggplot2)
library(MASS)
library(stringr)

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}


# XGBoost model with sequence motifs included ----------------------------------


### Get sequence data
threeprimeutr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/threeprimeUTR_seqs.csv")

motif_counts <- threeprimeutr_seq %>%
  group_by(transcript_id) %>%
  summarise(
    ARE_count = str_count(seq, pattern = "TATTTAT"),
    Pumilio_count = str_count(seq, pattern = "TGTAAAT") +
      str_count(seq, pattern = "TGTATA"),
    CPA_count = str_count(seq, pattern = "AATAAA"),
    rand_count = str_count(seq, pattern = "ATGCATG"),
    GC_cont = EZbakR:::logit((str_count(seq, pattern = "G") +
      str_count(seq, pattern = "C") + 1) / (nchar(seq) + 1))
  ) %>%
  mutate(
    ARE_count = ifelse(
      ARE_count > 10,
      10,
      ARE_count
    ),
    Pumilio_count = ifelse(
      Pumilio_count > 10,
      10,
      Pumilio_count
    ),
    CPA_count = ifelse(
      CPA_count > 10,
      10,
      CPA_count
    ),
    rand_count = ifelse(
      rand_count > 10,
      10,
      rand_count
    )
  )


# Z-score normalize
motif_counts <- motif_counts %>%
  ungroup() %>%
  mutate(ARE_z = (ARE_count - mean(ARE_count)) / sd(ARE_count),
         Pumilio_z = (Pumilio_count - mean(Pumilio_count)) / sd(Pumilio_count),
         CPA_z = (CPA_count - mean(CPA_count)) / sd(CPA_count),
         rand_z = (rand_count - mean(rand_count)) / sd(rand_count),
         GC_z = (GC_cont - mean(GC_cont)) / sd(GC_cont))


# Load data
ft <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_feature_table.csv")

features_to_use <- c("NMD_both",
                     "log10_3primeUTR",
                     "log_ksyn",
                     "log10_5primeUTR",
                     "log10_length",
                     "log10_numexons",
                     "ARE_z",
                     "Pumilio_z",
                     "CPA_z",
                     "rand_z",
                     "GC_z")

# Filter out low confidence ish
ft <- ft %>%
  filter(avg_lkd_se < exp(-2))

### Combine motif and lower-res data
combined_ft <- ft %>%
  inner_join(motif_counts,
             by = "transcript_id")

# Train-test split
ft_train <- combined_ft %>%
  filter(!(seqnames %in% c("chr1", "chr22")))

ft_test <- combined_ft %>%
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

depths <- c(3, 5, 7, 10, 15)
rounds <- c(5, 10, 15, 20, 25)
etas <- c(0.1, 0.3, 0.5, 0.7, 0.9)
subsamps <- c(0.3, 0.5, 0.7, 0.9, 1)

nfits <- length(depths) *
  length(rounds) *
  length(etas) *
  length(subsamps)

rmse_df <- tibble()
count <- 0
for(d in depths){
  for(r in rounds){
    for(e in etas){
     for(s in subsamps){
       bst <- xgboost(
         data = as.matrix(train_hyper_list$data),
         label = as.matrix(train_hyper_list$label),
         max.depth = d,
         nthread = 8,
         nrounds = r,
         eta = e,
         subsample = s,
         objective = "reg:squarederror",
         verbose = 0
       )

       pred <- predict(bst,
                       as.matrix(test_hyper_list$data))

       rmse <- sqrt(mean((pred - test_hyper_list$label$log_kdeg)^2))

       rmse_df <- rmse_df %>%
         bind_rows(tibble(
           depth = d,
           nrounds = r,
           eta = e,
           subsample = s,
           RMSE = rmse
         ))

       count <- count + 1
       print(paste0((count*100) / nfits, "% done"))

     }
    }
  }
}


op <- rmse_df %>%
  filter(RMSE == min(RMSE))
# With motif data:
  # depth = 7
  # nrounds = 20
  # eta = 0.3
  # subsample = 0.9

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
  max.depth = op$depth,
  nthread = 8,
  nrounds = op$nrounds,
  subsample = op$subsample,
  eta = op$eta,
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
    Feature == "log10_length" ~ "Length",
    .default = Feature
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


ggain


plot(combined_ft$log_ksyn,
     combined_ft$log_kdeg)

setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/Hogg_IWV_2024_10_21/")
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


plot(combined_ft$GC_z,
     combined_ft$log_kdeg)

# Explore isoform stability trends ---------------------------------------------

### What it look like?
RNAdeg_data <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_dataset.csv")


# Length normalize ksyn
RNAdeg_data <- RNAdeg_data %>%
  mutate(ksyn = ksyn / (exonic_length/1000),
         log_ksyn = log(ksyn))



RNAdeg_data %>%
  mutate(density = get_density(
    x = log(ksyn / (exonic_length / 1000 )),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log(ksyn / (exonic_length / 1000 )),
           y = log_kdeg,
           color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)")


RNAdeg_data %>%
  dplyr::filter(nmd == "yes") %>%
  ggplot(aes(x = log(ksyn / (exonic_length / 1000 )),
             y = log_kdeg)) +
  geom_point(alpha = 0.1) +
  theme_classic() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)")

RNAdeg_data %>%
  dplyr::filter(nmd == "no") %>%
  ggplot(aes(x = log(ksyn / (exonic_length / 1000 )),
             y = log_kdeg)) +
  geom_point(alpha = 0.1) +
  theme_classic() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)")




NAdeg_data %>%
  filter(avg_reads > 50) %>%
  mutate(density = get_density(
    x = log10(avg_reads),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(avg_reads),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(reads)") +
  ylab("log(kdeg)")




RNAdeg_data %>%
  filter(avg_reads > 50) %>%
  mutate(density = get_density(
    x = log10(threeprimeUTR_lngth + 1),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(threeprimeUTR_lngth + 1),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(3' UTR length)") +
  ylab("log(kdeg)")



g5UTR <- RNAdeg_data %>%
  filter(avg_reads > 50) %>%
  mutate(density = get_density(
    x = log10(fiveprimeUTR_lngth + 1),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(fiveprimeUTR_lngth + 1),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(5' UTR length)") +
  ylab("log(kdeg)")



RNAdeg_data %>%
  group_by(gene_id) %>%
  summarise(niso = length(unique(transcript_id))) %>%
  ggplot(aes(x = factor(niso))) +
  geom_histogram(stat = "count") +
  theme_classic() +
  xlab("Number of isoforms") +
  ylab("Number of genes")



g3UTRnoNMD <- RNAdeg_data %>%
  filter(avg_reads > 50 & nmd == "no") %>%
  mutate(density = get_density(
    x = log10(`3'UTR_length` + 1),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(`3'UTR_length` + 1),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(3' UTR length)") +
  ylab("log(kdeg)")




g3UTRNMD <- RNAdeg_data %>%
  filter(avg_reads > 50 & nmd == "yes") %>%
  mutate(density = get_density(
    x = log10(`3'UTR_length` + 1),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(`3'UTR_length` + 1),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(3' UTR length)") +
  ylab("log(kdeg)")



g3UTR <- RNAdeg_data %>%
  filter(avg_reads > 50) %>%
  mutate(density = get_density(
    x = log10(`3'UTR_length` + 1),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(`3'UTR_length` + 1),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(3' UTR length)") +
  ylab("log(kdeg)")



RNAdeg_data %>%
  filter(avg_reads > 50 & nmd != "yes") %>%
  mutate(density = get_density(
    x = log10(avg_length + 1),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(avg_length + 1),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(avg. exon length + 1)") +
  ylab("log(kdeg)")


RNAdeg_data %>%
  filter(avg_reads > 50 & nmd != "yes") %>%
  mutate(density = get_density(
    x = log10(avg_reads / (exonic_length / 1000)),
    y = log_kdeg,
    n = 200
  )) %>%
  ggplot(aes(x = log10(avg_reads / (exonic_length / 1000)),
             y = log_kdeg,
             color = density)) +
  geom_point() +
  theme_classic() +
  geom_smooth(method='lm', formula= y~x) +
  scale_color_viridis_c() +
  xlab("log10(avg. RPK)") +
  ylab("log(kdeg)")




RNAdeg_data %>%
  filter(avg_reads > 50) %>%
  ggplot(aes(x = log_kdeg,
             color = nmd)) +
  geom_density() +
  theme_classic() +
  scale_color_manual(values = c('darkgray', 'darkred'))
  xlab("log(kdeg)") +
  ylab("density")


setwd("C:/Users/isaac/Documents/Simon_Lab/Presentations/2024_10_07_IBDD_RIP/Figures/")
ggsave(filename = "Isokdeg_vs_3primeUTR.png",
       plot = g3UTR,
       width = 5,
       height = 3.5)
ggsave(filename = "Isokdeg_vs_3primeUTR_noNMD.png",
       plot = g3UTRnoNMD,
       width = 5,
       height = 3.5)
ggsave(filename = "Isokdeg_vs_3primeUTR_NMD.png",
       plot = g3UTRNMD,
       width = 5,
       height = 3.5)
ggsave(filename = "Isokdeg_vs_5primeUTR.png",
       plot = g5UTR,
       width = 5,
       height = 3.5)

# Fit XGBoost model ------------------------------------------------------------

### Load data and split into train and test
RNAdeg_data <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_dataset.csv")

# Length normalize ksyn
RNAdeg_data <- RNAdeg_data %>%
  mutate(ksyn = ksyn / (exonic_length/1000),
         log_ksyn = log(ksyn),
         log5primeUTR = log(fiveprimeUTR_lngth + 1),
         log_avglen = log(avg_length + 1),
         log3primeUTR = log(`3'UTR_length` + 1),
         log_reads = log(avg_reads + 1))


train <- RNAdeg_data %>%
  dplyr::filter(!(seqnames %in% c("chr1", "chr22"))) %>%
  dplyr::select(log_avglen, log5primeUTR, log_reads,
                log3primeUTR, log_ksyn, stop_to_lastEJ, num_of_downEJs,
                log_kdeg)



train_list <- list(
  data = train %>% dplyr::select(-log_kdeg),
  label = train %>% dplyr::select(log_kdeg)
)


test <- RNAdeg_data %>%
  dplyr::filter(seqnames %in% c("chr1", "chr22")) %>%
  dplyr::select(log_avglen, log5primeUTR, log_reads,
                log3primeUTR, log_ksyn, stop_to_lastEJ, num_of_downEJs,
                log_kdeg)


test_list <- list(
  data = test %>% dplyr::select(-log_kdeg),
  label = test %>% dplyr::select(log_kdeg)
)



### Fit XGBoost
bst <- xgboost(
  data = as.matrix(train_list$data),
  label = as.matrix(train_list$label),
  max.depth = 10,
  nthread = 8,
  nrounds = 25,
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



importance_matrix <- xgb.importance(model = bst)

importance_matrix
