### PURPOSE OF THIS SCRIPT
## Fit XGBoost model to isoform degradation information

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
