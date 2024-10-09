### PURPOSE OF THIS SCRIPT
## Check out general trends in RNAdeg data


# Load dependencies ------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(readr)
library(MASS)

# Get density of points in 2 dimensions.
# @param x A numeric vector.
# @param y A numeric vector.
# @param n Create a square n by n grid to compute density.
# @return The density within each square.
get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}



# Assess transcript isoform stability trends -----------------------------------

isoform_data <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_dataset.csv")


# Some plot characteristics
point_size = 0.5


##### Length vs. stability #####

glen <- isoform_data %>%
  mutate(
    density = get_density(
      x = log10(effective_length),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log10(effective_length),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log10(length)") +
  ylab("log(kdeg)")

glen



##### Number of exons vs. stability #####

gex <- isoform_data %>%
  mutate(
    density = get_density(
      x = log(num_exons),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(num_exons),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(# of exons)") +
  ylab("log(kdeg)")


gex


gex_n <- isoform_data %>%
  mutate(
    density = get_density(
      x = num_exons,
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = num_exons,
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(# of exons)") +
  ylab("log(kdeg)")


gex_n


gex_nz <- isoform_data %>%
  mutate(
    density = get_density(
      x = num_exons,
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = num_exons,
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(# of exons)") +
  ylab("log(kdeg)") +
  coord_cartesian(xlim = c(0, 50))


gex_nz

##### ksyn vs. stability #####

gksyn <- isoform_data %>%
  mutate(
    density = get_density(
      x = log_ksyn,
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log_ksyn,
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)")

gksyn


# Is correlation a product of uncertainty
gksyn_u <- isoform_data %>%
  filter(avg_lkd_se < 1) %>%
  arrange(-avg_lkd_se) %>%
  ggplot(aes(x = log_ksyn,
             y = log_kdeg,
             color = log(avg_lkd_se))) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)")

gksyn_u


# Is correlation a product of coverage
gksyn_r <- isoform_data %>%
  filter(avg_lkd_se < 1) %>%
  arrange(-avg_lkd_se) %>%
  ggplot(aes(x = log_ksyn,
             y = log_kdeg,
             color = log10(avg_reads))) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)")

gksyn_r


##### 3'UTR length vs. stability #####

g3p <- isoform_data %>%
  mutate(
    density = get_density(
      x = log(`3'UTR_length` + 1),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(`3'UTR_length` + 1),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(3' UTR length + 1)") +
  ylab("log(kdeg)")

g3p


# Is read coverage a confounder?
g3p_hr <- isoform_data %>%
  dplyr::filter(avg_reads > 3000) %>%
  mutate(
    density = get_density(
      x = log(`3'UTR_length` + 1),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(`3'UTR_length` + 1),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(3' UTR length + 1)") +
  ylab("log(kdeg)")

g3p_hr


g3p_lr <- isoform_data %>%
  dplyr::filter(avg_reads < 3000 & avg_reads > 100) %>%
  mutate(
    density = get_density(
      x = log(`3'UTR_length` + 1),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(`3'UTR_length` + 1),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(3' UTR length + 1)") +
  ylab("log(kdeg)")

g3p_lr


g3p_r <- isoform_data %>%
  dplyr::arrange(-avg_reads) %>%
  ggplot(aes(x = log(`3'UTR_length` + 1),
             y = log_kdeg,
             color = log10(avg_reads))) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(3' UTR length + 1)") +
  ylab("log(kdeg)")

g3p_r


# Does UTR length also correlate with ksyn?

g3pks <- isoform_data %>%
  mutate(
    density = get_density(
      x = log(`3'UTR_length` + 1),
      y = log_ksyn,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(`3'UTR_length` + 1),
             y = log_ksyn,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(3' UTR length + 1)") +
  ylab("log(ksyn)")

g3pks


# How does correlation stratify as a function of NMD sensitivity?
g3pnmd <- isoform_data %>%
  mutate(NMD_both = (nmd == "yes") & EZbakR_nmd) %>%
  arrange(NMD_both) %>%
  ggplot(aes(x = log(`3'UTR_length` + 1),
             y = log_kdeg,
             color = NMD_both)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_manual(values = c('darkred', 'darkgray')) +
  xlab("log(3' UTR length + 1)") +
  ylab("log(kdeg)")

g3pnmd

sum(isoform_data$EZbakR_nmd)
sum(isoform_data$EZbakR_nmd & (isoform_data$nmd == "yes"))



##### 5'UTR length vs. stability #####

g5p <- isoform_data %>%
  mutate(
    fiveprimeUTR_lngth = effective_length - `3'UTR_length`
  ) %>%
  filter(fiveprimeUTR_lngth > 0) %>%
  mutate(
    density = get_density(
      x = log(fiveprimeUTR_lngth + 1),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(fiveprimeUTR_lngth + 1),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(5' UTR length + 1)") +
  ylab("log(kdeg)")

g5p


# Is my UTR length estimation strategy just jank?

g5p3p <- isoform_data %>%
  mutate(
    density = get_density(
      x = log(threeprimeUTR_lngth + 1),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(threeprimeUTR_lngth + 1),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(Isaac's 3' UTR length + 1)") +
  ylab("log(kdeg)")

g5p3p


##### TPM vs. stability ######

gtpm <- isoform_data %>%
  mutate(
    density = get_density(
      x = log(avg_TPM),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(avg_TPM),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(TPM)") +
  ylab("log(kdeg)")

gtpm


gtpm_r <- isoform_data %>%
  arrange(-avg_reads) %>%
  ggplot(aes(x = log(avg_TPM),
             y = log_kdeg,
             color = log10(avg_reads))) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(TPM)") +
  ylab("log(kdeg)")

gtpm_r



##### Save all plots

setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/DC_IWV_2024_10_08/Figures/")

# Exon count investigations
ggsave(filename = "NumExon_vs_kdeg.png",
       plot = gex,
       height = 3,
       width = 4)
ggsave(filename = "NumExon_vs_kdeg_natscale.png",
       plot = gex_n,
       height = 3,
       width = 4)
ggsave(filename = "NumExon_vs_kdeg_natscalezoom.png",
       plot = gex_nz,
       height = 3,
       width = 4)

# 3'UTR investigations
ggsave(filename = "ThreePrimeUTR_vs_kdeg.png",
       plot = g3p,
       width = 4,
       height = 3)
ggsave(filename = "ThreePrimeUTR_vs_kdeg_highreadcount.png",
       plot = g3p_hr,
       width = 4,
       height = 3)
ggsave(filename = "ThreePrimeUTR_vs_kdeg_lowreadcount.png",
       plot = g3p_lr,
       width = 4,
       height = 3)
ggsave(filename = "ThreePrimeUTR_vs_kdeg_readcolor.png",
       plot = g3p_r,
       width = 4,
       height = 3)
ggsave(filename = "ThreePrimeUTR_vs_ksyn.png",
       plot = g3pks,
       width = 4,
       height = 3)
ggsave(filename = "ThreePrimeUTR_vs_ksyn_IsaacsCalc.png",
       plot = g5p3p,
       width = 4,
       height = 3)
ggsave(filename = "ThreePrimeUTR_vs_ksyn_NMDcolor.png",
       plot = g3pnmd,
       width = 4,
       height = 3)


# 5'UTR investigations
ggsave(filename = "FivePrimeUTR_vs_kdeg.png",
       plot = g5p,
       width = 4,
       height = 3)

# ksyn investigations
ggsave(filename = "ksyn_vs_kdeg.png",
       plot = gksyn,
       width = 4,
       height = 3)
ggsave(filename = "ksyn_vs_kdeg_readcolor.png",
       plot = gksyn_r,
       width = 4,
       height = 3)
ggsave(filename = "ksyn_vs_kdeg_secolor.png",
       plot = gksyn_u,
       width = 4,
       height = 3)

# length investigations
ggsave(filename = "length_vs_kdeg.png",
       plot = glen,
       width = 4,
       height = 3)

# TPM investigations
ggsave(filename = "TPM_vs_kdeg.png",
       plot = gtpm,
       width = 4,
       height = 3)

# Assess gene-wise stability trends --------------------------------------------

gene_data <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/RNAdeg_genewise_dataset.csv")


# Some plot characteristics
point_size <- 0.75


##### Length vs. kdeg #####

gglen <- gene_data %>%
  mutate(
    density = get_density(
      x = log10(exonic_length),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log10(exonic_length),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log10(exonic length)") +
  ylab("log(kdeg)")

gglen


gglen_u <- gene_data %>%
  arrange(-avg_lkd_se) %>%
  ggplot(aes(x = log10(exonic_length),
             y = log_kdeg,
             color = log(avg_lkd_se))) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log10(exonic length)") +
  ylab("log(kdeg)")

gglen_u


##### ksyn vs. kdeg #####

ggks <- gene_data %>%
  mutate(
    density = get_density(
      x = log_ksyn,
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log_ksyn,
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(ksyn)") +
  ylab("log(kdeg)") +
  coord_cartesian(
    xlim = c(0, 10)
  )

ggks


##### RPK vs. kdeg #####

ggrpk <- gene_data %>%
  mutate(
    density = get_density(
      x = log10(avg_RPK),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log10(avg_RPK),
             y = log_kdeg,
             color = density)) +
  geom_point(size = point_size) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log10(RPK)") +
  ylab("log(kdeg)")

ggrpk


##### Save all plots

setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/DC_IWV_2024_10_08/Figures/")
ggsave(filename = "Genewise_length_vs_kdeg.png",
       plot = gglen,
       width = 4,
       heigh = 3)
ggsave(filename = "Genewise_length_vs_kdeg_secolor.png",
       plot = gglen_u,
       width = 4,
       heigh = 3)
ggsave(filename = "Genewise_ksyn_vs_kdeg.png",
       plot = ggks,
       width = 4,
       heigh = 3)
ggsave(filename = "Genewise_RPK_vs_kdeg.png",
       plot = ggrpk,
       width = 4,
       heigh = 3)
