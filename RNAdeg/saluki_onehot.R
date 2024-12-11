### PURPOSE OF THIS SCRIPT
## One-hot encode sequence as Saluki did


# Load dependencies ------------------------------------------------------------

library(data.table)
library(readr)
library(rtracklayer)
library(GenomicFeatures)
library(Biostrings)
library(dplyr)

# Full Saluki one-hot encoding -------------------------------------------------

features <- fread("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/DataTables/RNAdeg_data_model_features.csv")
gtf <- rtracklayer::import("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/Annotation_gamut_analyses/Annotations/mix_trimmed.gtf")
gtf_df <- as_tibble(gtf)


gtf_df %>%
  mutate(exon_number = as.numeric(exon_number)) %>%
  group_by(transcript_id) %>%
  filter(type == "exon" &
          ( ((exon_number < max(exon_number)) & strand == "+" ) |
            ((exon_number > min(exon_number)) & strand == "-")  )) %>%
  mutate(fivepss = ifelse(
    strand == "+", end - min(start),
    abs(start - max(end))
  ))

get_5pss <- function(gtf){

}

num_to_one_hot = function (x, bits) {
  diag(1L, bits)[, x]
}

OHE <- function(seq){

  strsplit(seq, '') |>
    unlist() |>
    match(c('A', 'C', 'G', 'T')) |>
    num_to_one_hot(bits = 4L)

}

OHE_phase <- function(cds){

  cds_as_nums <- seq(from = 1, to = nchar(cds)) %% 3
  cds_as_nums <- ifelse(cds_as_nums == 1, 1, 0)
  return(cds_as_nums)

}

OHE_saluki <- function(fivep, cds, threep){

  fivep_ohe <- OHE(fivep)
  cds_ohe <- OHE(cds)
  threep_ohe <- OHE(threep)
  phase_ohe <- c(rep(0, times = nchar(fivep)),
                 OHE_phase(cds),
                 rep(0, times = nchar(threep)))

  final_OHE <- do.call(cbind,
                       list(fivep_ohe,
                       cds_ohe,
                       threep_ohe)) %>%
    rbind(phase_ohe)

}

OHE_test <- OHE_saluki(features$fiveputr_seq[1],
                       cds = features$CDS_seq[1],
                       threep = features$threeputr_seq[1])

