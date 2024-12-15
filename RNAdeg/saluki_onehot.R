### PURPOSE OF THIS SCRIPT
## One-hot encode sequence as Saluki did


# Load dependencies ------------------------------------------------------------

library(data.table)
library(readr)
library(rtracklayer)
library(GenomicFeatures)
library(Biostrings)
library(dplyr)
library(stringr)


# Full Saluki one-hot encoding -------------------------------------------------

features <- fread("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/DataTables/RNAdeg_data_model_features.csv")


features %>% dplyr::count(gene_id) %>% dplyr::count(n)

length(unique(TSSs))


gtf <- rtracklayer::import("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/Annotation_gamut_analyses/Annotations/mix_trimmed.gtf")

fivepss_df <- tibble()
txs <- unique(gtf$transcript_id)
for(t in unique(gtf$transcript_id)){

  exons <- gtf[gtf$transcript_id == t & gtf$type == "exon"]

  if(all(strand(exons) == "+")){

    widths <- width(exons)
    positions <- cumsum(widths)
    fivepss <- positions[1:(length(positions) - 1)]

  }else{

    widths <- width(exons)
    positions <- cumsum(widths)
    positions <- abs(positions - positions[length(positions)]) + 1
    fivepss <- positions[(length(positions)-1):1]


  }

  fivepss_df <- bind_rows(
    fivepss_df,
    tibble(transcript_id = t,
           fivepss = fivepss)
  )

  print(paste0((which(txs == t) / length(txs))*100, "% done"))

}


write_csv(fivepss_df,
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimed_fivepss_indices.csv")



widths <- width(test_exons)
positions <- cumsum(widths)
fivepss <- positions[1:(length(positions) - 1)]

txdb <- makeTxDbFromGRanges(gtf)

gtf_tx <- exonsBy(txdb, by = "tx", use.names = TRUE)
gtf_tx$MSTRG.2.1

gtf_df <- as_tibble(gtf)

# Split by transcript
transcripts <- split(gtf, gtf$transcript_id)

# Reduce the ranges within each transcript
reduced_transcripts <- reduce(transcripts)

ss_df <- gtf_df %>%
  filter(type == "exon") %>%
  mutate(exon_number = as.numeric(exon_number)) %>%
  dplyr::group_by(transcript_id) %>%
  mutate(
    start = min(start)
  )

ss_df

# 1 3
# 10 14
# 20 22
# 100 124
#
# 1 3
# 4 8
# 9 11
# 12 36

ss_df %>%
  filter(transcript_id == "MSTRG.11206.1") %>%
  dplyr::select(strand, fivepss, exon_number, exonic_start, exonic_end, start, end, exon_number)

ss_table <- ss_df %>%
  dplyr::select(transcript_id, fivepss)


write_csv(ss_table,
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/fivep_splice_sites.csv")


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

OHE_saluki <- function(fivep, cds, threep, splices,
                       max_nt = 12288){

  fivep_ohe <- OHE(fivep)
  cds_ohe <- OHE(cds)
  threep_ohe <- OHE(threep)
  phase_ohe <- c(rep(0, times = nchar(fivep)),
                 OHE_phase(cds),
                 rep(0, times = nchar(threep)))
  splice_ohe <- seq(from = 1, to = nchar(fivep) + nchar(cds) + nchar(threep))
  splice_ohe <- ifelse(splice_ohe %in% splices,
                       1, 0)


  final_OHE <- do.call(cbind,
                       list(fivep_ohe,
                       cds_ohe,
                       threep_ohe)) %>%
    rbind(phase_ohe) %>%
    rbind(splice_ohe)

  # Pad or truncate
  if(ncol(final_OHE) < max_nt){

    cols_to_add <- max_nt - ncol(final_OHE)

    final_OHE <- final_OHE %>%
      cbind(matrix(0, nrow = 6, ncol = cols_to_add))


  }else if(ncol(final_OHE) > max_nt){

    final_OHE <- final_OHE[,1:max_nt]

  }

  return(final_OHE)

}

features_to_ohe <- features %>%
  dplyr::select(fiveputr_seq,
                CDS_seq,
                threeputr_seq)


OHEs <- pmap(list(f = fivep_seqs,
               c = CDS_seqs,
               t = threep_seqs),
               function(f, c, t, s){
                 OHE_saluki(f, c, t, s)
               })


OHE_test <- OHE_saluki(features$fiveputr_seq[1],
                       cds = features$CDS_seq[1],
                       threep = features$threeputr_seq[1],
                       ss_df$fivepss[ss_df$transcript_id == ss_df$transcript_id[1]])

