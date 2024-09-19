### PURPOSE OF THIS SCRIPT
## Get features to train ML models on from annotation and
## genome fasta file.

# Load dependencies ------------------------------------------------------------

library(rtracklayer)
library(Biostrings)
library(dplyr)
library(readr)

# Get a bunch of relevant features from GTF ------------------------------------



##### Get features from the factR2 annotation
factR_GR <- rtracklayer::import(
  "G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/longread_justin/LRSR_f05_factR2/factR2_start_and_stop.gtf"
)
gtf_df <- as_tibble(factR_GR)

exonic_features <- gtf_df %>%
  dplyr::filter(type == "exon") %>%
  dplyr::group_by(seqnames, transcript_id, gene_id, gene_name, old_gene_id) %>%
  dplyr::summarise(
    total_length = sum(width),
    avg_length = total_length/dplyr::n(),
    max_length = max(width),
    min_length = min(width),
    num_exons = length(unique(exon_number))
  )


UTR_features <- gtf_df %>%
  dplyr::filter(strand != "*") %>%
  dplyr::group_by(seqnames, transcript_id, gene_id, gene_name, old_gene_id, strand) %>%
  filter(sum(type == "CDS") > 0) %>%
  dplyr::summarise(
    fiveprimeUTR_lngth = case_when(
      all(strand == "+") ~ unique(min(start[type == "start_codon"]) - start[type == "transcript"]),
      all(strand == "-") ~ unique(end[type == "transcript"] - max(start[type == "start_codon"]))
    ),
    threeprimeUTR_lngth = case_when(
      all(strand == "-") ~ unique(max(end[type == "stop_codon"]) - start[type == "transcript"]),
      all(strand == "+") ~ unique(end[type == "transcript"] - min(end[type == "stop_codon"]))
    )
  )


##### Get NMD features

NMD_status <- fread("G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/longread_justin/LRSR_f05_factR2/factR2_transcript.tsv")
NMD_status <- NMD_status %>%
  dplyr::select(-V1)

NMD_status <- NMD_status %>%
  dplyr::rename(
    exonic_length = width
  )

NMD_status <- NMD_status %>%
  dplyr::select(-novel, -is_NMD, -PTC_coord)


##### Combine everything
combined_table <- dplyr::inner_join(
  exonic_features,
  UTR_features %>% dplyr::select(-strand),
  by = c("seqnames", "transcript_id", "gene_id",
         "gene_name", "old_gene_id")
) %>%
  dplyr::inner_join(
    NMD_status,
    by = c("transcript_id", "gene_id",
           "gene_name")
  )


combined_table


##### Add isoform stabilities

ezbdo <- readRDS("G:/Shared drives/Matthew_Simon/IWV/Hogg_lab/EZbakRFits/jmdata_LRSR_f05/Full_EZbakRFit_withgenewide.rds")
isoforms <- EZget(ezbdo,
                  type = "kinetics",
                  features = "transcript_id",
                  exactMatch = FALSE) %>%
  dplyr::filter(grepl("DMSO", sample)) %>%
  dplyr::group_by(
    XF, transcript_id
  ) %>%
  dplyr::summarise(
    log_kdeg = mean(log_kdeg),
    log_ksyn = mean(log_ksyn),
    avg_reads = mean(n)
  ) %>%
  dplyr::mutate(
    kdeg = exp(log_kdeg),
    ksyn = exp(log_ksyn)
  ) %>%
  dplyr::filter(avg_reads > 25) %>%
  dplyr::rename(
    old_gene_id = XF
  )

combined_table <- combined_table %>%
  inner_join(isoforms,
             by = c("old_gene_id",
                    "transcript_id"))

setwd("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/")
write_csv(combined_table,
          file = "RNAdeg_dataset.csv")

combined_table
