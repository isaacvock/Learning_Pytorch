### PURPOSE OF THIS SCRIPT
## Use functions in processing_functions.R to process EZbakR and factR2
## analyses to generate tables of features to train ML models on.

# Load dependencies ------------------------------------------------------------

library(GenomicFeatures)
library(Biostrings)
library(rtracklayer)
library(dplyr)
library(readr)
library(ggplot2)
library(data.table)
library(EZbakR)

library(BSgenome.Hsapiens.UCSC.hg38)
library(stringr)

source("C:/Users/isaac/Documents/ML_pytorch/Scripts/RNAdeg/processing_functions.R")

# Trimmed mix annotation from the gamut ----------------------------------------

### Curate input
mix_gtf <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2.gtf")
isoforms_to_keep <- read_csv(
  "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/isoforms_to_keep.csv"
)

# Filter rsome stuff
mcols(mix_gtf)$ID <- NULL
mix_gtf <- mix_gtf[!grepl("_", seqnames(mix_gtf))]

transcript_file <- "C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2_transcript.tsv"
ezbdo <- readRDS("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/EZbakRFits/Mix_trimmed_EZbakRFit_withgenewide.rds")

RNAdeg_data <-  assemble_data(
  gtf = mix_gtf, factr_transcript_file = transcript_file,
  isoform_filter = isoforms_to_keep,
  ezbdo = ezbdo,
  outdir = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered"
)

clean_table <- clean_feature_table(RNAdeg_data,
                                   output = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/RNAdeg_feature_table.csv")

seqs <- get_sequencing_info(mix_gtf, outdir = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/")

threepcheck <- seqs$ThreePrimeUTR


# Sandbox ----------------------------------------------------------------------
mix_gtf <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2.gtf")
transcript_file <- "C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2_transcript.tsv"
ezbdo <- readRDS("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/EZbakRFits/Mix_trimmed_EZbakRFit_withgenewide.rds")


read_cutoff <- 25
TPM_cutoff <- 2
returnTable <- TRUE
filename <- "RNAdeg_dataset"
nmd_diff_cutoff <- -1
nmd_padj_cutoff <- 0.01
sample_pattern <- "DMSO"
gtf_feature_cols <- c("seqnames",
                     "transcript_id",
                     "gene_id",
                     "gene_name",
                     "old_gene_id",
                     "strand")

gtf_df <- as_tibble(mix_gtf)

exonic_features <- gtf_df %>%
  dplyr::filter(type == "exon") %>%
  dplyr::group_by(dplyr::across(dplyr::all_of(gtf_feature_cols))) %>%
  dplyr::summarise(
    total_length = sum(width),
    avg_length = total_length/dplyr::n(),
    max_length = max(width),
    min_length = min(width),
    num_exons = length(unique(exon_number))
  )


testout <- gtf_df %>%
  dplyr::filter(strand != "*") %>%
  dplyr::filter(type %in% c("transcript", "CDS")) %>%
  dplyr::group_by(dplyr::across(dplyr::all_of(gtf_feature_cols[gtf_feature_cols != "old_gene_id"]))) %>%
  filter(any(type == "CDS")) %>%
  summarise(CDS_start = min(start[type == "CDS"]),
         CDS_end = max(end[type == "CDS"]),
         transcript_start = min(start[type == "transcript"]),
         transcript_end = max(end[type == "transcript"]))


UTR_features <- gtf_df %>%
  dplyr::filter(strand != "*") %>%
  dplyr::filter(type %in% c("transcript", "CDS")) %>%
  dplyr::group_by(dplyr::across(dplyr::all_of(gtf_feature_cols[gtf_feature_cols != "old_gene_id"]))) %>%
  filter(any(type == "CDS")) %>%
  summarise(CDS_start = min(start[type == "CDS"]),
            CDS_end = max(end[type == "CDS"]),
            transcript_start = min(start[type == "transcript"]),
            transcript_end = max(end[type == "transcript"])) %>%
  dplyr::mutate(
    fiveprimeUTR_lngth = dplyr::case_when(
      strand == "+" ~ CDS_start - transcript_start,
      strand == "-" ~ transcript_end - CDS_end
    ),
    threeprimeUTR_lngth = dplyr::case_when(
      strand == "+" ~ transcript_end - CDS_end,
      strand == "-" ~ CDS_start - transcript_start
    )
  )


##### Get NMD features

NMD_status <- fread(factr_transcript_file)
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
  UTR_features,
  by = c(gtf_feature_cols)
) %>%
  dplyr::inner_join(
    NMD_status,
    by = c("transcript_id", "gene_id",
           "gene_name")
  )



##### Add isoform stabilities

isoforms <- EZget(ezbdo,
                  type = "kinetics",
                  features = "transcript_id",
                  exactMatch = FALSE) %>%
  dplyr::inner_join(
    ezbdo$readcounts$isoform_quant_rsem,
    by = c("transcript_id", "sample")
  ) %>%
  dplyr::filter(grepl(sample_pattern, sample)) %>%
  dplyr::group_by(
    XF, transcript_id
  ) %>%
  dplyr::summarise(
    log_kdeg = mean(log_kdeg),
    log_ksyn = mean(log(exp(log_kdeg)*TPM)),
    avg_reads = mean(n),
    avg_TPM = mean(TPM),
    effective_length = mean(effective_length),
    avg_lkd_se = mean(se_log_kdeg)
  ) %>%
  dplyr::mutate(
    kdeg = exp(log_kdeg),
    ksyn = exp(log_ksyn)
  ) %>%
  dplyr::filter(avg_reads > read_cutoff &
                  avg_TPM > TPM_cutoff) %>%
  dplyr::rename(
    old_gene_id = XF
  )


combined_table <- combined_table %>%
  inner_join(isoforms,
             by = c("old_gene_id",
                    "transcript_id"))


### What if I also add my own NMD determination?
combined_table <- EZget(ezbdo,
                        type = "comparisons",
                        features = "transcript_id",
                        exactMatch = FALSE) %>%
  mutate(EZbakR_nmd = case_when(
    difference < nmd_diff_cutoff & padj < nmd_padj_cutoff ~ TRUE,
    .default = FALSE
  )) %>%
  dplyr::select(XF, transcript_id, EZbakR_nmd) %>%
  dplyr::rename(old_gene_id = XF) %>%
  dplyr::inner_join(combined_table,
                    by = c("old_gene_id", "transcript_id"))

setwd(outdir)
write_csv(combined_table,
          file = paste0(filename, ".csv"))
