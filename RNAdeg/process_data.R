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
library(coRdon)
library(tidyr)

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


# Codon optimality explortation ------------------------------------------------
### Analysis process
## Going to look at the codon distribution among stable and unstable transcripts
## Ranking of codon optimality from previous study
## 1) GCT (A)
## 2) GGT (G)
## 3) GTC (V)
## 4) TTG (L)
## 5) GTT (V)
## 6) GCC (A)
## 7) CCA (P)
## 8) ACT (T)
## 9) TCT (S)
## 10) TCC (S)
## 11) ACC (T)
## 12) ATC (I)
## 13) AAG (K)
## 14) TAC (Y)
## 15) TTC (F)
## 16) GAA (E)
## 17) CGT (R)
## 18) CAA (Q)
## 19) CAC (H)
## 20) AAC (N)
## 21) GAC (D)
## 22) ATT (I)
## 23) AGA (R)
## 24) CCT (P)
## 25) GGC (G)
## 26) TGG (W)
## 27) TGT (C)
## 28) TTA (L) Start "Non-optimal codons"
## 29) GAT (D)
## 30) ATG (M)
## 31) TTT (F)
## 32) TGC (C)
## 33) CAT (H)
## 34) GCA (A)
## 35) TAT (Y)
## 36) CCC (P)
## 37) GGG (G)
## 38) GTG (V)
## 39) GCG (A)
## 40) CGC (R)
## 41) TCA (S)
## 42) GAG (E)
## 43) GGA (G)
## 44) TCG (S)
## 45) CGG (R)
## 46) AAT (N)
## 47) CTT (L)
## 48) CTA (L)
## 49) CAG (Q)
## 50) CTC (L)
## 51) ACA (T)
## 52) AGC (S)
## 53) AAA (K)
## 54) AGT (S)
## 55) ACG (T)
## 56) CTG (L)
## 57) CCG (P)
## 58) GTA (V)
## 59) AGG (R)
## 60) CGA (R)
## 61) ATA (I)


codon_dict <- fread(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/nuclear_codon_statistics.tsv"
) %>%
  as_tibble()


abundances <- ezbdo$readcounts$isoform_quant_rsem %>%
  dplyr::group_by(transcript_id) %>%
  dplyr::summarise(
    TPM_avg = mean(TPM[grepl("DMSO", sample)])
  )

CDS <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/CDS_seqs.csv") %>%
  dplyr::rename(CDS = seq)
threepUTR <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/threeprimeUTR_seqs.csv") %>%
  dplyr::rename(threepUTR = seq)

abundance_with_seq <- abundances %>%
  dplyr::inner_join(CDS,
                    by = "transcript_id") %>%
  dplyr::inner_join(threepUTR,
                    by = "transcript_id") %>%
  dplyr::rowwise() %>%
  mutate(
    seq = paste0(CDS, str_sub(threepUTR, 1, 3))
  ) %>%
  filter(str_length(seq) > 25)

abundant_CDS <- abundance_with_seq %>%
  ungroup() %>%
  dplyr::mutate(
    TPM_rank = ntile(TPM_avg,
                     10)
  ) %>%
  filter(TPM_rank > 8)

CDS_seqs <- DNAStringSet(
  x = abundant_CDS$seq
)

CDS_codons <- codonTable(CDS_seqs)

CDS_codon_counts <- CDS_codons@counts %>% as_tibble()
CDS_codon_counts$length <- CDS_codons@len
CDS_codon_counts$transcript_id <- abundant_CDS$transcript_id

CDS_codon_tidy <- CDS_codon_counts %>%
  pivot_longer(
    cols = !c(length,transcript_id),
    values_to = "count",
    names_to = "CODON"
  ) %>%
  dplyr::inner_join(
    codon_dict %>%
      dplyr::select(CODON, `Amino acid`, RSCU),
    by = "CODON"
  )

codon_optimality <- CDS_codon_tidy %>%
  group_by(CODON, `Amino acid`) %>%
  summarise(
    RSCU = mean(RSCU),
    count = sum(count)
  ) %>%
  group_by(`Amino acid`) %>%
  mutate(
    RSCU_hogg = count / mean(count)
  ) %>%
  mutate(weight = RSCU / max(RSCU),
         weight_hogg = RSCU_hogg / max(RSCU_hogg))


# Exclude start and stop
all_CDS_seqs <- DNAStringSet(
  x = str_sub(abundance_with_seq$seq, start = 4, end = str_length(abundance_with_seq$seq) - 3)
)


all_CDS_codons <- codonTable(all_CDS_seqs)

all_CDS_codon_counts <- all_CDS_codons@counts %>% as_tibble()
all_CDS_codon_counts$length <- all_CDS_codons@len
all_CDS_codon_counts$transcript_id <- abundance_with_seq$transcript_id

all_CDS_codon_tidy <- all_CDS_codon_counts %>%
  pivot_longer(
    cols = !c(length,transcript_id),
    values_to = "count",
    names_to = "CODON"
  ) %>%
  dplyr::inner_join(
    codon_optimality %>%
      dplyr::select(-count),
    by = "CODON"
  )

all_CDS_CAI <- all_CDS_codon_tidy %>%
  dplyr::group_by(transcript_id) %>%
  dplyr::summarise(
    log_CAI = (1 / sum(count))*sum(log(weight_hogg)*count)
  )

setwd("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/")
write_csv(all_CDS_CAI,
          "CAI_codon_scores.csv")

# Sandbox geneal deg features --------------------------------------------------

mix_gtf <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2.gtf")
transcript_file <- "C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2_transcript.tsv"
ezbdo <- readRDS("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/EZbakRFits/Mix_trimmed_EZbakRFit_withgenewide.rds")

##### Codon optimality calculation #####


##### Process m6A site data #####


##### Process AU-rich 3'UTR element data #####


##### Process miRNA data #####


##### Calculate Kozak scores #####


#####



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
