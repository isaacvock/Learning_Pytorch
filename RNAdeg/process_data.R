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
library(readxl)

source("C:/Users/isaac/Documents/ML_pytorch/Scripts/RNAdeg/processing_functions.R")


# Process SQUANTI augmented RNAdeg data ----------------------------------------

RNAdeg_data <- read_csv(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_all_features.csv"
) %>%
  dplyr::rename(
    gene_id = gene_id.x,
    gene_name = gene_name.x,
    strand = strand.x,
    threepUTR_length = `3'UTR_length`
  ) %>%
  dplyr::select(
    chrom,
    gene_id,
    transcript_id,
    MSTRG_gene_id,
    strand,
    is_NMD,
    predicted_NMD,
    num_of_downEJs,
    stop_to_lastEJ,
    threepUTR_length,
    difference,
    uncertainty,
    padj,
    log_kdeg_DMSO,
    avg_lkd_se_DMSO,
    log_kdeg_SMG1i,
    avg_lkd_se_SMG1i,
    log_ksyn_DMSO,
    log_ksyn_SMG1i,
    avg_TPM_DMSO,
    avg_TPM_SMG1i,
    log_CAI,
    avg_CSC,
    avg_AASC,
    threeputr_miRNAseed_count,
    fiveputr_miRNAseed_count,
    CDS_miRNAseed_count,
    threeputr_DRACH_count,
    fiveputr_DRACH_count,
    CDS_DRACH_count,
    m6A_site_cnt_Schwartz2014,
    first_exon_length,
    longest_internal_exon_length,
    last_exon_length,
    gene_length,
    mean_intron_length,
    longest_intron_length,
    total_exons,
    cds_length,
    distance_to_last_junction,
    distance_to_nearest_downstream_junction,
    distance_5prime_to_stop,
    stop_codon_exon_size,
    stop_codon_exon_number,
    structural_category,
    promoter_seq,
    threeputr_seq,
    fiveputr_seq,
    CDS_seq,
    ORF_seq
  ) %>%
  dplyr::mutate(
    minus1_AA = str_sub(ORF_seq,
                        start = str_length(ORF_seq),
                        end = str_length(ORF_seq)),
    minus2_AA = str_sub(ORF_seq,
                        start = str_length(ORF_seq)-1,
                        end = str_length(ORF_seq)-1),
    termination_codon = str_sub(CDS_seq,
                                start = str_length(CDS_seq)-2,
                                end = str_length(CDS_seq)),
    TC_tetranucleotide = paste0(str_sub(CDS_seq,
                                 start = str_length(CDS_seq)-2,
                                 end = str_length(CDS_seq)),
                                str_sub(threeputr_seq,
                                        start = 1,
                                        end = 1)),
    GCcontent_downstream_TC = str_count(
      str_sub(threeputr_seq, start = 2,
              end = pmin(str_length(threeputr_seq),
                         202)),
              "[GC]") / pmin(str_length(threeputr_seq),
                             202)
  ) %>%
  dplyr::group_by(
    chrom,
    gene_id,
    transcript_id,
    MSTRG_gene_id,
    strand
  ) %>%
  dplyr::mutate(
    nonPTC_log_kdeg_DMSO = mean(log_kdeg_DMSO[!is_NMD & avg_TPM_DMSO == max(avg_TPM_DMSO[!is_NMD])]),
    nonPTC_log_kdeg_SMG1i = mean(log_kdeg_SMG1i[!is_NMD & avg_TPM_SMG1i == max(avg_TPM_SMG1i[!is_NMD])])
  )

write_csv(RNAdeg_data,
          "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_model_features.csv"
          )


# Trimmed mix SQUANTI starting point -------------------------------------------

SQUANTI_info <- fread("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/DataTables/Mix_trimmed_GMST_isoforms_EZbakR_factR2_SQANTI_merged.tsv") %>%
  as_tibble()

isoforms_to_keep <- read_csv(
  "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/isoforms_to_keep.csv"
)

mix_gtf_nofactr <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/mix_trimmed.gtf")
mix_gtf <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2.gtf")
mcols(mix_gtf)$ID <- NULL
mix_gtf <- mix_gtf[!grepl("_", seqnames(mix_gtf))]


ezbdo <- readRDS("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/EZbakRFits/Mix_trimmed_EZbakRFit_withgenewide.rds")


##### Add EZbakR information #####

# Stability information already in there, but just want to
# double check that it is right estimates
isoforms_ez <- EZget(ezbdo,
                  type = "kinetics",
                  features = "transcript_id",
                  exactMatch = FALSE) %>%
  dplyr::inner_join(
    ezbdo$readcounts$isoform_quant_rsem,
    by = c("transcript_id", "sample")
  ) %>%
  dplyr::group_by(
    XF, transcript_id
  ) %>%
  dplyr::summarise(
    log_kdeg_DMSO = mean(log_kdeg[base::grepl("DMSO", sample)]),
    log_kdeg_SMG1i = mean(log_kdeg[base::grepl("11j", sample)]),
    avg_lkd_se_DMSO = mean(se_log_kdeg[base::grepl("DMSO", sample)]),
    avg_lkd_se_SMG1i = mean(se_log_kdeg[base::grepl("11j", sample)]),
    log_ksyn_DMSO = mean(log(exp(log_kdeg[base::grepl("DMSO", sample)])*TPM[base::grepl("DMSO", sample)])),
    log_ksyn_SMG1i = mean(log(exp(log_kdeg[base::grepl("11j", sample)])*TPM[base::grepl("11j", sample)])),
    avg_reads_DMSO = mean(n[base::grepl("DMSO", sample)]),
    avg_reads_SMG1i = mean(n[base::grepl("11j", sample)]),
    avg_TPM_DMSO = mean(TPM[base::grepl("DMSO", sample)]),
    avg_TPM_SMG1i = mean(TPM[base::grepl("11j", sample)]),
    effective_length = mean(effective_length)
  ) %>%
  dplyr::filter((avg_reads_DMSO > 25 &
                  avg_TPM_DMSO > 2) |
                  (avg_reads_SMG1i > 25 &
                     avg_TPM_SMG1i > 2)) %>%
  dplyr::rename(
    old_gene_id = XF
  )


RNAdeg_data <- SQUANTI_info %>%
  dplyr::inner_join(
    isoforms_ez,
    by = c("transcript_id")
  )


# Filter out questionable shit
RNAdeg_data_filter <- RNAdeg_data %>%
  inner_join(
    isoforms_to_keep,
    by = "transcript_id"
  ) %>%
  dplyr::filter(
    !problematic & !(structural_category %in% c("antisense", "genic_intron", "intergenic") )
  )

write_csv(RNAdeg_data_filter,
          file = "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_filter.csv")


##### Impute new CDSs into GTF, then get sequences #####

seqname_dict <- as_tibble(mix_gtf) %>%
  dplyr::filter(type == "transcript") %>%
  dplyr::select(transcript_id, seqnames) %>%
  dplyr::distinct()

RNAdeg_data_CDS <- RNAdeg_data_filter %>%
  inner_join(seqname_dict,
             by = "transcript_id") %>%
  dplyr::filter(
    !is.na(CDS_genomic_start) &
      !is.na(CDS_genomic_end)
  ) %>%
  dplyr::mutate(CDS_genomic_start_corrected = case_when(
    strand == "-" ~ CDS_genomic_end,
    .default = CDS_genomic_start
  ),
  CDS_genomic_end_corrected = case_when(
    strand == "-" ~ CDS_genomic_start,
    .default = CDS_genomic_end
  )
  )

cds <- GenomicRanges::GRanges(
  seqnames = RNAdeg_data_CDS$seqnames,
  ranges = IRanges(
    start = RNAdeg_data_CDS$CDS_genomic_start_corrected,
    end = RNAdeg_data_CDS$CDS_genomic_end_corrected
  ),
  strand = RNAdeg_data_CDS$strand,
  transcript_id = RNAdeg_data_CDS$transcript_id,
  gene_id = RNAdeg_data_CDS$gene_id,
  gene_name = RNAdeg_data_CDS$gene_name,
  type = "CDS"
)

# cds_exon <- GenomicRanges::intersect(mix_gtf[mix_gtf$type == "exon"],
#           cds)


# as_tibble(mix_gtf) %>%
#   dplyr::filter(type == "gene"
#                 & gene_id == "ENSG00000000457.14")

exons <- mix_gtf[mix_gtf$type == "exon"]
exons$gt_id <- paste0(exons$gene_id, "_", exons$transcript_id)

gt_ids <- paste0(cds$gene_id, "_", cds$transcript_id)
cds$gt_id <- gt_ids

cds_exon_list <- vector(
  mode = "list",
  length = length(gt_ids)
)

# This is annoyingly slow, like 10-20 minutes to run. I can do intersection on full GRange objects,
# but then I lose CDS metadata with no way to easily retrieve it.
count <- 1
for(g in gt_ids){

  cds_t <- cds[cds$gt_id == g]
  exons_t <- exons[exons$gt_id == g]

  cds_exon_t <- GenomicRanges::intersect(
    cds_t,
    exons_t
  )

  mcols(cds_exon_t) <- mcols(cds_t)
  cds_exon_t$type <- "CDS"

  cds_exon_list[[count]] <- cds_exon_t
  count <- count + 1

  if((count %% 100) == 0){
    print(paste0((count / length(gt_ids))* 100, "% complete"))
  }

}

cds_exon <- do.call(
  c,
  cds_exon_list
)

cds_exon$gt_id <- NULL


# Strip mcols down to basic information
mix_gtf_simple <- mix_gtf[!(mix_gtf$type %in% c("CDS"))]
mcols(mix_gtf_simple) <- mcols(mix_gtf_simple) %>%
  as_tibble() %>%
  dplyr::select(
    transcript_id,
    gene_id,
    gene_name,
    type
  )

mix_gtf_cds <- c(
  mix_gtf_simple,
  cds_exon
)

mix_gtf_cds

rtracklayer::export(
  mix_gtf_cds,
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/Annotation_gamut_analyses/Annotations/mix_trimmed_squantiCDS.gtf"
)

mix_gtf_cds


txdb <- GenomicFeatures::makeTxDbFromGRanges(mix_gtf_cds)
outdir <- "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/"


### Promoter sequences

promoter_seqs <- getSeq(Hsapiens, promoters(txdb))

seq_as_str <- as.character(promoter_seqs)

promoter_df <- tibble(
  seq = seq_as_str,
  transcript_id = names(seq_as_str)
)

write_csv(promoter_df,
          file = paste0(outdir, "/promoter_seqs.csv"))


### 3'UTR sequences

threeprimeutr <- threeUTRsByTranscript(txdb,
                                       use.names = TRUE)

threeprimeutr_seq <- getSeq(Hsapiens,
                            unlist(threeprimeutr))

threeprimeutr_str <- as.character(threeprimeutr_seq)

threeprimeutr_df <- tibble(
  seq = threeprimeutr_str,
  transcript_id = names(threeprimeutr_str)
) %>%
  group_by(transcript_id) %>%
  summarise(seq = paste(seq, collapse = ""))


write_csv(threeprimeutr_df,
          file = paste0(outdir, "/threeprimeUTR_seqs.csv"))



### 5'UTR sequences

fiveprimeutr <- fiveUTRsByTranscript(txdb,
                                     use.names = TRUE)

fiveprimeutr_seq <- getSeq(Hsapiens,
                           fiveprimeutr) %>%
  unlist()

fiveprimeutr_str <- as.character(fiveprimeutr_seq)

fiveprimeutr_df <- tibble(
  seq = fiveprimeutr_str,
  transcript_id = names(fiveprimeutr_str)
) %>%
  group_by(transcript_id) %>%
  summarise(seq = paste(seq, collapse = ""))

write_csv(fiveprimeutr_df,
          file = paste0(outdir, "/fiveprimeUTR_seqs.csv"))


### Get CDS sequence

cds <- cdsBy(txdb, by = "tx",
             use.names = TRUE)

cds_seq <- getSeq(Hsapiens,
                  cds) %>%
  unlist()

cds_str <- as.character(cds_seq)

cds_df <- tibble(
  seq = cds_str,
  transcript_id = names(cds_str)
) %>%
  group_by(transcript_id) %>%
  summarise(seq = paste(seq, collapse = ""))


write_csv(cds_df,
          file = paste0(outdir, "/CDS_seqs.csv"))


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


### Starting table
RNAdeg_data <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_filter.csv")
ezbdo <- readRDS("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/EZbakRFits/Mix_trimmed_EZbakRFit_withgenewide.rds")

### Codon data
codon_dict <- fread(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/nuclear_codon_statistics.tsv"
) %>%
  as_tibble()

aa_stability <- read_xlsx(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/Stability_scores_Narula2019.xlsx",
  skip = 4
) %>%
  dplyr::select(`Amino acid`,
                `hORF14 AASC`) %>%
  dplyr::mutate(`Amino acid` =
                  case_when(
                    `Amino acid` == "M" ~ "Met",
                    `Amino acid` == "W" ~ "Trp",
                    `Amino acid` == "F" ~ "Phe",
                    `Amino acid` == "V" ~ "Val",
                    `Amino acid` == "D" ~ "Asp",
                    `Amino acid` == "I" ~ "Ile",
                    `Amino acid` == "Y" ~ "Tyr",
                    `Amino acid` == "A" ~ "Ala",
                    `Amino acid` == "G" ~ "Gly",
                    `Amino acid` == "K" ~ "Lys",
                    `Amino acid` == "E" ~ "Glu",
                    `Amino acid` == "R" ~ "Arg",
                    `Amino acid` == "L" ~ "Leu",
                    `Amino acid` == "T" ~ "Thr",
                    `Amino acid` == "N" ~ "Asn",
                    `Amino acid` == "P" ~ "Pro",
                    `Amino acid` == "Q" ~ "Gln",
                    `Amino acid` == "C" ~ "Cys",
                    `Amino acid` == "H" ~ "His",
                    `Amino acid` == "S" ~ "Ser"
                  )
  )

codon_stability <- read_xlsx(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/Stability_scores_codons_Narula2019.xlsx",
  skip = 5
) %>%
  dplyr::mutate(
    CODON = toupper(Codon)
  ) %>%
  dplyr::select(-Codon) %>%
  dplyr::select(
    CODON, `ORF14 CSC`
  )



abundances <- ezbdo$readcounts$isoform_quant_rsem %>%
  dplyr::group_by(transcript_id) %>%
  dplyr::summarise(
    TPM_avg = mean(TPM[grepl("DMSO", sample)])
  )

CDS <- read_csv(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/CDS_seqs.csv"
)


abundance_with_seq <- abundances %>%
  dplyr::inner_join(CDS,
                    by = "transcript_id") %>%
  dplyr::rowwise() %>%
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
  ) %>%
  dplyr::inner_join(
    codon_stability,
    by = "CODON"
  ) %>%
  dplyr::inner_join(
    aa_stability,
    by = "Amino acid"
  )

codon_optimality <- CDS_codon_tidy %>%
  group_by(CODON, `Amino acid`) %>%
  summarise(
    RSCU = mean(RSCU),
    count = sum(count),
    CSC = mean(`ORF14 CSC`),
    AASC = mean(`hORF14 AASC`)
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
    log_CAI = (1 / sum(count))*sum(log(weight_hogg)*count),
    avg_CSC = sum(CSC*count)/sum(count),
    avg_AASC = sum(AASC*count)/sum(count)
  )

setwd("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/")
write_csv(all_CDS_CAI,
          "Various_codon_scores.csv")


RNAdeg_data_codon <- RNAdeg_data %>%
  left_join(all_CDS_CAI,
             by = c("transcript_id"))

write_csv(RNAdeg_data_codon,
          file = "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_with_codon_scores.csv")


# Sequence motifs --------------------------------------------------------------


### Load data

promoter_seq <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/promoter_seqs.csv")

threeputr_seq <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/threeprimeUTR_seqs.csv")

CDS_seq <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/CDS_seqs.csv")

fiveputr_seq <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/fiveprimeUTR_seqs.csv")

RNAdeg_data <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_with_codon_scores.csv") %>%
  dplyr::select(gene_id, gene_name, strand, transcript_id)


seq_df <- RNAdeg_data %>%
  inner_join(promoter_seq %>%
               dplyr::rename(
                 promoter_seq = seq
               ),
             by = "transcript_id") %>%
  inner_join(threeputr_seq %>%
               dplyr::rename(
                 threeputr_seq = seq
               ),
             by = "transcript_id") %>%
  inner_join(fiveputr_seq %>%
               dplyr::rename(
                 fiveputr_seq = seq
               ),
             by = "transcript_id") %>%
  inner_join(CDS_seq %>%
               dplyr::rename(
                 CDS_seq = seq
               ),
             by = "transcript_id")



miR_fasta <- readRNAStringSet(
  "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/HEK293T_miRNA.fa"
)

miR_seeds <- tibble(
  name = names(miR_fasta),
  sequence = sapply(
    miR_fasta,
    function(seq){
      as.character(subseq(seq, start = 2, end = 7))
    }
  )
) %>%
  dplyr::mutate(
    sequence = chartr("AGCTN", "TCGAN",sequence) # Complement sequence
  )


### Count miRNA seed sequence hits in each sequence

seq_df$threeputr_miRNAseed_count <- sapply(
  seq_df[["threeputr_seq"]],
  function(query_string) {
    sum(sapply(
      miR_seeds$sequence,
      function(seed){
        str_count(query_string, seed)
      }
    ))
  }
)


seq_df$fiveputr_miRNAseed_count <- sapply(
  seq_df[["fiveputr_seq"]],
  function(query_string) {
    sum(sapply(
      miR_seeds$sequence,
      function(seed){
        str_count(query_string, seed)
      }
    ))
  }
)


seq_df$CDS_miRNAseed_count <- sapply(
  seq_df[["CDS_seq"]],
  function(query_string) {
    sum(sapply(
      miR_seeds$sequence,
      function(seed){
        str_count(query_string, seed)
      }
    ))
  }
)


### Count DRACH motifs in each

seq_df$threeputr_DRACH_count <- str_count(seq_df$threeputr_seq,
                                          "[TC][GA]AC[ATC]")
seq_df$fiveputr_DRACH_count <- str_count(seq_df$fiveputr_seq,
                                          "[TC][GA]AC[ATC]")
seq_df$CDS_DRACH_count <- str_count(seq_df$CDS_seq,
                                          "[TC][GA]AC[ATC]")


### Merge back into table

RNAdeg_data <- read_csv("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_with_codon_scores.csv")

RNAdeg_data <- RNAdeg_data %>%
  inner_join(seq_df,
             by = "transcript_id")

write_csv(RNAdeg_data,
          "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_all_features.csv")

write_csv(seq_df,
          "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/Motif_and_miRNAseed_counts.csv")



# m6A data ---------------------------------------------------------------------


##### SCHWARTZ SITES #####

# Gets a couple warnings due to NAs, nothing to worry about though
sites_df <- read_xlsx("C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/m6A_sites_Schwartz2014.xlsx",
                      sheet = 2)
gtf_gr <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/mix_trimmed.gtf")


# Convert your data frame to GRanges
sites_gr <- GRanges(
  seqnames = sites_df$chr,
  ranges = IRanges(start = sites_df$position, end = sites_df$position),
  strand = sites_df$strand
)

# Ensure your GRanges object from the GTF file has 'transcript_id' and represents exons
exons_gr <- gtf_gr[gtf_gr$type == "exon"]

# Find overlaps between your sites and the exons
overlaps <- findOverlaps(sites_gr, exons_gr, ignore.strand = FALSE)

# Extract transcript IDs that overlap with your sites
overlapping_transcripts <- mcols(exons_gr)$transcript_id[subjectHits(overlaps)]

# Count the number of overlaps per transcript
m6A_cnt_df <- tibble(
  transcript_id = overlapping_transcripts
) %>%
  dplyr::count(transcript_id)

# Compile all transcript IDs and indicate which contain m6A sites
all_transcript_ids <- unique(mcols(exons_gr)$transcript_id)

schwartz_df <- tibble(
  transcript_id = all_transcript_ids
) %>%
  dplyr::left_join(m6A_cnt_df,
                   by = "transcript_id") %>%
  dplyr::mutate(n = ifelse(is.na(n), 0, n)) %>%
  dplyr::rename(
    m6A_site_cnt_Schwartz2014 = n
  )


schwartz_df


### Add to table
RNAdeg_table <- read_csv(
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_all_features.csv"
)


RNAdeg_table <- RNAdeg_table %>%
  inner_join(schwartz_df,
             by = "transcript_id")

write_csv(
  RNAdeg_table,
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/RNAdeg_data_all_features.csv"
)

write_csv(
  schwartz_df,
  "C:/Users/isaac/Documents/Simon_Lab/Isoform_Kinetics/Data/ML_features/SQUANTI_seqs/Schwartz_m6A_overlap.csv"
)



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
