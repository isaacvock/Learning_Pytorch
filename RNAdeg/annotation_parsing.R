### PURPOSE OF THIS SCRIPT
## Get features to train ML models on from annotation and
## genome fasta file.

# Load dependencies ------------------------------------------------------------

library(rtracklayer)
library(Biostrings)
library(dplyr)
library(readr)
library(devtools)
devtools::load_all("C:/Users/isaac/Documents/Simon_Lab/EZbakR/")
library(GenomicFeatures)

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
  dplyr::inner_join(
    ezbdo$readcounts$isoform_quant_rsem,
    by = c("transcript_id", "sample")
  ) %>%
  dplyr::filter(grepl("DMSO", sample)) %>%
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
  dplyr::filter(avg_reads > 25) %>%
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
    difference < -1 & padj < 0.01 ~ TRUE,
    .default = FALSE
  )) %>%
  dplyr::select(XF, transcript_id, EZbakR_nmd) %>%
  dplyr::rename(old_gene_id = XF) %>%
  dplyr::inner_join(combined_table,
                    by = c("old_gene_id", "transcript_id"))

setwd("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/")
write_csv(combined_table,
          file = "RNAdeg_dataset.csv")


###### Want to also make a table of gene-wise stabilties


ezbdo <- readRDS("G:/Shared drives/Matthew_Simon/IWV/Hogg_lab/EZbakRFits/jmdata_LRSR_f05/Full_EZbakRFit_withgenewide.rds")


### Step 1: get effective exonic lengths

# First, import the GTF-file
gtf <- rtracklayer::import(
  "G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/longread_justin/11j_LRSR_subreads_cons_f0.05.merged.sorted.sorted.gtf"
)

gtf <- gtf[strand(gtf) != "*"]

txdb <- makeTxDbFromGRanges(gtf)

# then collect the exons per gene id
exons.list.per.gene <- exonsBy(txdb,
                               by="gene")

# then for each gene, reduce all the exons to a set of non overlapping exons, calculate their lengths (widths) and sum then
exonic.gene.sizes <- sum(width(reduce(exons.list.per.gene)))

# Combine to infer "intron" length
exon_df <- tibble(exon_width = exonic.gene.sizes,
                  gene_id = names(exonic.gene.sizes))


lengths <- exon_df %>%
  dplyr::rename(XF = gene_id,
                length = exon_width)



### Step 2: get tables of kinetic parameters

genes <- EZget(ezbdo,
               type = "kinetics",
               features = "XF",
               exactMatch = TRUE) %>%
  dplyr::inner_join(lengths,
                    by = "XF") %>%
  dplyr::filter(grepl("DMSO", sample)) %>%
  dplyr::group_by(
    XF
  ) %>%
  dplyr::summarise(
    log_kdeg = mean(log_kdeg),
    log_ksyn = mean(log(exp(log_kdeg)*(n/(length/1000)))),
    avg_reads = mean(n),
    avg_RPK = mean(n/(length/1000)),
    exonic_length = mean(length),
    avg_lkd_se = mean(se_log_kdeg)
  ) %>%
  dplyr::mutate(
    kdeg = exp(log_kdeg),
    ksyn = exp(log_ksyn)
  ) %>%
  dplyr::filter(avg_reads > 25) %>%
  dplyr::rename(
    old_gene_id = XF
  )

setwd("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/")
write_csv(genes,
          file = "RNAdeg_genewise_dataset.csv")

