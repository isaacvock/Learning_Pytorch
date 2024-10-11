### PURPOSE OF THIS SCRIPT
## Get sequence features to train ML model on

# Load dependencies ------------------------------------------------------------

library(GenomicFeatures)
library(Biostrings)
library(rtracklayer)
library(dplyr)
library(readr)
library(ggplot2)

library(BSgenome.Hsapiens.UCSC.hg38)

# Get various sequence features of isoforms ------------------------------------

gtf <- rtracklayer::import(
  "G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/longread_justin/11j_LRSR_subreads_cons_f0.05.merged.sorted.sorted.gtf"
)

gtf <- gtf[strand(gtf) != "*"]

txdb <- GenomicFeatures::makeTxDbFromGRanges(gtf)

transcript_coords <- transcriptsBy(txdb,
                                   by = "gene")

promoter_seqs <- getPromoterSeq(transcript_coords,
                               Hsapiens,
                               upstream = 1000,
                               downstream = 100)

promoter_seqs <- getSeq(Hsapiens, promoters(txdb))

seq_as_str <- as.character(promoter_seqs)

promoter_df <- tibble(
  seq = seq_as_str,
  transcript_id = names(seq_as_str)
)

write_csv(promoter_df,
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/promoter_seqs.csv")
