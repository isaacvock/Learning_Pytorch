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
library(stringr)

# Get various sequence features of isoforms ------------------------------------

# gtf <- rtracklayer::import(
#   "G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/longread_justin/11j_LRSR_subreads_cons_f0.05.merged.sorted.sorted.gtf"
# )

gtf <- rtracklayer::import(
  "G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/longread_justin/LRSR_f05_factR2/factR2.gtf"
)

gtf <- gtf[strand(gtf) != "*"]

txdb <- GenomicFeatures::makeTxDbFromGRanges(gtf)

# transcript_coords <- transcriptsBy(txdb,
#                                    by = "gene")




### Promoter sequences

promoter_seqs <- getSeq(Hsapiens, promoters(txdb))

seq_as_str <- as.character(promoter_seqs)

promoter_df <- tibble(
  seq = seq_as_str,
  transcript_id = names(seq_as_str)
)

write_csv(promoter_df,
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/promoter_seqs.csv")


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
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/threeprimeUTR_seqs.csv")



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
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/fiveprimeUTR_seqs.csv")


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
          file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/CDS_seqs.csv")

# Stop codons are off by one codon currently...
# Start codons are right though
cds_df %>%
  mutate(
    stop_codon = str_sub(seq, -3, -1)
  ) %>%
  filter(transcript_id == "MSTRG.9.1")
