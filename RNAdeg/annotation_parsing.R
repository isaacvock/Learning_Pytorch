### PURPOSE OF THIS SCRIPT
## Get features to train ML models on from annotation and
## genome fasta file.

# Load dependencies ------------------------------------------------------------

library(rtracklayer)
library(Biostrings)
library(dplyr)
library(readr)
library(factR2)
library(biomaRt)

# Get a bunch of relevant features from GTF ------------------------------------


##### Run factR2 to annotate start and stop codons (which SpliceWiz annoyingly needs)

# Generate the gene ID table
mart <- biomaRt::useMart("ENSEMBL_MART_ENSEMBL", dataset="hsapiens_gene_ensembl",host = 'https://www.ensembl.org')

# look at all biomaRt attributes
attributes <-  listAttributes(mart = mart)

# Add external gene identifiers if not present
t2g <- biomaRt::getBM(attributes = c('ensembl_gene_id_version','ensembl_gene_id',"external_gene_name", "gene_biotype", "description"), mart = mart)
geneid2name <- dplyr::select(t2g, c(ensembl_gene_id_version, ensembl_gene_id,external_gene_name,gene_biotype,description))
t2g <- dplyr::select(t2g, c(ensembl_gene_id_version,ensembl_gene_id,external_gene_name,gene_biotype,description))

### Set factR2 variables

# directory containing gtf annotation file
ac.gtf.file <- 'G:/Shared drives/Matthew_Simon/IWV/Annotations/Hogg_annotations/Simple_Annotations/js_ensembl_stringtie_pruned_try2/cleaned_reference.sorted.gtf'

check_gtf <- as_tibble(rtracklayer::import(ac.gtf.file))

check_gtf <- check_gtf %>% filter(strand != "*") %>%
  filter(!grepl("KI", seqnames) &
           !grepl("GL", seqnames) &
           !(seqnames %in% c("chrY", "chrMT")))

# Creating a factRObject from annotation and genome
fobj <- createfactRObject(ac.gtf.file, reference = "Human")
activeSet(fobj) <- "transcript"

# A count the number of events by the splicing type (AStype)
ase(fobj) %>%
  group_by(AStype) %>%
  tally()


### Running factR pipeline


fobj <- runfactR(fobj)


### Exporting Data
exportAll(fobj,
          path = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg")


##### Filter and add start/stop codons to factR2 annotation

# Load gtf
setwd("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg")
factR2 <- as_tibble(rtracklayer::import("factR2.gtf"))


### Start codons
# Start codon starts at either the minimum of all exon starts in a trancsript's CDS (if
# transcript on + strand), or the maximum of all exon ends in a transcript's
# CDS (if transcript on - strand).
starts <- factR2 %>%
  filter(type == "CDS") %>%
  ungroup() %>%
  group_by(seqnames, transcript_id, gene_id,
           gene_name, strand) %>%
  summarise(possible_start = min(start),
            possible_start2 = max(end),
            type =  "start_codon") %>%
  mutate(start = ifelse(strand == "+",
                        possible_start,
                        possible_start2 - 2),
         end = ifelse(strand == "+",
                      start + 2,
                      possible_start2),
         width = 3)

### Stop codons
# Stop codon ends at either the max CDS exon end (if transcript is on the + strand), or
# the minimum CDS exon start (if transcript is on the - strand).
stops <- factR2 %>%
  filter(type == "CDS") %>%
  ungroup() %>%
  group_by(seqnames, transcript_id, gene_id,
           gene_name, strand) %>%
  summarise(possible_end = min(start),
            possible_end2 = max(end),
            type =  "stop_codon") %>%
  mutate(end = ifelse(strand == "+",
                      possible_end2,
                      possible_end + 2),
         start = ifelse(strand == "+",
                        end - 2,
                        possible_end),
         width = 3)

factR2_new <- bind_rows(list(factR2,
                             starts,
                             stops))



# Convert to GenomicRanges object and export
new_GR <- GRanges(seqnames = Rle(factR2_new$seqnames),
                  ranges = IRanges(factR2_new$start, end = factR2_new$end,
                                   names = 1:nrow(factR2_new)),
                  strand = Rle(factR2_new$strand))

mcols(new_GR) <- factR2_new %>%
  dplyr::select(-seqnames, -start, -end, -strand)

rtracklayer::export(new_GR,
                    con = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/factR2_start_and_stop.gtf")


##### Get features from the factR2 annotation

gtf_df <- as_tibble(new_GR)

exonic_features <- gtf_df %>%
  dplyr::filter(type == "exon") %>%
  dplyr::group_by(seqnames, transcript_id, gene_id, gene_name, old_gene_id) %>%
  dplyr::summarise(
    total_length = sum(width),
    avg_length = total_length/dplyr::n(),
    max_length = max(width),
    min_length = min(width),
    num_exons = max(exon_number),
    problematic = unique(problematic)
  )


UTR_features <- gtf_df %>%
  dplyr::group_by(seqnames, transcript_id, gene_id, gene_name, old_gene_id) %>%
  dplyr::summarise(
    fiveprimeUTR_lngth = case_when(
      strand == "+" ~ max(start[type == "CDS"]) - start[type == "transcript"],
      strand == "-" ~ end[type == "transcript"] - max(end[type == "CDS"])
    ),
    threeprimeUTR_lngth = case_when(
      strand == "-" ~ max(start[type == "CDS"]) - start[type == "transcript"],
      strand == "+" ~ end[type == "transcript"] - max(end[type == "CDS"])
    )
  )

