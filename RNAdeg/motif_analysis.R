### PURPOSE OF THIS SCRIPT
## Perform simpler motif analysis to associate sequences with stability
## trends. Going to try and borrow methods from previous MPRA studies,
## Oikonomou et al. 2014, and Rabani et al. 2017.

# Load dependencies ------------------------------------------------------------

library(caret)
library(xgboost)
library(dplyr)
library(readr)
library(ggplot2)
library(MASS)
library(stringr)
library(gtools)
library(stringdist)
library(Biostrings)
library(seqLogo)
#library(devtools)
#load_all("C:/Users/isaac/Documents/Simon_Lab/monaLisa/")
library(monaLisa)
library(GenomicRanges)
library(SummarizedExperiment)
library(ComplexHeatmap)
library(circlize)
library(transite)
library(JASPAR2020)
library(TFBSTools)
library(tidyr)

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}


##### Load and process data #####

promoter_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/promoter_seqs.csv")

threeputr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/threeprimeUTR_seqs.csv")

CDS_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/CDS_seqs.csv")

fiveputr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/fiveprimeUTR_seqs.csv")



ft <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/filtered/RNAdeg_feature_table.csv")

# Filter out low confidence ish
ft <- ft %>%
  filter(avg_lkd_se < exp(-2))

### Combine motif and lower-res data
combined_ft <- ft %>%
  inner_join(promoter_seq %>%
               dplyr::rename(promoter_seq = seq),
             by = "transcript_id") %>%
  inner_join(threeputr_seq %>%
               dplyr::rename(threeputr_seq = seq),
             by = "transcript_id") %>%
  inner_join(fiveputr_seq %>%
               dplyr::rename(fiveputr_seq = seq),
             by = "transcript_id") %>%
  inner_join(CDS_seq %>%
               dplyr::rename(CDS_seq = seq),
             by = "transcript_id")

# Bin transcripts
combined_df <- combined_ft %>%
  dplyr::mutate(
    kdeg_bin = ntile(log_kdeg, 10),
    ksyn_bin = ntile(log_ksyn, 10),
    NMD = ifelse(
      NMD_both == max(NMD_both),
      TRUE,
      FALSE
    )
  ) %>%
  filter(NMD == FALSE) %>%
  dplyr::select(-strand.y) %>%
  dplyr::rename(strand = strand.x)



# Derpy motif counting and search ----------------------------------------------
# I want to simplify shit and just look for known motifs in the UTRs of RNA and
# see the extent to which counts of these motifs is correlated with estimated
# kdeg.




##### Motif counting #####

seq_to_use <- "threeputr_seq"
file_prefix <- "NoNMD_threepUTR_9merctl"
length_xlab <- "log(3'UTR length)"


### HEK293-T miRNA sequences
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
)

count_df <- combined_df

# Takes a hot second
count_df$contains_seed <- sapply(
  combined_df[[seq_to_use]],
  function(query_string) {
    any(sapply(
      miR_seeds$sequence,
      function(seed){
        str_detect(query_string, seed)
      }
    ))
  }
)


### Custom from paper
ARE <- gsub("U","T","UAUUUAUUUAUUU")
Pumilio <- "TGTAAATA"
let7 <- gsub("U", "T","GAGGUAG")
random13 <- "GACTGACTG"
random8 <- "GACTGACT"
random6 <- "GACTGA"

drach_1 <- c("TGACA")
drach_2 <- c("TGACT")
drach_3 <- c("TGACC")
drach_4 <- c("TAACA")
drach_5 <- c("TAACT")
drach_6 <- c("TAACC")
drach_7 <- c("CGACA")
drach_2 <- c("CGACT")
drach_3 <- c("CGACC")
drach_4 <- c("CAACA")
drach_5 <- c("CAACT")
drach_6 <- c("CAACC")



count_df <- count_df %>%
  mutate(
    ARE_count = str_count(!!dplyr::sym(seq_to_use),
                          pattern = ARE),
    Pumilio_count = str_count(!!dplyr::sym(seq_to_use),
                              pattern = Pumilio),
    let7_count = str_count(!!dplyr::sym(seq_to_use),
                           pattern = let7),
    random13_count = str_count(!!dplyr::sym(seq_to_use),
                             pattern = random13),
    random8_count = str_count(!!dplyr::sym(seq_to_use),
                               pattern = random8),
    random6_count = str_count(!!dplyr::sym(seq_to_use),
                               pattern = random6),
    seq_len = str_length(!!dplyr::sym(seq_to_use)),
    GC_cont = (str_count(!!dplyr::sym(seq_to_use), "G") +
      str_count(!!dplyr::sym(seq_to_use), "C")) / seq_len
  ) %>%
  mutate(
    ARE_count_norm = ARE_count  / (seq_len / 1000),
    Pumilio_count_norm = Pumilio_count  / (seq_len / 1000),
    let7_count_norm = let7_count  / (seq_len / 1000)
  ) %>%
  dplyr::filter(seq_len > 100)

count_df <- count_df %>%
  mutate(
    contains_any = factor(ifelse(
      ARE_count + Pumilio_count + let7_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE)),
    contains_ARE = factor(ifelse(
      ARE_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE)),
    contains_Pumilio = factor(ifelse(
      Pumilio_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE)),
    contains_let7 = factor(ifelse(
      let7_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE)),
    contains_random13 = factor(ifelse(
      random13_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE)),
    contains_random8 = factor(ifelse(
      random8_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE)),
    contains_random6 = factor(ifelse(
      random6_count > 0,
      TRUE,
      FALSE
    ),
    c(FALSE, TRUE))
  )

# count_df %>%
#   ggplot(
#     aes(x = contains_any,
#         y = log_kdeg)
#   ) +
#   geom_violin(fill = 'darkgray') +
#   geom_boxplot(
#     outlier.color = NA,
#     color = 'black',
#     fill = 'white',
#     width = 0.3
#   ) +
#   theme_classic() +
#   scale_color_viridis_c() +
#   xlab("log(motif count)") +
#   ylab("log(kdeg)")


count_tidy <- count_df %>%
  dplyr::mutate(
    contains_seed = factor(contains_seed,
                           c(FALSE, TRUE))
  ) %>%
  dplyr::select(starts_with("contains"), "log_kdeg", "transcript_id",
                "seqnames", "old_gene_id") %>%
  pivot_longer(
    cols = starts_with("contains"),
    names_to = "motif",
    values_to = "contains",
    names_pattern = "contains_(.*)"
  )

count_tidy %>%
  dplyr::filter(motif != "any") %>%
  dplyr::mutate(
    motif = factor(motif,
                   levels = c(
                     "seed",
                     "random6",
                     "ARE",
                     "random13",
                     "let7",
                     "Pumilio",
                     "random8"
                   ))
  ) %>%
  dplyr::group_by(motif, contains) %>%
  dplyr::summarise(
    count = n(),
    .groups = "drop")


count_tidy <- count_tidy %>%
  dplyr::filter(motif != "any") %>%
  dplyr::mutate(
    motif = factor(motif,
                   levels = c(
                     "seed",
                     "random6",
                     "ARE",
                     "random13",
                     "let7",
                     "Pumilio",
                     "random8"
                   ))
  )

counts_for_plot <- count_tidy %>%
  dplyr::group_by(
    motif, contains
  ) %>%
  dplyr::summarise(
    count = dplyr::n()
  )


gmotif_kdeg <-
  ggplot(
    count_tidy,
    aes(x = motif,
        y = log_kdeg,
        fill = contains)
  ) +
  geom_boxplot(
    outlier.color = NA,
    color = 'black',
    width = 0.3,
    notch = FALSE
  ) +
  scale_fill_manual(
    values = c("darkgray", "darkgreen")
  ) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("log(motif count)") +
  ylab("log(kdeg)") +
  geom_text(
    data = counts_for_plot,
    aes(label = count,
        y = max(count_tidy$log_kdeg) + 0.1),
    position = position_dodge(width = 1),
    size = 2
  )


gutr_len <- count_df %>%
  dplyr::mutate(
    density = get_density(
      x = log(seq_len + 1),
      y = log_kdeg,
      n = 200
    )
  ) %>%
  ggplot(aes(x = log(seq_len + 1),
             y = log_kdeg,
             color = density)) +
  geom_point(size = 0.6) +
  theme_classic() +
  scale_color_viridis_c() +
  xlab(length_xlab) +
  ylab("log(kdeg)")



setwd("C:/Users/isaac/Documents/Simon_Lab/Meetings/Hogg_IWV_2024_11_08/Figures/")
ggsave(
  filename = paste0(file_prefix, "_length_vs_kdeg.png"),
  plot = gutr_len,
  width = 5,
  height = 3.5
)
ggsave(
  filename = paste0(file_prefix, "_motifs_vs_kdeg.png"),
  plot = gmotif_kdeg,
  width = 5,
  height = 3.5
)



sum(count_df$contains_ARE == TRUE)
sum(count_df$contains_ARE == FALSE)


count_df %>%
  mutate(
    density =
      get_density(
        x = GC_cont,
        y = log_kdeg,
        n = 200
      )
  ) %>%
  ggplot(
    aes(x = GC_cont,
        y = log_kdeg,
        color = density)
  ) +
  geom_point() +
  theme_classic() +
  scale_color_viridis_c() +
  xlab("GC content") +
  ylab("log(kdeg)")



count_df %>%
  dplyr::select(
    seqnames,
    old_gene_id,
    transcript_id,
    threeputr_seq
  )

count_df$threeputr_seq[3]

count_df %>% filter(
  transcript_id == "MSTRG.10834.2"
) %>%
  dplyr::select(
    threeputr_seq
  ) %>%
  unlist() %>%
  unname()

gtf <- rtracklayer::import("C:/Users/isaac/Box/TimeLapse/Annotation_gamut/Annotations/factR2/mix_trimmed/mix_trimmed_factR2.gtf")

as_tibble(gtf) %>%
  filter(transcript_id == "MSTRG.10834.2" &
           type %in% c("transcript", "CDS")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(location = paste0(seqnames, ":", start, "-",end)) %>%
  dplyr::select(seqnames, location, type,
                start, end,
                strand, transcript_id)

# Functionalize monaLisa pipeline ----------------------------------------------

# A
# C
# G
# T

drach_vect <- c(
  0, 0.5, 1, 0, 0.428,
  0.5, 0, 0, 1, 0.142,
  0, 0.5, 0, 0, 0,
  0.5, 0, 0, 0, 0.285
)

drach_vect <- EZbakR:::logit(drach_vect)
drach_vect <- case_when(
  drach_vect == Inf ~ 4,
  drach_vect == -Inf ~ -4,
  .default = drach_vect
)

DRACH_matrix <- matrix(
  c(-4, 0, 4, -4, -0.29,
    0, -4, -4, 4, -1.8,
    -4, 0, -4, -4, -4,
    0, -4, -4, -4, -0.92),
  nrow = 4,
  ncol = 5,
  byrow = TRUE
)



##### Functions #####

get_motifs <- function(strategy = c("JASPAR",
                                    "transite")){

  strategy <- match.arg(strategy)

  if(strategy == "JASPAR"){

    TF_pwmList <- getMatrixSet(JASPAR2020, list(matrixtype = "PWM", tax_group = "vertebrates"))

    return(TF_pwmList)

  }

  if(strategy == "transite"){

    data(motifs)

    ## Try to create PFMatrices
    pwm_list <- vector(
      mode = "list",
      length = length(motifs) + 3
    )

    list_names <- c()

    for(m in seq_along(motifs)){

      motif <- motifs[[m]]

      matrix <- motif@matrix %>%
        as.matrix()
      colnames(matrix) <- c("A", "C", "G", "T")

      pwm_list[[m]] <- PWMatrix(
        ID = paste0(motif@id, "_", m),
        name = paste(motif@rbps, collapse = "_"),
        matrixClass = "Unknown",
        strand = "+",
        bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
        tags = list(
          species = motif@species,
          type = motif@type
        ),
        profileMatrix = t(matrix),
        pseudocounts = numeric()
      )

      list_names[m] <- paste0(motif@id, "_", m)

    }


    ### Manually add custom motif matrices
    ARE_matrix <- matrix(
      c(0, 0.6, 0, 0.6, 0, 0, 0.6, 0, 0.6, 0,
        -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
        -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
        1.5, 0.6, 1.5, 0.6, 1.5, 1.5, 0.6, 1.5, 0.6, 1.5),
      nrow = 4,
      ncol = 10,
      byrow = TRUE
    )
    rownames(ARE_matrix) <- c("A", "C", "G", "T")

    pwm_list[[m+1]] <- PWMatrix(
      ID = "ARE_custom",
      name = "ARE",
      matrixClass = "Unknown",
      strand = "+",
      bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
      tags = list(
        species = motif@species,
        type = "custom"
      ),
      profileMatrix = ARE_matrix,
      pseudocounts = numeric()
    )

    Pumilio_matrix <- matrix(
      c(-4, -4, -4, 3, 0, 1, -4, 3,
        -4, -4, -4, -4, 0, 0.25, -4, -4,
        -4, 3, -4, -4, -4, -4, -4, -4,
        4, -4, 3, -4, 0, 0, 3, -4),
      nrow = 4,
      ncol = 8,
      byrow = TRUE
    )
    rownames(Pumilio_matrix) <- c("A", "C", "G", "T")


    pwm_list[[m+2]] <- PWMatrix(
      ID = "Pumilio_custom",
      name = "Pumilio",
      matrixClass = "Unknown",
      strand = "+",
      bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
      tags = list(
        species = motif@species,
        type = "custom"
      ),
      profileMatrix = Pumilio_matrix,
      pseudocounts = numeric()
    )


    DRACH_matrix <- matrix(
      c(-4, 0, 4, -4, -0.29,
        0, -4, -4, 4, -1.8,
        -4, 0, -4, -4, -4,
        0, -4, -4, -4, -0.92),
      nrow = 4,
      ncol = 5,
      byrow = TRUE
    )
    rownames(DRACH_matrix) <- c("A", "C", "G", "T")


    pwm_list[[m+3]] <- PWMatrix(
      ID = "DRACH_custom",
      name = "DRACH",
      matrixClass = "Unknown",
      strand = "+",
      bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
      tags = list(
        species = motif@species,
        type = "custom"
      ),
      profileMatrix = DRACH_matrix,
      pseudocounts = numeric()
    )

    names(pwm_list) <- c(list_names, "ARE_custom", "Pumilio_custom",
                         "DRACH_custom")

    pwmList <- do.call(
      PWMatrixList,
      pwm_list
    )

    return(pwmList)

  }


}

get_seqs_and_bins <- function(seqdf, subseq_len = 400,
                              bins_to_use = 2:9,
                              seq_type = c("threeputr",
                                           "fiveputr",
                                           "promoter",
                                           "CDS"),
                              bin_type = c("kdeg", "ksyn")){

  bin_type <- match.arg(bin_type)
  bin_col <- paste0(bin_type, "_bin")

  seq_type <- match.arg(seq_type)
  seq_col <- paste0(seq_type, "_seq")

  subset <- seqdf %>%
    dplyr::filter(!!dplyr::sym(bin_col) %in% bins_to_use) %>%
    filter(str_length(!!dplyr::sym(seq_col)) > subseq_len + 10) %>%
    dplyr::mutate(
      subseq = str_sub(!!dplyr::sym(seq_col),
                       round((str_length(!!dplyr::sym(seq_col)) - subseq_len) / 2),
                       round(str_length(!!dplyr::sym(seq_col)) - ((str_length(!!dplyr::sym(seq_col)) - subseq_len) / 2)))
    )

  seqs <- Biostrings::DNAStringSet(subset$subseq)
  bins <- factor(subset[[bin_col]])

  return(
    list(
      seqs = seqs,
      bins = bins
    )
  )

}


# data(motifs)
#
# lapply(
#   motifs,
#   function(x){
#     return(grepl("ELAV", x@rbps))
#   }
# )
#
# motifs[[1]]@rbps
# motifs[[2]]@matrix


##### Get motifs and sequences ######

pwmList <- get_motifs(strategy = "transite")

data_to_analyze <- get_seqs_and_bins(
  seqdf = combined_df,
  bins_to_use = c(2, 9),
  seq_type = "threeputr",
  subseq_len = 500,
  bin_type = "kdeg"
)

##### Run monaLisa and assess #####

se <- calcBinnedMotifEnrR(
  seqs = data_to_analyze$seqs,
  bins = data_to_analyze$bins,
  pwmL = pwmList,
  verbose = TRUE
)

sel <- apply(assay(se, "negLog10P"), 1,
             function(x) max(abs(x), 0, na.rm = TRUE)) > 2


plotMotifHeatmaps(x = se[sel,], which.plots = c("log2enr", "negLog10P"),
                  width = 2.0, cluster = TRUE, maxEnr = 2, maxSig = 10,
                  show_motif_GC = TRUE,
                  row_names_gp = gpar(fontsize = 4))



sek <- calcBinnedKmerEnr(
  seqs = data_to_analyze$seqs,
  bins = data_to_analyze$bins,
  verbose = TRUE
)

selk <- apply(assay(sek, "negLog10P"), 1,
             function(x) max(abs(x), 0, na.rm = TRUE)) > 2.5


plotMotifHeatmaps(x = sek[selk,], which.plots = c("log2enr", "negLog10P"),
                  width = 2.0, cluster = TRUE, maxEnr = 2, maxSig = 10,
                  show_motif_GC = TRUE,
                  row_names_gp = gpar(fontsize = 4))




# Motif analysis try 1 ---------------------------------------------------------

##### LOAD DATA #####

promoter_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/promoter_seqs.csv")

threeputr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/threeprimeUTR_seqs.csv")

CDS_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/CDS_seqs.csv")

fiveputr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/fiveprimeUTR_seqs.csv")



ft <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/RNAdeg_feature_table.csv")

# Filter out low confidence ish
ft <- ft %>%
  filter(avg_lkd_se < exp(-2))

### Combine motif and lower-res data
combined_ft <- ft %>%
  inner_join(promoter_seq %>%
               dplyr::rename(promoter_seq = seq),
             by = "transcript_id") %>%
  inner_join(threeputr_seq %>%
               dplyr::rename(threeputr_seq = seq),
             by = "transcript_id") %>%
  inner_join(fiveputr_seq %>%
               dplyr::rename(fiveputr_seq = seq),
             by = "transcript_id") %>%
  inner_join(CDS_seq %>%
               dplyr::rename(CDS_seq = seq),
             by = "transcript_id")

# Bin transcripts
combined_df <- combined_ft %>%
  dplyr::mutate(
    kdeg_bin = ntile(log_kdeg, 10)
  )


# min(str_length(combined_df$threeputr_seq))
#
# write_csv(combined_df,
#           file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/Seqs_and_kdeg_combined_table.csv")
#

##### Trim 3'UTR and perform motif analysis #####

threeUTR_subset <- combined_df %>%
  filter(str_length(threeputr_seq) > 103) %>%
  dplyr::mutate(
    threeputr_start = str_sub(threeputr_seq, 4, 103),
    threeputr_end = str_sub(threeputr_seq, str_length(threeputr_seq) - 99,
                            str_length(threeputr_seq))
  ) %>%
  dplyr::select(
    seqnames, transcript_id, log_kdeg, kdeg_bin,
    threeputr_start, threeputr_end, threeputr_seq
  )

# bins <- unique(threeUTR_subset$kdeg_bin)
# startPSSM_list <- vector(mode = "list",
#                     length = length(bins))
# endPSSM_list <- startPSSM_list
# for(b in seq_along(bins)){
#
#   start_seqs <- threeUTR_subset$threeputr_start[threeUTR_subset$kdeg_bin == bins[b]]
#   start_string_set <- Biostrings::DNAStringSet(start_seqs)
#   startPSSM_list[[b]] <- consensusMatrix(start_string_set,
#                           as.prob = TRUE)
#
#
#   end_seqs <- threeUTR_subset$threeputr_end[threeUTR_subset$kdeg_bin == bins[b]]
#   end_string_set <- Biostrings::DNAStringSet(end_seqs)
#   endPSSM_list[[b]] <- consensusMatrix(end_string_set,
#                                          as.prob = TRUE)
#
#
#
# }
#
# startPSSM_list[[1]]
#
#
# seqLogo(startPSSM_list[[7]][1:4,])


##### Mona lisa kmer enrichment #####
threeUTR_subset <- combined_df %>%
  dplyr::filter(kdeg_bin %in% 2:9) %>%
  filter(str_length(threeputr_seq) > 500) %>%
  dplyr::mutate(
    threeputr_subseq = str_sub(threeputr_seq,
                               str_length(threeputr_seq) - 409,
                               str_length(threeputr_seq) - 10)
  ) %>%
  dplyr::select(
    seqnames, transcript_id, log_kdeg, kdeg_bin,
    threeputr_subseq, threeputr_seq
  )

threeUTR_subset %>%
  group_by(kdeg_bin) %>%
  summarise(
    avg_len = mean(str_length(threeputr_seq)),
    min_len = min(str_length(threeputr_seq)),
    max_len = max(str_length(threeputr_seq)),
    count = n()
  )

seqs <- Biostrings::DNAStringSet(threeUTR_subset$threeputr_subseq)
bins <- factor(threeUTR_subset$kdeg_bin)

sekm <- calcBinnedKmerEnr(
  seqs,
  bins,
  background = "allBins",
  includeRevComp = FALSE
)

selkm <- apply(assay(sekm, "negLog10P"), 1,
               function(x) max(abs(x), 0, na.rm = TRUE)) > 2
sekmSel <- sekm[selkm, ]


assay(sekm, "negLog10Padj")


rowData(sekmSel)
assay(sekmSel, "negLog10Padj")


plotBinDiagnostics(seqs = seqs, bins = bins, aspect = "dinucfreq")


plotMotifHeatmaps(x = sekmSel, which.plots = c("log2enr", "negLog10Padj"),
                  width = 2.0, cluster = TRUE, maxEnr = 2, maxSig = 10,
                  width.seqlogo = 0.25,
                  show_motif_GC = TRUE,
                  row_names_gp = gpar(fontsize = 6))


# monaLisa analysis ------------------------------------------------------------


##### Load data #####

promoter_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/promoter_seqs.csv")

threeputr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/threeprimeUTR_seqs.csv")

CDS_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/CDS_seqs.csv")

fiveputr_seq <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/fiveprimeUTR_seqs.csv")



ft <- read_csv("C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/mix_trimmed/RNAdeg_feature_table.csv")

# Filter out low confidence ish
ft <- ft %>%
  filter(avg_lkd_se < exp(-2))

### Combine motif and lower-res data
combined_ft <- ft %>%
  inner_join(promoter_seq %>%
               dplyr::rename(promoter_seq = seq),
             by = "transcript_id") %>%
  inner_join(threeputr_seq %>%
               dplyr::rename(threeputr_seq = seq),
             by = "transcript_id") %>%
  inner_join(fiveputr_seq %>%
               dplyr::rename(fiveputr_seq = seq),
             by = "transcript_id") %>%
  inner_join(CDS_seq %>%
               dplyr::rename(CDS_seq = seq),
             by = "transcript_id")

# Bin transcripts
combined_df <- combined_ft %>%
  dplyr::mutate(
    kdeg_bin = ntile(log_kdeg, 10)
  )


##### Grab middle portion of 3'UTR seqs #####


subseq_len <- 400
threeUTR_subset <- combined_df %>%
  dplyr::filter(kdeg_bin %in% 2:9) %>%
  filter(str_length(threeputr_seq) > 500) %>%
  dplyr::mutate(
    threeputr_subseq = str_sub(threeputr_seq,
                               round((str_length(threeputr_seq) - subseq_len) / 2),
                               round(str_length(threeputr_seq) - ((str_length(threeputr_seq) - subseq_len) / 2)))
  ) %>%
  dplyr::select(
    seqnames, transcript_id, log_kdeg, kdeg_bin,
    threeputr_subseq, threeputr_seq
  )

seqs <- Biostrings::DNAStringSet(threeUTR_subset$threeputr_subseq)
bins <- factor(threeUTR_subset$kdeg_bin)


##### Compile motifs to look for #####

data(motifs)

motifs

## Try to create PFMatrices
pwm_list <- vector(
  mode = "list",
  length = length(motifs) + 2
)

list_names <- c()

for(m in seq_along(motifs)){

  motif <- motifs[[m]]

  matrix <- motif@matrix %>%
    as.matrix()
  colnames(matrix) <- c("A", "C", "G", "T")

  pwm_list[[m]] <- PWMatrix(
    ID = paste0(motif@id, "_", m),
    name = paste(motif@rbps, collapse = "_"),
    matrixClass = "Unknown",
    strand = "+",
    bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
    tags = list(
      species = motif@species,
      type = motif@type
    ),
    profileMatrix = t(matrix),
    pseudocounts = numeric()
  )

  list_names[m] <- paste0(motif@id, "_", m)

}


### Manually add custom motif matrices
ARE_matrix <- matrix(
  c(0, 0.6, 0, 0.6, 0, 0, 0.6, 0, 0.6, 0,
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
    1.5, 0.6, 1.5, 0.6, 1.5, 1.5, 0.6, 1.5, 0.6, 1.5),
  nrow = 4,
  ncol = 10,
  byrow = TRUE
)
rownames(ARE_matrix) <- c("A", "C", "G", "T")

pwm_list[[m+1]] <- PWMatrix(
  ID = "ARE_custom",
  name = "ARE",
  matrixClass = "Unknown",
  strand = "+",
  bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
  tags = list(
    species = motif@species,
    type = "custom"
  ),
  profileMatrix = ARE_matrix,
  pseudocounts = numeric()
)

Pumilio_matrix <- matrix(
  c(-4, -4, -4, 3, 0, 1, -4, 3,
    -4, -4, -4, -4, 0, 0.25, -4, -4,
    -4, 3, -4, -4, -4, -4, -4, -4,
    4, -4, 3, -4, 0, 0, 3, -4),
  nrow = 4,
  ncol = 8,
  byrow = TRUE
)
rownames(Pumilio_matrix) <- c("A", "C", "G", "T")


pwm_list[[m+2]] <- PWMatrix(
  ID = "Pumilio_custom",
  name = "Pumilio",
  matrixClass = "Unknown",
  strand = "+",
  bg=c(A=0.25, C=0.25, G=0.25, T=0.25),
  tags = list(
    species = motif@species,
    type = "custom"
  ),
  profileMatrix = Pumilio_matrix,
  pseudocounts = numeric()
)

names(pwm_list) <- c(list_names, "ARE_custom", "Pumilio_custom")

pwmList <- do.call(
  PWMatrixList,
  pwm_list
)

# pwmList@listData


##### Run monaLisa #####


se <- calcBinnedMotifEnrR(
  seqs = seqs,
  bins = bins,
  pwmL = pwmList,
  verbose = TRUE
)

se


##### Assess results #####

assay(se, "log2enr")["Pumilio_custom",]
assay(se, "log2enr")["ARE_custom",]

sel <- apply(assay(se, "negLog10P"), 1,
             function(x) max(abs(x), 0, na.rm = TRUE)) > 2


plotMotifHeatmaps(x = se[sel,], which.plots = c("log2enr", "negLog10P"),
                  width = 2.0, cluster = TRUE, maxEnr = 2, maxSig = 10,
                  show_motif_GC = TRUE,
                  row_names_gp = gpar(fontsize = 4),
                  show_seqlogo = TRUE)


pwms <- getMatrixSet(JASPAR2020, list(matrixtype = "PWM", tax_group = "vertebrates"))
pwms$MA0004.1


##### Assess PWMs, especially to check custom ones #####

mat_to_test <- DRACH_matrix

mat_to_test

for(c in 1:ncol(mat_to_test)){

  mat_to_test[,c] <- exp(mat_to_test[,c])/sum(exp(mat_to_test[,c]))

}

mat_to_test

seqLogo::seqLogo(mat_to_test)


##### Run mona Lisa with TF motifs as a negative control of sort #####

TF_pwmList <- getMatrixSet(JASPAR2020, list(matrixtype = "PWM", tax_group = "vertebrates"))


se <- calcBinnedMotifEnrR(
  seqs = seqs,
  bins = bins,
  pwmL = TF_pwmList,
  verbose = TRUE
)

sel <- apply(assay(se, "negLog10P"), 1,
             function(x) max(abs(x), 0, na.rm = TRUE))


plotMotifHeatmaps(x = se[sel,], which.plots = c("log2enr", "negLog10P"),
                  width = 2.0, cluster = TRUE, maxEnr = 2, maxSig = 10,
                  show_motif_GC = TRUE,
                  row_names_gp = gpar(fontsize = 4),
                  show_seqlogo = TRUE)




