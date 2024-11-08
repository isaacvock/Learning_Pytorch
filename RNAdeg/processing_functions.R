### PURPOSE OF THIS SCRIPT
## Collection of functions to process kdeg estimation data for various
## model inputs.

# Get simple feature table -----------------------------------------------------


assemble_data <- function(gtf, factr_transcript_file,
                          ezbdo, outdir,
                          isoform_filter = NULL,
                          read_cutoff = 25,
                          TPM_cutoff = 2,
                          returnTable = TRUE,
                          filename = "RNAdeg_dataset",
                          nmd_diff_cutoff = -1, nmd_padj_cutoff = 0.01,
                          sample_pattern = "DMSO",
                          gtf_feature_cols = c("seqnames",
                                               "transcript_id",
                                               "gene_id",
                                               "gene_name",
                                               "old_gene_id",
                                               "strand")){

  ##### Get features from the factR2 annotation
  gtf_df <- as_tibble(gtf)

  if(!is.null(isoform_filter)){

    gtf_df <- gtf_df %>%
      dplyr::inner_join(
        isoform_filter,
        by = "transcript_id"
      )

  }

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
    by = c(gtf_feature_cols[gtf_feature_cols != "old_gene_id"])
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

  if(returnTable){
    return(combined_table)
  }


}


# Feature engineering
clean_feature_table <- function(feature_table, output,
                                returnTable = TRUE,
                                se_cutoff = 0.25){


  feature_table_processed <- feature_table  %>%
    dplyr::filter(avg_lkd_se < se_cutoff) %>% # 90%
    dplyr::mutate(NMD_both = EZbakR_nmd & (nmd == "yes"),
                  log10_avg_TPM = log10(avg_TPM),
                  log10_avg_reads = log10(avg_reads),
                  log10_length = log10(effective_length),
                  log10_numexons = log10(num_exons),
                  log10_3primeUTR = log10(`3'UTR_length` + 1),
                  log10_5primeUTR = log10(fiveprimeUTR_lngth + 1)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(NMD_both = as.numeric(NMD_both)) %>%
    dplyr::mutate(NMD_both_z = (NMD_both - mean(NMD_both)) / sd(NMD_both),
                  log10_avg_TPM_z = (log10_avg_TPM - mean(log10_avg_TPM)) / sd(log10_avg_TPM),
                  log10_avg_reads_z = (log10_avg_reads - mean(log10_avg_reads)) / sd(log10_avg_reads),
                  log10_length_z = (log10_length - mean(log10_length))/sd(log10_length),
                  log10_numexons_z = (log10_numexons - mean(log10_numexons)) / sd(log10_numexons),
                  log10_3primeUTR_z = (log10_3primeUTR - mean(log10_3primeUTR)) / sd(log10_3primeUTR),
                  log10_5primeUTR_z = (log10_5primeUTR - mean(log10_5primeUTR)) / sd(log10_5primeUTR),
                  log_kdeg_z = (log_kdeg - mean(log_kdeg))/sd(log_kdeg),
                  log_ksyn_z = (log_ksyn - mean(log_ksyn))/sd(log_ksyn))


  fwrite(feature_table_processed,
            output)

  if(returnTable){
    return(feature_table_processed)
  }

}



# Get sequence information -----------------------------------------------------

get_sequencing_info <- function(gtf,
                                outdir,
                                returnTable = TRUE){

  gtf <- gtf[strand(gtf) != "*"]

  # Not really sure why I need to do this, but I get a "ID missing for some
  # genes" error if I don't
  mcols(gtf)$ID <- NULL
  txdb <- GenomicFeatures::makeTxDbFromGRanges(gtf)



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
            file = "C:/Users/isaac/Documents/ML_pytorch/Data/RNAdeg/CDS_seqs.csv")
  write_csv(cds_df,
            file = paste0(outdir, "/CDS_seqs.csv"))


  ### Optional output
  if(returnTable){

    return(
      list(ThreePrimeUTR = threeprimeutr_df,
           FivePrimeUTR = fiveprimeutr_df,
           CDS = cds_df)
    )

  }

}
