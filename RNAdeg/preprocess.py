'''
Script for processing data to develop a simple model
for predicting RNA stability with various annotation
and genomic features.
'''

# LOAD DEPENDENCIES ---------------------------------

import pandas as pd
import polars as pl
import gffutils as gf
import Bio
import os



# LOAD ANNOTATION AND GENOME FASTA -----------------

os.chdir("G:/Shared drives/Matthew_Simon/IWV/Test_datasets/TimeLapse/annotation/")
db = gf.create_db('genome.gtf', 'C:/Users/isaac/Documents/ML_pytorch/Data/chr21.db')

os.chdir("C:/Users/isaac/Documents/ML_pytorch/Data/")
db = gf.FeatureDB('chr21.db')

gene = db['MSTRG.9150']
gene.start
gene.end

for feature in db.all_features():

    feature.start
    feature.sequence
