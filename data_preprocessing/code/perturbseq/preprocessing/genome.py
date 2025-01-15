# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: crispyKC
#     language: python
#     name: python3
# ---

# %%
import crispyKC as ckc

import subprocess
from pathlib import Path

# %%
file = Path(ckc.get_data()) / "singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly.fa.gz"
if not file.exists():
    subprocess.run([
        "wget", 
        "http://ftp.ensembl.org/pub/release-111/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna_rm.primary_assembly.fa.gz", 
        "-O", str(file)
    ])

# %%
file = Path(ckc.get_data()) / "singlecell/references/Mus_musculus.GRCm39.111.chr.gtf.gz"
if not file.exists():
    subprocess.run([
        "wget", 
        "http://ftp.ensembl.org/pub/release-111/gtf/mus_musculus/Mus_musculus.GRCm39.111.chr.gtf.gz", 
        "-O", str(file)
    ])

# %%
import pandas as pd
import pysam

# %% [markdown]
# ## Fasta

# %%
# with open(f"{ckc.get_data()}/singlecell/references/CAS9.txt") as f:
#     cas9 = f.read().strip().replace("\n", "")
# n = len(cas9)
# import gzip
# with open(f"{ckc.get_data()}/singlecell/references/CAS9.fa", "w") as f:
#     f.write(">CAS9\n")
#     for i in range(0, len(cas9), 60):
#         f.write(cas9[i:i+60] + "\n")

# # %%
# with open(f"{ckc.get_data()}/singlecell/references/pBA900.txt") as f:
#     pba900 = f.read().strip().replace("\n", "")
# n = len(pba900)
# import gzip
# with open(f"{ckc.get_data()}/singlecell/references/pBA900.fa", "w") as f:
#     f.write(">pBA900\n")
#     for i in range(0, len(pba900), 60):
#         f.write(pba900[i:i+60] + "\n")

# %%
# Open the CAS9.fa.gz file and read its content
with open(f"{ckc.get_data()}/singlecell/references/CAS9.fa", "rt") as f:
    # Skip the first line (header) and read the sequence
    next(f)
    cas9 = f.read().strip().replace("\n", "")

with open(f"{ckc.get_data()}/singlecell/references/pBA900.fa") as f:
    next(f)
    pba900 = f.read().strip().replace("\n", "")

# %%
# !cp f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly.fa.gz" f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9.fa.gz"
# !gzip f"{ckc.get_data()}/singlecell/references/CAS9.fa" -c >> f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9.fa.gz"

# %%
# !gunzip f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9.fa.gz" --force

# %%
# !cp f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9.fa" f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9_pBA900.fa"
# !cat f"{ckc.get_data()}/singlecell/references/pBA900.fa" >> f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9_pBA900.fa"

# %% [markdown]
# ## Create GTF

# %%
# 7894 dark green = WPRE
# 8498 purple = bGHpolyA
# 8774 dark red = pPGK
# 9342 yellow = Neo(mycin)

# %%
import pandas as pd
custom_gtf = pd.DataFrame(
    [
        ["CAS9", "unknown", "exon", 1, len(cas9), ".", "+", ".", "gene_id \"CAS9\"; transcript_id \"CAS9\"; gene_name \"CAS9\"; gene_biotype \"protein_coding\";"],
        # ["pBA900", "unknown", "exon", 1, len(pba900), ".", "+", ".", "gene_id \"pBA900\"; transcript_id \"pBA900\"; gene_name \"pBA900\"; gene_biotype \"protein_coding\";"]
    ],
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
)

# %%
import gzip

# %%
# !cp {ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr.gtf.gz {ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9.gtf.gz
with gzip.open(f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9.gtf.gz", "a") as f:
    custom_gtf.to_csv(f, sep="\t", header=False, index=False, quoting=0)

# %%
# !zcat {ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9.gtf.gz | tail -n 10

# %%
# !gunzip {ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9.gtf.gz --force

# %%
custom_gtf = pd.DataFrame(
    [
        ["CAS9", "unknown", "exon", 1, len(cas9), ".", "+", ".", "gene_id \"CAS9\"; transcript_id \"CAS9\"; gene_name \"CAS9\"; gene_biotype \"protein_coding\";"],
        ["pBA900", "unknown", "exon", 1, len(pba900), ".", "+", ".", "gene_id \"pBA900\"; transcript_id \"pBA900\"; gene_name \"pBA900\"; gene_biotype \"protein_coding\";"]
    ],
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
)

# %%
# !cp f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr.gtf.gz" f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9_pBA900.gtf.gz"
with gzip.open(f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9_pBA900.gtf.gz", "a") as f:
    custom_gtf.to_csv(f, sep="\t", header=False, index=False, quoting=0)

# %%
# !gunzip f"{ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9_pBA900.gtf.gz" --force

# %% [markdown]
# ## Make reference

# %%
cellranger = ckc.get_software() / "cellranger-8.0.0" / "bin" / "cellranger"
cellranger

# %%
# !rm -rf f"{ckc.get_data()}/singlecell/references/GRCm39_CAS9"

# %%
print(f"""{cellranger} mkref --genome=GRCm39_CAS9 \\
    --fasta={ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9.fa \\
    --genes={ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9.gtf \\
    --output-dir={ckc.get_data()}/singlecell/references/GRCm39_CAS9
""") 

# %%
print(f"""{cellranger} mkref --genome=GRCm39_CAS9_pBA900 \\
    --fasta={ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.dna_rm.primary_assembly_CAS9_pBA900.fa \\
    --genes={ckc.get_data()}/singlecell/references/Mus_musculus.GRCm39.111.chr_CAS9_pBA900.gtf \\
    --output-dir={ckc.get_data()}/singlecell/references/GRCm39_CAS9_pBA900
""") 

# %%
ckc.get_data()
