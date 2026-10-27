import os
from Bio import SeqIO

input_file = 'gene_lists.txt'
gbk_folder = '.'
output_file = 'output_genes_list_1.txt'
missing_files = 'missing_files_genes_list_1.txt'
missing_genes = 'missing_genes_genes_list_1.txt'

def parse_gbk_features_with_order(gbk_path):

    gene_list = []
    gene_dict = {}
    with open(gbk_path, 'r') as handle:
        for record in SeqIO.parse(handle, "genbank"):
            contig = record.name
            for feature in record.features:
                if "locus_tag" in feature.qualifiers:
                    locus_tag = feature.qualifiers["locus_tag"][0]
                    start = int(feature.location.start) + 1  # GenBank usa coordinate 0-based
                    end = int(feature.location.end)
                    strand = feature.location.strand
                    gene_list.append((locus_tag, start, end, strand, contig))

    for idx, (locus_tag, start, end, strand, contig) in enumerate(gene_list):
        gene_dict[locus_tag] = (start, end, strand, contig, idx)

    return gene_list, gene_dict

with open(input_file) as in_f:
    target_genes = [line.strip() for line in in_f if line.strip()]

already_written = set()
future_genes_set = set(target_genes)

current_gbk_prefix = None
gene_list = []
gene_dict = {}

genes_to_write = []

with open(missing_files, 'w') as miss_files_f, \
     open(missing_genes, 'w') as miss_genes_f:

    for gene_name in target_genes:
        if "_" not in gene_name:
            continue

        gbk_prefix = gene_name.split('_')[0]
        gbk_filename = f"{gbk_prefix}.gbk"
        gbk_path = os.path.join(gbk_folder, gbk_filename)

        if gbk_prefix != current_gbk_prefix:
            if not os.path.isfile(gbk_path):
                miss_files_f.write(f"{gbk_filename}\n")
                current_gbk_prefix = None
                gene_list = []
                gene_dict = {}
                continue
            gene_list, gene_dict = parse_gbk_features_with_order(gbk_path)
            current_gbk_prefix = gbk_prefix

        if gene_name not in gene_dict:
            miss_genes_f.write(f"{gbk_filename}\t{gene_name}\n")
            continue

        _, _, _, contig, idx = gene_dict[gene_name]

        for offset in range(-3, 4):
            neighbor_idx = idx + offset
            if neighbor_idx < 0 or neighbor_idx >= len(gene_list):
                continue

            neighbor_gene = gene_list[neighbor_idx]
            locus_tag, start, end, strand, neighbor_contig = neighbor_gene

            if neighbor_contig != contig:
                continue
            if locus_tag in already_written:
                continue
            if offset != 0 and locus_tag in future_genes_set:
                continue

            genes_to_write.append((locus_tag, start, end, strand, contig))
            already_written.add(locus_tag)

        future_genes_set.discard(gene_name)

genes_to_write.sort(key=lambda x: (x[4], x[1]))

with open(output_file, 'w') as out_f:
    for locus_tag, start, end, strand, contig in genes_to_write:
        strand_symbol = "+" if strand == 1 else "-"
        out_f.write(f"{locus_tag}\t{start}\t{end}\t{strand_symbol}\t{contig}\n")

