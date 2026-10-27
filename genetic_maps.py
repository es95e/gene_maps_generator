#!/usr/bin/env python3
"""
Comparative gene maps of a genomic locus across multiple species.

Run `python genetic_maps.py --help` for the command-line options.
"""
import argparse
import os
import sys
import warnings
from textwrap import wrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrow, Patch, Polygon

# ---------------------------------------------------------------------------
# Font handling: first available font of the list is used (no crash, no
# "findfont" warnings if Times New Roman is missing on Linux/macOS).
# ---------------------------------------------------------------------------
# Colour table used with "color by gene_name" (change here or use --colors)
DEFAULT_COLORS_FILE = "Genes_color_set.csv"

FONT_FALLBACKS = [
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "FreeSerif",
    "DejaVu Serif",  # shipped with matplotlib, always available
]


def set_font(preferred=None):
    """Set the global font to the first available one and return its name."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ([preferred] if preferred else []) + FONT_FALLBACKS
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = [name]
            if name != candidates[0]:
                print(f"[i] Font '{candidates[0]}' not found, using '{name}' instead.")
            return name
    plt.rcParams["font.family"] = ["DejaVu Serif"]
    return "DejaVu Serif"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_gene_name(name):
    if len(name) <= 3:
        return f"${name}$"
    # Italics for genes name
    first_three = name[:3]
    rest = name[3:]
    return f"$\\it{{{first_three}}}${rest}"


def load_csv(file_path):
    return pd.read_csv(file_path)


def load_gene_colors(file_path):
    df_colors = pd.read_csv(file_path)
    return dict(zip(df_colors["gene_name"], df_colors["color_code"]))


def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_missing(val):
    return val is None or (isinstance(val, float) and np.isnan(val)) or \
        (isinstance(val, str) and val.strip() == "")


REQUIRED_INFO_COLS = ["species", "gene_number", "gene_target", "phy", "order",
                      "start", "end", "verse", "gene_name"]
OPTIONAL_INFO_COLS = ["annotation", "strain", "identity", "value_identity"]
REQUIRED_CONTIG_COLS = ["species", "target"]


def prepare_dataframes(df, df_contig, species_reference=None):
    """Validate and normalise the input tables. Returns (df, df_contig, reference)."""
    df = df.copy()
    df_contig = df_contig.copy()

    missing = [c for c in REQUIRED_INFO_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in the locus info table: {missing}")
    missing = [c for c in REQUIRED_CONTIG_COLS if c not in df_contig.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in the contig table: {missing}")
    for c in OPTIONAL_INFO_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["start", "end"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["phy"] = pd.to_numeric(df["phy"], errors="coerce")
    df["verse"] = df["verse"].astype(str).str.strip()
    df["order"] = df["order"].astype(str).str.strip().str.upper()

    # identity must be in [0, 1]; percentages (0-100) are converted automatically
    ident = pd.to_numeric(df["identity"], errors="coerce")
    if ident.notna().any():
        if ident.min() < 0 or ident.max() > 100:
            raise ValueError("Column 'identity' must contain values in [0, 1] (or percentages in [0, 100]).")
        if ident.max() > 1:
            warnings.warn("Column 'identity' has values > 1: interpreted as percentages and divided by 100.")
            ident = ident / 100.0
    df["identity"] = ident

    # reference species: explicit, otherwise the species with the lowest 'phy'
    species_present = df["species"].dropna().unique().tolist()
    if species_reference is None:
        species_reference = (df.dropna(subset=["phy"]).sort_values("phy")["species"].iloc[0]
                             if df["phy"].notna().any() else species_present[0])
        print(f"[i] No reference species given: using '{species_reference}' (lowest 'phy').")
    elif species_reference not in species_present:
        raise ValueError(f"Reference species '{species_reference}' not found. "
                         f"Available species: {species_present}")
    return df, df_contig, species_reference


def make_contig_break_checker(df_contig):
    """
    Return a function break(row_a, row_b) -> bool telling whether a contig
    interruption lies between two genes that are neighbours in the drawing.
    Single criterion used both for the shortened spacing and for the '//' mark.
    The pair is ordered by genomic position (upstream/downstream) and a break
    is reported if:
      - 'continuity' of the downstream gene is "NO" (break before the gene), or
      - 'same_contig -1' of the downstream gene is "NO", or
      - 'same_contig +1' of the upstream gene is "NO".
    """
    idx = df_contig.drop_duplicates("target").set_index("target")

    def find_col(sign):
        for c in idx.columns:
            if "same_contig" in c and c.replace(" ", "").replace("_", "").endswith(sign + "1"):
                return c
        return None

    col_cont = "continuity" if "continuity" in idx.columns else None
    col_minus, col_plus = find_col("-"), find_col("+")

    def is_no(gene_number, col):
        if col is None or gene_number not in idx.index:
            return False
        return str(idx.at[gene_number, col]).strip().upper() == "NO"

    def contig_break(row_a, row_b):
        up, down = (row_a, row_b) if row_a["start"] <= row_b["start"] else (row_b, row_a)
        gu, gd = up["gene_number"], down["gene_number"]
        return is_no(gd, col_cont) or is_no(gd, col_minus) or is_no(gu, col_plus)

    return contig_break


# ---------------------------------------------------------------------------
# Main drawing function
# ---------------------------------------------------------------------------
def draw_all_species_loci(df, df_contig, output_file="mappa_genica_comparativa.png",
                          arrow_line_offset=0.0, inter_species_spacing=0.4,
                          species_reference=None, distance_threshold_bp=10000,
                          highlight_gene=None,
                          highlight_color="yellow",
                          highlight_edge="purple",
                          highlight_linewidth=2.5,
                          color_by_gene_name=False,
                          gene_colors=None,
                          flip_strand_on_reverse=True,
                          link_skip_missing=False):
    """
    flip_strand_on_reverse : if True, species with order == "R" are mirrored AND
        their strand is inverted (verse in the CSV = strand in the genome).
        Set False only if you already inverted 'verse' in the CSV for "R" species.
    link_skip_missing : if True, a gene missing in the adjacent species is linked
        to the next species that carries it. Default False: links are drawn only
        between strictly adjacent species.
    """
    df, df_contig, species_reference = prepare_dataframes(df, df_contig, species_reference)
    contig_break = make_contig_break_checker(df_contig)

    arrow_height = 0.3
    y_base = arrow_height * 2

    if gene_colors is None:
        cmap = plt.get_cmap("tab20")
        unique_keys = list((df["gene_name"] if color_by_gene_name else df["annotation"]).dropna().unique())
        gene_colors = {}
        for i, key in enumerate(unique_keys):
            gene_colors[key] = "#000000" if is_numeric(key) else cmap(i % 20)

    # Species list order
    other_species = df[df["species"] != species_reference][["species", "phy"]].drop_duplicates("species")
    other_species_sorted = other_species.sort_values(by=["phy", "species"])
    species_list = [species_reference] + other_species_sorted["species"].tolist()
    n_species = len(species_list)
    species_to_y = {
        sp: (n_species - 1 - i) * inter_species_spacing + y_base
        for i, sp in enumerate(species_list)
    }
    fig_height = 3 * n_species * (inter_species_spacing / 0.4)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    x_cursor_max = 0

    # positions[species][gene_name] = [ {xs, xe, y, gene_number, identity, value_identity}, ... ]
    positions = {}
    drawn_keys = set()  # gene_name / annotation values actually drawn (for the legend)
    locus_lengths = {}
    info_cols = [c for c in df.columns if c.startswith("gene_at_")] + ["gene_target"]

    for sp in species_list:
        gene_ids = pd.unique(df[df["species"] == sp][info_cols].values.ravel())
        gene_ids = [g for g in gene_ids if pd.notna(g)]
        gene_data = df[df["gene_number"].isin(gene_ids)].drop_duplicates("gene_number")
        if not gene_data.empty:
            locus_lengths[sp] = (gene_data["start"].min(), gene_data["end"].max())
        else:
            locus_lengths[sp] = (0, 0)

    SCALE_FACTOR = 1

    for species in species_list:
        df_sp = df[df["species"] == species].copy()
        order = df_sp["order"].iloc[0] if not df_sp.empty else "F"
        gene_ids = pd.unique(df_sp[info_cols].values.ravel())
        gene_ids = [g for g in gene_ids if pd.notna(g)]
        gene_data = df[df["gene_number"].isin(gene_ids)].drop_duplicates("gene_number")

        if order == "F":
            gene_data = gene_data.sort_values("start")
        else:
            gene_data = gene_data.sort_values("start", ascending=False)

        if gene_data.empty:
            print(f"[!] No gene to display for: {species}")
            continue

        y = species_to_y[species]
        arrow_y = y + arrow_line_offset
        x_cursor = 25
        prev_row = None
        gene_positions_visual = []
        positions[species] = {}

        start_min, end_max = locus_lengths[species]
        locus_len = max(0, end_max - start_min)

        for _, row in gene_data.iterrows():
            gene_number = row["gene_number"]
            gene_name = row["gene_name"] if pd.notna(row["gene_name"]) else None
            start = row["start"]
            end = row["end"]
            strand = row["verse"]
            if order == "R" and flip_strand_on_reverse:
                strand = "-" if strand == "+" else "+"
            width = end - start

            total_scaled_width = width * SCALE_FACTOR
            head_len = max(total_scaled_width * 0.9, 2)
            body_len = total_scaled_width - head_len
            arrow_len = body_len + head_len

            # Single gap criterion (long intergenic distance OR contig break),
            # reused below for the '//' mark.
            gap_before = False
            if prev_row is not None:
                if order == "F":
                    dist_genomic = abs(row["start"] - prev_row["end"])
                else:
                    dist_genomic = abs(prev_row["start"] - row["end"])
                gap_before = dist_genomic > distance_threshold_bp or contig_break(prev_row, row)
                if gap_before:
                    x_cursor += SCALE_FACTOR * 500
                elif dist_genomic > 0:
                    x_cursor += dist_genomic * SCALE_FACTOR

            rel_start = x_cursor
            rel_end = rel_start + total_scaled_width

            gene_annotation = row["annotation"] if pd.notna(row["annotation"]) else None

            if highlight_gene is not None and gene_name == highlight_gene:
                facecolor = highlight_color
                edgecolor = highlight_edge
                lw = highlight_linewidth
            else:
                if color_by_gene_name:
                    facecolor = "#FFFFFF" if gene_name is None else gene_colors.get(gene_name, "#FFFFFF")
                else:
                    facecolor = "#FFFFFF" if gene_annotation is None else gene_colors.get(gene_annotation, "#FFFFFF")
                edgecolor = "black"
                lw = 1

            arrow = FancyArrow(rel_start if strand == "+" else rel_end,
                               arrow_y,
                               arrow_len if strand == "+" else -arrow_len,
                               0,
                               width=arrow_height * 0.2,
                               head_width=arrow_height * 0.3,
                               head_length=head_len * 0.2,
                               length_includes_head=True,
                               facecolor=facecolor,
                               edgecolor=edgecolor,
                               linewidth=lw)
            ax.add_patch(arrow)

            legend_key = gene_name if color_by_gene_name else gene_annotation
            if legend_key is not None:
                drawn_keys.add(legend_key)

            if gene_name:
                positions[species].setdefault(gene_name, []).append({
                    "xs": rel_start, "xe": rel_end, "y": arrow_y,
                    "gene_number": gene_number,
                    "identity": row["identity"],
                    "value_identity": row["value_identity"],
                })

            gene_positions_visual.append({"gene_number": gene_number, "rel_start": rel_start,
                                          "rel_end": rel_end, "gap_before": gap_before})
            prev_row = row
            x_cursor = rel_end

        ax.hlines(arrow_y, 15, x_cursor, colors='black', linewidth=3.2, zorder=0)
        x_cursor_max = max(x_cursor_max, x_cursor)

        # Double-slash marks: exactly the gaps that were shortened above
        for i in range(1, len(gene_positions_visual)):
            if gene_positions_visual[i]["gap_before"]:
                prev_end = gene_positions_visual[i - 1]["rel_end"]
                curr_start = gene_positions_visual[i]["rel_start"]
                center_x = (prev_end + curr_start) / 2
                offset = 75
                height_offset = 0.0375
                lw = 2.5
                ax.plot([center_x - offset, center_x], [arrow_y - height_offset, arrow_y + height_offset], color="black", lw=lw)
                ax.plot([center_x, center_x + offset], [arrow_y - height_offset, arrow_y + height_offset], color="black", lw=lw)

    # -----------------------------------------------------------------------
    # Links between homologous genes (same gene_name) of different species.
    # Paralogs: the n-th copy in a species is linked to the n-th copy in the
    # next species; links are never drawn within the same species.
    # -----------------------------------------------------------------------
    for i, sp in enumerate(species_list):
        for gene_name, copies in positions.get(sp, {}).items():
            candidates = species_list[i + 1:] if link_skip_missing else species_list[i + 1:i + 2]
            target = next((s for s in candidates if gene_name in positions.get(s, {})), None)
            if target is None:
                continue
            for c1, c2 in zip(copies, positions[target][gene_name]):
                x1s, x1e, y1 = c1["xs"], c1["xe"], c1["y"]
                x2s, x2e, y2 = c2["xs"], c2["xe"], c2["y"]

                alpha = c2["identity"]
                alpha = 0.5 if pd.isna(alpha) else float(min(max(alpha, 0.0), 1.0))
                value_id = c2["value_identity"]

                verts = [(x1s, y1), (x1e, y1), (x2e, y2), (x2s, y2)]
                ax.add_patch(Polygon(verts, closed=True, facecolor="gray", alpha=alpha, edgecolor=None, zorder=0))
                ax.plot([x1s, x2s], [y1, y2], color="gray", linestyle="--", linewidth=1.5, zorder=1)
                ax.plot([x1e, x2e], [y1, y2], color="gray", linestyle="--", linewidth=1.5, zorder=1)

                if not is_missing(value_id):
                    mid_x = (x1s + x1e + x2s + x2e) / 4
                    mid_y = (y1 + y2) / 2
                    try:
                        text_to_show = f"{float(value_id):.2f}"
                    except (ValueError, TypeError):
                        text_to_show = str(value_id)
                    ax.text(mid_x, mid_y, text_to_show, ha='center', va='center', fontsize=10,
                            color='black', fontweight='bold', zorder=2)

    # Species / strain labels
    for sp in species_list:
        y = species_to_y[sp]
        ax.text(-200, y + arrow_line_offset + 0.03, wrap(sp, 25)[0], ha='right', va='center',
                fontsize=14, fontstyle='italic')

        strain_vals = df[df["species"] == sp]["strain"].dropna().unique()
        strain = strain_vals[0] if len(strain_vals) > 0 else None
        if strain is not None and str(strain).strip() != "":
            ax.text(-200, y + arrow_line_offset - 0.055, str(strain), ha='right', va='center',
                    fontsize=12, fontstyle='normal', color='black')

    ax.set_xlim(-350, x_cursor_max + 800)
    ax.set_ylim(-arrow_height + arrow_line_offset,
                inter_species_spacing * n_species + arrow_height * 3 + arrow_line_offset)
    ax.axis('off')
    plt.tight_layout()

    # Scale bar
    sl = 1000 * SCALE_FACTOR
    sx = 10
    sy = min(species_to_y.values()) - 0.175
    ax.hlines(sy, sx, sx + sl, colors='black', linewidth=3)
    ax.vlines([sx, sx + sl], sy - 0.025, sy + 0.025, colors='black', linewidth=3)
    ax.text(sx + sl / 2, sy + 0.075, "1000 bp", ha='center', va='top', fontsize=14)

    # Legend
    leg = []
    legend_title = "Gene name" if color_by_gene_name else "Annotation"
    for label, c in sorted(gene_colors.items()):
        if label not in drawn_keys:
            continue  # in the color table but not in the drawn loci
        if color_by_gene_name:
            if is_numeric(label):
                continue
            formatted_label = format_gene_name(label)
        else:
            formatted_label = label
        leg.append(Patch(facecolor=c, edgecolor='black', label=formatted_label))
    if highlight_gene is not None:
        leg.append(Patch(facecolor=highlight_color, edgecolor=highlight_edge,
                         linewidth=highlight_linewidth, label=f"{highlight_gene} (highlighted)"))
    leg.append(Patch(facecolor="#FFFFFF", edgecolor='black', label="No categories"))
    ax.legend(handles=leg, title=legend_title, loc='lower center', bbox_to_anchor=(0.5, 0.065),
              fontsize=16, title_fontsize=18, frameon=True, ncol=min(len(leg), 5),
              columnspacing=1.5, handletextpad=0.5, borderpad=1.0)

    # Output: the format follows the file extension (.png, .svg, .pdf, ...).
    # SVG/PDF keep the text as editable text instead of converting it to paths.
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Map saved in {output_file}")


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Draw comparative gene maps of a locus across multiple species.")
    p.add_argument("--info", default="example_locus_info.csv", help="locus info table (CSV)")
    p.add_argument("--contig", default="example_locus_contig.csv", help="contig continuity table (CSV)")
    p.add_argument("--colors", default=None,
                   help=f"CSV with gene_name,color_code, required when coloring by gene_name "
                        f"(default: {DEFAULT_COLORS_FILE})")
    p.add_argument("-o", "--output", default="Comparative_genetic_map.png",
                   help="output file; format from extension: .png, .svg, .pdf")
    p.add_argument("-r", "--reference", default=None,
                   help="reference species shown on top (default: species with the lowest 'phy')")
    p.add_argument("--highlight", default=None, help="gene_name to highlight")
    p.add_argument("--color-by-gene-name", action="store_true",
                   help="color arrows by gene_name instead of annotation")
    p.add_argument("--spacing", type=float, default=0.2, help="vertical spacing between species")
    p.add_argument("--threshold", type=int, default=10000,
                   help="max intergenic distance (bp) drawn at real scale")
    p.add_argument("--no-flip-strand", action="store_true",
                   help="do NOT invert the strand of 'R' species (use if 'verse' is already inverted in the CSV)")
    p.add_argument("--link-skip-missing", action="store_true",
                   help="link a gene to the next species carrying it when it is absent in the adjacent one")
    p.add_argument("--font", default=None, help="preferred font family (fallbacks are applied automatically)")
    p.add_argument("--no-interactive", action="store_true",
                   help="never ask questions (for pipelines). By default the script asks whether to "
                        "highlight a gene and how to color the arrows, unless --highlight or "
                        "--color-by-gene-name are given or the input is not a terminal")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_font(args.font)

    df_info = load_csv(args.info)
    df_contig = load_csv(args.contig)

    highlight_gene = args.highlight
    color_by_gene_name = args.color_by_gene_name

    interactive = (not args.no_interactive and sys.stdin.isatty()
                   and args.highlight is None and not args.color_by_gene_name)

    if interactive:
        ask = input("Do you want to highlight a particular gene? (Y/n): ").strip().lower()
        if ask in ("y", ""):
            highlight_gene = input("Insert the exact gene_name reported in the info table: ").strip()
            if highlight_gene not in df_info["gene_name"].values:
                print(f"[!] Gene '{highlight_gene}' not found. No gene will be highlighted.")
                highlight_gene = None
            else:
                print(f"[OK] Gene '{highlight_gene}' found and highlighted in the image.")
        ask_color = input("Color the arrows by gene_name rather than by annotation? (y/N): ").strip().lower()
        color_by_gene_name = ask_color == "y"
    elif highlight_gene is not None and highlight_gene not in df_info["gene_name"].values:
        print(f"[!] Gene '{highlight_gene}' not found. No gene will be highlighted.")
        highlight_gene = None

    gene_colors = None
    if color_by_gene_name:
        colors_file = args.colors or DEFAULT_COLORS_FILE
        if not os.path.exists(colors_file):
            csvs = sorted(f for f in os.listdir(".") if f.lower().endswith(".csv"))
            sys.exit(f"[!] Color file '{colors_file}' not found in {os.getcwd()}.\n"
                     f"    CSV files in this folder: {csvs}\n"
                     f"    Use --colors <file> or change DEFAULT_COLORS_FILE at the top of the script.")
        gene_colors = load_gene_colors(colors_file)

    draw_all_species_loci(df_info, df_contig,
                          output_file=args.output,
                          arrow_line_offset=0.0,
                          inter_species_spacing=args.spacing,
                          species_reference=args.reference,
                          distance_threshold_bp=args.threshold,
                          highlight_gene=highlight_gene,
                          color_by_gene_name=color_by_gene_name,
                          gene_colors=gene_colors,
                          flip_strand_on_reverse=not args.no_flip_strand,
                          link_skip_missing=args.link_skip_missing)


if __name__ == "__main__":
    sys.exit(main())
