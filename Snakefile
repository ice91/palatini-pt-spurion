PDFS = [
  "figs/pdf/fig1_c1_pure_trace.pdf",
  "figs/pdf/fig2_c1_alignment.pdf",
  "figs/pdf/fig3_c2_coeff_compare.pdf",
  "figs/pdf/fig4_c3_cT_heatmap.pdf",
  "figs/pdf/fig5_c3_dispersion.pdf",
  "figs/pdf/fig6_c3_degeneracy.pdf",
  "figs/pdf/fig7_gw_waveform_overlay.pdf",
  "figs/pdf/fig8_nlo_offsets.pdf",
  "figs/pdf/fig9_flux_ratio.pdf",
]

rule all:
    input: PDFS

rule fig8:
    output: "figs/pdf/fig8_nlo_offsets.pdf"
    shell:  "python scripts/fig_nlo_offsets.py --config configs/paper_grids.yaml"

rule fig9:
    output: "figs/pdf/fig9_flux_ratio.pdf"
    shell:  "python scripts/fig_flux_ratio.py --config configs/paper_grids.yaml"
