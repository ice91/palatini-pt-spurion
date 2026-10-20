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

SCRIPT_BY_FIGNAME = {
  "fig1_c1_pure_trace":     "scripts/fig_c1_pure_trace.py",
  "fig2_c1_alignment":      "scripts/fig_c1_alignment.py",
  "fig3_c2_coeff_compare":  "scripts/fig_c2_coeff_compare.py",
  "fig4_c3_cT_heatmap":     "scripts/fig_c3_cT_heatmap.py",
  "fig5_c3_dispersion":     "scripts/fig_c3_dispersion.py",
  "fig6_c3_degeneracy":     "scripts/fig_c3_degeneracy.py",
  "fig7_gw_waveform_overlay":"scripts/fig_gw_waveform_overlay.py",
  "fig8_nlo_offsets":       "scripts/fig_nlo_offsets.py",
  "fig9_flux_ratio":        "scripts/fig_flux_ratio.py",
}

rule all:
    input: PDFS

rule fig:
    output: "figs/pdf/{fig}.pdf"
    params:
        script=lambda wc: SCRIPT_BY_FIGNAME[wc.fig]
    shell:
        "python {params.script} --config configs/paper_grids.yaml"