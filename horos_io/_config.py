slice_types = ("cine_sa", "cine_2ch", "cine_3ch", "cine_4ch", "perfusion_b3s", "perfusion_b2s", "perfusion_b1s")

contour_types = (
    "sax_lv_endo",
    "sax_lv_epi",
    "sax_rv",
    "lax_2ch_lv_wall",
    "lax_2ch_lv_vol",
    "lax_2ch_la",
    "lax_3ch_lv_wall",
    "lax_3ch_lv_vol",
    "lax_3ch_rv",
    "lax_4ch_lv_wall",
    "lax_4ch_lv_vol",
    "lax_4ch_rv",
    "lax_4ch_la",
    "lax_4ch_ra",
    "omega_4ch",
    "omega_3ch",
    "omega_2ch",
    "omega_sa",
    "omega_perf_b1s",
    "omega_perf_b2s",
    "omega_perf_b3s",

)

omega_types = ("omega_sa", "omega_2ch", "omega_3ch", "omega_4ch", "omega_perf_b1s", "omega_perf_b2s", "omega_perf_b3s")

# TODO: omega_3ch_names, omega_2ch_names, omega_sax_names
omega_4ch_names = ("rv_vol", "lv_vol", "ra", "la", "lv_wall")  # now also denotes plotting order
omega_3ch_names = ("rv_vol", "lv_vol", "aroot", "la", "lv_wall")
omega_2ch_names = ("lv_vol", "la", "lv_wall")
omega_sa_names = ("rv_vol", "lv_wall", "lv_vol")
omega_perf_b1s_names = ("rv_vol", "lv_wall", "lv_vol")
omega_perf_b2s_names = ("rv_vol", "lv_wall", "lv_vol")
omega_perf_b3s_names = ("rv_vol", "lv_wall", "lv_vol")

time_format = "%Y/%m/%d, %H:%M:%S"

study_date_tag = ("0040", "0244")


task_id = "033"
task_name = f"Dataset{task_id}_Perfusion"
nn_UNet_raw_database = "nnUNet_raw_data_base"
