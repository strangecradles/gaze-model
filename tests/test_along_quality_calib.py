import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import along_quality_calib as aqc  # noqa: E402


def test_cache_tag_for_config_keeps_canonical_cache_separate():
    cfg = aqc.CandidateConfig(2.0, 6.0, 1.0)
    assert aqc.cache_tag_for_config(cfg) == "m4_dpf_physics_aq_qv_power_s2_6_g1"
    assert aqc.cache_tag_for_config(cfg, 20.0) == "m4_dpf_physics_aq_qv_power_s2_6_g1_d20"
    assert aqc.cache_tag_for_config(
        cfg, 20.0, slew_gate=True, slew_max_deg_s=100.0
    ) == "m4_dpf_physics_aq_qv_power_s2_6_g1_sg100_d20"
    assert aqc.cache_tag_for_config(
        cfg, 20.0, slew_gate=True, slew_max_deg_s=100.0, velocity_cost=2.0
    ) == "m4_dpf_physics_aq_qv_power_s2_6_g1_sg100_vc2_d20"
    assert aqc.cache_tag_for_config(
        cfg, 20.0, slew_gate=True, slew_max_deg_s=100.0, acceleration_cost=3.0
    ) == "m4_dpf_physics_aq_qv_power_s2_6_g1_sg100_ac3_d20"
    assert aqc.cache_tag_for_config(
        cfg, 20.0, slew_gate=True, slew_max_deg_s=100.0,
        motion_prior=True, motion_prior_sigma_rows=2.0, motion_prior_tau_s=0.003,
    ) == "m4_dpf_physics_aq_qv_power_s2_6_g1_sg100_mp_s2_tau3_d20"
    assert aqc.variant_id_for_config(
        cfg, slew_gate=True, slew_max_deg_s=75.0
    ) == "qv_power_s2_6_g1_sg75"
    assert aqc.variant_id_for_config(
        cfg, slew_gate=True, slew_max_deg_s=75.0, velocity_cost=2.0
    ) == "qv_power_s2_6_g1_sg75_vc2"
    assert aqc.variant_id_for_config(
        cfg, slew_gate=True, slew_max_deg_s=75.0, acceleration_cost=3.0
    ) == "qv_power_s2_6_g1_sg75_ac3"
    assert aqc.variant_id_for_config(
        cfg, slew_gate=True, slew_max_deg_s=75.0,
        motion_prior=True, motion_prior_sigma_rows=2.0, motion_prior_tau_s=0.003,
    ) == "qv_power_s2_6_g1_sg75_mp_s2_tau3"
    assert aqc.cache_tag_for_config(cfg) != "m4_dpf_physics"
    const = aqc.config_lookup()["constant"]
    assert aqc.cache_tag_for_config(
        const, 20.0, slew_gate=True, slew_max_deg_s=100.0
    ) == "m4_dpf_physics_aq_constant_sg100_d20"


def test_calibration_script_dry_run_writes_csv_and_md(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration"
    rc = aqc.main([
        "--dry-run",
        "--max-configs", "2",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    assert csv_path.exists()
    assert md_path.exists()
    csv_text = csv_path.read_text()
    md_text = md_path.read_text()
    assert "raw_jump_reduction" in csv_text
    assert "Config Summary" in md_text
    assert "Leave-One-Subject-Out" in md_text


def test_calibration_dry_run_keeps_slew_variants_separate(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_slew"
    rc = aqc.main([
        "--dry-run",
        "--max-configs", "1",
        "--slew-max-grid", "75,100",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    assert "qv_power_s2_6_g0p5_sg75" in text
    assert "qv_power_s2_6_g0p5_sg100" in text
    md = out_prefix.with_suffix(".md").read_text()
    assert "variant_id" in md


def test_calibration_dry_run_supports_constant_candidate(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_constant"
    rc = aqc.main([
        "--dry-run",
        "--configs", "constant",
        "--slew-max-grid", "100",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    assert "constant_sg100" in text
    assert "m4_dpf_physics_aq_constant_sg100" in text


def test_calibration_dry_run_supports_velocity_cost_variant(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_vc"
    rc = aqc.main([
        "--dry-run",
        "--configs", "constant",
        "--slew-max-grid", "100",
        "--hypothesis-velocity-cost", "2",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    assert "constant_sg100_vc2" in text
    assert "hypothesis_velocity_cost" in text


def test_calibration_dry_run_supports_acceleration_cost_variant(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_ac"
    rc = aqc.main([
        "--dry-run",
        "--configs", "constant",
        "--slew-max-grid", "100",
        "--hypothesis-acceleration-cost", "3",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    md = out_prefix.with_suffix(".md").read_text()
    assert "constant_sg100_ac3" in text
    assert "m4_dpf_physics_aq_constant_sg100_ac3" in text
    assert "hypothesis_acceleration_cost" in md


def test_calibration_dry_run_supports_motion_prior_variant(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_mp"
    rc = aqc.main([
        "--dry-run",
        "--configs", "constant",
        "--slew-max-grid", "100",
        "--motion-prior-grid", "2:3",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    md = out_prefix.with_suffix(".md").read_text()
    assert "constant_sg100_mp_s2_tau3" in text
    assert "m4_dpf_physics_aq_constant_sg100_mp_s2_tau3" in text
    assert "motion_prior_sigma_rows" in md


def test_calibration_dry_run_supports_resolver_knob_variants(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_resolver_knobs"
    rc = aqc.main([
        "--dry-run",
        "--configs", "constant",
        "--lag-ms-grid", "1,2",
        "--hypothesis-transition-sigma-grid", "2,5",
        "--hypothesis-obs-weight-grid", "3,8",
        "--slew-max-grid", "125",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    md = out_prefix.with_suffix(".md").read_text()
    assert "constant_ts2_ow3_sg125" in text
    assert "constant_lag2_ts5_ow8_sg125" in text
    assert "m4_dpf_physics_aq_constant_lag2_ts5_ow8_sg125" in text
    assert "hypothesis_obs_weight" in md


def test_calibration_dry_run_supports_blend_variant(tmp_path):
    out_prefix = tmp_path / "along_quality_calibration_blend"
    rc = aqc.main([
        "--dry-run",
        "--configs", "constant",
        "--slew-max-grid", "100",
        "--hypothesis-blend-immediate",
        "--hypothesis-blend-delta-rows", "12",
        "--hypothesis-blend-alpha", "0.5",
        "--out-prefix", str(out_prefix),
    ])
    assert rc == 0
    text = out_prefix.with_suffix(".csv").read_text()
    assert "constant_bi_d12_a0p5_sg100" in text
    assert "m4_dpf_physics_aq_constant_bi_d12_a0p5_sg100" in text


def test_policy_writer_and_policy_dry_run(tmp_path):
    source_prefix = tmp_path / "along_quality_policy_source"
    rc = aqc.main([
        "--dry-run",
        "--subjects", "Ashton3,Chong",
        "--configs", "constant",
        "--slew-max-grid", "100",
        "--motion-prior-grid", "2:3,4:6",
        "--out-prefix", str(source_prefix),
    ])
    assert rc == 0

    policy_prefix = tmp_path / "along_quality_policy"
    rc = aqc.main([
        "--write-policy-from", str(source_prefix.with_suffix(".csv")),
        "--policy-out-prefix", str(policy_prefix),
    ])
    assert rc == 0
    policy_json = policy_prefix.with_suffix(".json")
    policy_md = policy_prefix.with_suffix(".md")
    assert policy_json.exists()
    assert policy_md.exists()
    variants = aqc.load_subject_policy(str(policy_json))
    assert set(variants) == {"Ashton3", "Chong"}
    assert all(v.motion_prior for v in variants.values())

    run_prefix = tmp_path / "along_quality_policy_run"
    rc = aqc.main([
        "--dry-run",
        "--subjects", "Ashton3,Chong",
        "--policy-from", str(policy_json),
        "--out-prefix", str(run_prefix),
    ])
    assert rc == 0
    text = run_prefix.with_suffix(".csv").read_text()
    assert "mp_s" in text
    assert "motion_prior" in text
