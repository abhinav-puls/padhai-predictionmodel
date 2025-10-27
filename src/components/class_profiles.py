import os
import pandas as pd
import datetime
from typing import Tuple

# label maps (kept from original)
lang_labels = {
    0: "Lang_Beginner",
    1: "Lang_Letter",
    2: "Lang_Word",
    3: "Lang_Paragraph",
    4: "Lang_Story",
}

math_labels = {
    0: "Maths_Beginner",
    1: "Maths_NR1",
    2: "Maths_NR2",
    3: "Maths_Sub",
    4: "Maths_Div",
}

cluster_labels = {
    0: "Math Challenged",
    1: "Balance Mix",
    2: "Foundationally Challenged",
    3: "Language Challenged",
    4: "Progressed",
    9: "Data Not Available",
}


def compute_cluster(math_dist: dict, lang_dist: dict, class_size: int | None = None) -> int:
    """Compute cluster id from normalized distributions (values 0..1)."""
    MBL2 = math_dist.get("Maths_Beginner", 0) + math_dist.get("Maths_NR1", 0) + math_dist.get("Maths_NR2", 0)
    RBL2 = lang_dist.get("Lang_Beginner", 0) + lang_dist.get("Lang_Letter", 0) + lang_dist.get("Lang_Word", 0)

    if MBL2 < 0.2 and RBL2 < 0.2:
        return 4
    if 0.65 <= MBL2 <= 1 and 0.8 <= RBL2 <= 1:
        return 2
    if 0.7 <= MBL2 <= 1 and RBL2 < 0.8:
        return 0
    if 0 <= MBL2 <= 0.65 and 0.7 <= RBL2 <= 1:
        return 3
    if 0 <= MBL2 <= 0.7 and 0 <= RBL2 <= 0.7:
        return 1
    return 9


def _compute_phase_cluster_df(df_phase: pd.DataFrame, lang_col: str, math_col: str, community_col: str = "community_id") -> pd.DataFrame:
    """
    Given a dataframe filtered to a phase and the column names for language/math
    (these columns should contain numeric labels 0..4), compute cluster per community.
    Returns DataFrame with columns: community_id, cluster_id, cluster_label
    """
    rows = []
    for community, grp in df_phase.groupby(community_col):
        total = len(grp)
        if total == 0:
            rows.append((community, 9, cluster_labels.get(9)))
            continue

        # normalized distributions for labels 0..4
        lang_dist = (
            grp[lang_col]
            .value_counts(normalize=True)
            .reindex(range(5), fill_value=0)
            .rename(index=lang_labels)
            .to_dict()
        )
        math_dist = (
            grp[math_col]
            .value_counts(normalize=True)
            .reindex(range(5), fill_value=0)
            .rename(index=math_labels)
            .to_dict()
        )

        cluster_id = compute_cluster(math_dist, lang_dist, total)
        rows.append((community, cluster_id, cluster_labels.get(cluster_id, "Unknown")))

    return pd.DataFrame(rows, columns=[community_col, "cluster_id", "cluster_label"])


def compute_class_profiles(pred_df: pd.DataFrame, output_dir: str = os.path.join("artifact", "output")) -> Tuple[pd.DataFrame, str]:
    """
    Compute class/community profiles for all phases.

    Accepts either:
    - pred_df with per-phase columns: el1_prediction_lang, el1_prediction_maths, el2_..., el3_...
    OR
    - pred_df with 'Phase' column and el_prediction_lang / el_prediction_maths per-row.

    Returns (profiles_df, output_path). profiles_df contains community_id and cluster info for each phase.
    """
    if pred_df is None or pred_df.empty:
        raise ValueError("pred_df is empty or None")

    # Determine available format
    has_per_phase = all(col in pred_df.columns for col in ("el1_prediction_lang", "el1_prediction_maths"))
    results = []

    if has_per_phase:
        # build Phase1/2/3 dfs
        Ph0 = pred_df[["community_id", "student_id", "bl_language", "bl_mathematics"]].copy()
        Ph1 = pred_df[["community_id", "student_id", "el1_prediction_lang", "el1_prediction_maths"]].copy()
        Ph2 = pred_df[["community_id", "student_id", "el2_prediction_lang", "el2_prediction_maths"]].copy()
        Ph3 = pred_df[["community_id", "student_id", "el3_prediction_lang", "el3_prediction_maths"]].copy()

        Ph0_cluster = _compute_phase_cluster_df(Ph0, "bl_language", "bl_mathematics")        
        Ph1_cluster = _compute_phase_cluster_df(Ph1, "el1_prediction_lang", "el1_prediction_maths")
        Ph2_cluster = _compute_phase_cluster_df(Ph2, "el2_prediction_lang", "el2_prediction_maths")
        Ph3_cluster = _compute_phase_cluster_df(Ph3, "el3_prediction_lang", "el3_prediction_maths")
    else:
        # expect Phase column and el_prediction_* columns
        if "Phase" not in pred_df.columns or "el_prediction_lang" not in pred_df.columns or "el_prediction_maths" not in pred_df.columns:
            raise ValueError("pred_df does not contain expected prediction columns for either per-phase or Phase format")

        Ph0_df = pred_df[pred_df["Phase"] == 1][["community_id", "student_id", "bl_language", "bl_mathematics"]].copy()
        Ph1_df = pred_df[pred_df["Phase"] == 1][["community_id", "student_id", "el_prediction_lang", "el_prediction_maths"]].copy()
        Ph2_df = pred_df[pred_df["Phase"] == 2][["community_id", "student_id", "el_prediction_lang", "el_prediction_maths"]].copy()
        Ph3_df = pred_df[pred_df["Phase"] == 3][["community_id", "student_id", "el_prediction_lang", "el_prediction_maths"]].copy()

        # rename to per-phase names for cluster function reuse
        Ph1_df = Ph1_df.rename(columns={"el_prediction_lang": "el1_prediction_lang", "el_prediction_maths": "el1_prediction_maths"})
        Ph2_df = Ph2_df.rename(columns={"el_prediction_lang": "el2_prediction_lang", "el_prediction_maths": "el2_prediction_maths"})
        Ph3_df = Ph3_df.rename(columns={"el_prediction_lang": "el3_prediction_lang", "el_prediction_maths": "el3_prediction_maths"})

        Ph0_cluster = _compute_phase_cluster_df(Ph0_df, "bl_language", "bl_mathematics")
        Ph1_cluster = _compute_phase_cluster_df(Ph1_df, "el1_prediction_lang", "el1_prediction_maths")
        Ph2_cluster = _compute_phase_cluster_df(Ph2_df, "el2_prediction_lang", "el2_prediction_maths")
        Ph3_cluster = _compute_phase_cluster_df(Ph3_df, "el3_prediction_lang", "el3_prediction_maths")

        # merge clusters by community_id
    merged = (
            Ph0_cluster.rename(columns={"cluster_id": "cluster_bl", "cluster_label": "cluster_bl_label"})
            .merge(Ph1_cluster.rename(columns={"cluster_id": "cluster_el1", "cluster_label": "cluster_el1_label"}), on="community_id", how="outer")
            .merge(Ph2_cluster.rename(columns={"cluster_id": "cluster_el2", "cluster_label": "cluster_el2_label"}), on="community_id", how="outer")
            .merge(Ph3_cluster.rename(columns={"cluster_id": "cluster_el3", "cluster_label": "cluster_el3_label"}), on="community_id", how="outer"))


    # persist
    os.makedirs(output_dir, exist_ok=True)
    filename = f"class_profiles.csv"
    output_path = os.path.join(output_dir, filename)
    merged.to_csv(output_path, index=False)

    return merged, output_path
