"""
Local Growth Estimator (PPC + Organic + Maps)
==============================================
A professional estimation tool for local SEO, Maps, and PPC.
Outputs ranges, not guarantees. Designed for agencies and consultants.

Key changes in this version:
- Fixed st.stop() breaking cross-tab rendering (now uses conditional rendering)
- Added cross-field validation (low > high checks)
- Moved GSC upload to top-level for visibility
- Added channel overlap disclaimer in Summary
- Improved PPC posture naming (Conservative/Aggressive bidding)
- Added assumptions export functionality
- Collapsible advanced inputs to reduce overwhelm
- Constants for magic strings
- Multi-format file support (CSV, XLSX, TSV with auto-encoding detection)
- Clean CSV exports with descriptive filenames
"""

import math
import re
import json
import io
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import pandas as pd
import streamlit as st


# =========================================================
# CONSTANTS - No more magic strings scattered everywhere
# =========================================================
class Scenario:
    """Scenario names used throughout the app"""
    CONSERVATIVE = "Conservative"
    EXPECTED = "Expected"
    AGGRESSIVE = "Aggressive"


class Confidence:
    """Confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataMode:
    """Data source modes for organic estimation"""
    KW_ONLY = "KW"
    GSC_ONLY = "GSC"
    BOTH = "BOTH"
    NONE = "NONE"


class GSCDateRange:
    """GSC export date range options and their divisors for monthly conversion"""
    LAST_7_DAYS = "Last 7 days"
    LAST_28_DAYS = "Last 28 days"
    LAST_3_MONTHS = "Last 3 months"
    LAST_6_MONTHS = "Last 6 months"
    LAST_12_MONTHS = "Last 12 months"
    LAST_16_MONTHS = "Last 16 months"
    
    # Divisors to convert period totals to monthly averages
    DIVISORS = {
        LAST_7_DAYS: 7 / 30,      # ~0.23 months
        LAST_28_DAYS: 28 / 30,    # ~0.93 months  
        LAST_3_MONTHS: 3,
        LAST_6_MONTHS: 6,
        LAST_12_MONTHS: 12,
        LAST_16_MONTHS: 16,
    }
    
    @classmethod
    def get_divisor(cls, range_selection: str) -> float:
        """Get the divisor to convert totals to monthly averages"""
        return cls.DIVISORS.get(range_selection, 1.0)
    
    @classmethod
    def get_options(cls) -> List[str]:
        """Get list of date range options"""
        return [
            cls.LAST_7_DAYS,
            cls.LAST_28_DAYS,
            cls.LAST_3_MONTHS,
            cls.LAST_6_MONTHS,
            cls.LAST_12_MONTHS,
            cls.LAST_16_MONTHS,
        ]


# =========================================================
# Formatting + parsing helpers
# =========================================================
def money(x: float) -> str:
    """Format a number as currency"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if math.isinf(x):
        return "∞"
    return f"${x:,.0f}"


def num(x: float, decimals: int = 0) -> str:
    """Format a number with specified decimal places"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if math.isinf(x):
        return "∞"
    return f"{x:,.{decimals}f}"


def pct_str(dec: float, decimals: int = 0) -> str:
    """Convert decimal to percentage string (0.35 -> '35%')"""
    if dec is None or (isinstance(dec, float) and math.isnan(dec)):
        return ""
    if math.isinf(dec):
        return "∞"
    return f"{dec * 100:.{decimals}f}%"


def parse_spend_tiers(text: str) -> List[float]:
    """Parse comma/newline/space separated spend tiers into sorted list"""
    parts = [p.strip() for p in text.replace("\n", ",").replace(" ", ",").split(",")]
    tiers = []
    for p in parts:
        if not p:
            continue
        try:
            tiers.append(float(p))
        except ValueError:
            continue  # Skip invalid entries
    return sorted(set(tiers))


def safe_div(a: float, b: float) -> float:
    """Safe division that returns infinity instead of raising error"""
    return math.inf if b == 0 else (a / b)


def pct_to_dec(pct_value: float) -> float:
    """Convert human percentage (35) to decimal (0.35)"""
    return pct_value / 100.0


def df_to_clean_csv(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to clean CSV string without BOM or index.
    Uses regular hyphens instead of en-dashes for compatibility.
    """
    # Replace en-dashes with regular hyphens for compatibility
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].str.replace('–', '-', regex=False)
    
    return df_clean.to_csv(index=False)


def download_table_button(
    df: pd.DataFrame,
    filename: str,
    button_label: str = "📥 Download CSV",
    key: str = None
):
    """
    Render a download button for a DataFrame with clean CSV export.
    """
    csv_data = df_to_clean_csv(df)
    st.download_button(
        label=button_label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=key
    )


# =========================================================
# Guardrailed number inputs (no sliders)
# =========================================================
def percent_input(
    label: str,
    default_pct: float,
    recommended_min_pct: float,
    recommended_max_pct: float,
    help_text: str,
    explanation_md: str,
    key: str,
    hard_min_pct: float = 0.0,
    hard_max_pct: float = 100.0,
    step: float = 1.0,
) -> float:
    """
    User types a percent like 35 (meaning 35%).
    Returns decimal like 0.35.
    Warns if outside recommended range. Does not block.
    """
    st.caption(f"Recommended range: **{recommended_min_pct:.0f}%–{recommended_max_pct:.0f}%**")
    val_pct = st.number_input(
        label,
        min_value=float(hard_min_pct),
        max_value=float(hard_max_pct),
        value=float(default_pct),
        step=float(step),
        key=key,
        help=help_text,
    )
    if val_pct < recommended_min_pct or val_pct > recommended_max_pct:
        st.warning(
            f"⚠️ {label}: **{val_pct:.0f}%** is outside the typical range "
            f"(**{recommended_min_pct:.0f}%–{recommended_max_pct:.0f}%**). "
            "That doesn't mean it's wrong — but it will materially change estimates."
        )
    with st.expander(f"ℹ️ What this means: {label}"):
        st.markdown(explanation_md)
    return pct_to_dec(val_pct)


def money_input(
    label: str,
    default_value: float,
    recommended_min: float,
    recommended_max: float,
    help_text: str,
    explanation_md: str,
    key: str,
    hard_min: float = 0.01,
    hard_max: float = 1000.0,
    step: float = 0.5,
) -> float:
    """
    Money input with recommended range and warnings.
    Returns the raw float value.
    """
    st.caption(f"Typical range: **{recommended_min:.2f}–{recommended_max:.2f}**")
    val = st.number_input(
        label,
        min_value=float(hard_min),
        max_value=float(hard_max),
        value=float(default_value),
        step=float(step),
        key=key,
        help=help_text,
    )
    if val < recommended_min or val > recommended_max:
        st.warning(
            f"⚠️ {label}: **{val:.2f}** is outside the typical range "
            f"(**{recommended_min:.2f}–{recommended_max:.2f}**). "
            "That's allowed — but be sure it matches real auction conditions."
        )
    with st.expander(f"ℹ️ What this means: {label}"):
        st.markdown(explanation_md)
    return float(val)


def confidence_badge(level: str) -> str:
    """Return a formatted confidence badge string"""
    level = (level or "").strip().lower()
    if level == Confidence.HIGH:
        return "🟢 **Confidence: High** — Multiple data sources cross-validated"
    if level == Confidence.MEDIUM:
        return "🟡 **Confidence: Medium** — Single reliable data source"
    return "🔴 **Confidence: Low** — Market-based estimate only (no site-specific data)"


# =========================================================
# Cross-field validation
# FIX: Validates that low/high pairs are correctly ordered
# =========================================================
def validate_range_pair(
    low_val: float,
    high_val: float,
    low_label: str,
    high_label: str,
    context: str = ""
) -> bool:
    """
    Validates that low <= high for a range pair.
    Returns True if valid, False if invalid.
    Shows error message if invalid.
    """
    if low_val > high_val:
        ctx = f" ({context})" if context else ""
        st.error(
            f"⛔ **Invalid range{ctx}:** {low_label} ({low_val:.2f}) cannot exceed "
            f"{high_label} ({high_val:.2f}). Please correct this before proceeding."
        )
        return False
    return True


# =========================================================
# Core estimation math
# =========================================================
def compute_click_caps(
    total_searches: float,
    business_hours_factor: float,
    impr_share_low: float,
    impr_share_high: float,
    ctr_low: float,
    ctr_high: float
) -> Tuple[float, float]:
    """
    PPC click cap = searches * biz_hours_factor * impression_share * CTR
    This represents the MAXIMUM clicks possible given market demand.
    """
    impressions_low = total_searches * business_hours_factor * impr_share_low
    impressions_high = total_searches * business_hours_factor * impr_share_high
    clicks_low = max(0.0, impressions_low * ctr_low)
    clicks_high = max(clicks_low, impressions_high * ctr_high)
    return clicks_low, clicks_high


def compute_ranges_with_caps(
    spend: float,
    cpc_low: float,
    cpc_high: float,
    clicks_cap_low: float,
    clicks_cap_high: float,
    cvr: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute clicks, leads, and CPL ranges given spend and caps.
    
    Returns:
      clicks (low–high), leads (low–high), CPL (best–worst)
    """
    # Raw clicks from spend ÷ CPC
    raw_clicks_high = spend / max(cpc_low, 0.01)   # best case (low CPC = more clicks)
    raw_clicks_low = spend / max(cpc_high, 0.01)   # worst case (high CPC = fewer clicks)

    # Apply market caps - can't get more clicks than market allows
    clicks_high = min(raw_clicks_high, clicks_cap_high)
    clicks_low = min(raw_clicks_low, clicks_cap_low)
    clicks_low = min(clicks_low, clicks_high)  # Ensure low <= high

    # Convert to leads
    leads_low = clicks_low * cvr
    leads_high = clicks_high * cvr

    # CPL calculation (best CPL = lowest cost per lead)
    cpl_best = safe_div(spend, leads_high) if leads_high > 0 else math.inf
    cpl_worst = safe_div(spend, leads_low) if leads_low > 0 else math.inf

    return (clicks_low, clicks_high), (leads_low, leads_high), (cpl_best, cpl_worst)


def organic_clicks_from_keyword_demand(
    total_searches: float,
    business_hours_factor: float,
    ctr_low: float,
    ctr_high: float,
    serp_leakage_factor: float,
) -> Tuple[float, float]:
    """
    Organic clicks (Keyword Planner mode) =
      searches * biz_hours_factor * CTR * leakage_factor
    """
    addressable = total_searches * business_hours_factor
    clicks_low = max(0.0, addressable * ctr_low * serp_leakage_factor)
    clicks_high = max(clicks_low, addressable * ctr_high * serp_leakage_factor)
    return clicks_low, clicks_high


def organic_clicks_from_gsc(
    gsc_impressions: float,
    target_ctr_low: float,
    target_ctr_high: float,
) -> Tuple[float, float]:
    """
    Organic clicks (GSC mode) =
      GSC impressions * target CTR
    Note: SERP leakage is already "baked into" GSC; don't apply leakage again.
    """
    clicks_low = max(0.0, gsc_impressions * target_ctr_low)
    clicks_high = max(clicks_low, gsc_impressions * target_ctr_high)
    return clicks_low, clicks_high


def maps_actions_from_demand(
    total_searches: float,
    business_hours_factor: float,
    action_share_low: float,
    action_share_high: float,
) -> Tuple[float, float]:
    """
    Maps actions = searches * biz_hours_factor * action_share
    """
    addressable = total_searches * business_hours_factor
    actions_low = max(0.0, addressable * action_share_low)
    actions_high = max(actions_low, addressable * action_share_high)
    return actions_low, actions_high


def apply_ramp(full_low: float, full_high: float, ramp: float) -> Tuple[float, float]:
    """Apply a ramp percentage to a range"""
    return (full_low * ramp, full_high * ramp)


# =========================================================
# Reusable DataFrame builders
# FIX: Reduces repetitive DataFrame construction code
# =========================================================
def build_scenario_table(
    scenarios: List[Tuple[str, Tuple[float, float]]],
    cvr: float,
    value_label: str = "Clicks/mo",
    lead_label: str = "Leads/mo"
) -> pd.DataFrame:
    """
    Build a standardized scenario comparison table.
    
    Args:
        scenarios: List of (scenario_name, (low, high)) tuples
        cvr: Conversion rate for lead calculation
        value_label: Label for the primary metric column
        lead_label: Label for the leads column
    """
    rows = []
    for name, (low, high) in scenarios:
        rows.append({
            "Scenario": name,
            f"{value_label} (low–high)": f"{num(low, 0)}–{num(high, 0)}",
            f"{lead_label} (low–high)": f"{num(low * cvr, 1)}–{num(high * cvr, 1)}",
        })
    return pd.DataFrame(rows)


def build_ramp_table(
    scenarios: List[Tuple[str, Tuple[float, float]]],
    cvr: float,
    ramps: List[Tuple[str, float]],
) -> pd.DataFrame:
    """
    Build a time-ramp table showing leads over time.
    
    Args:
        scenarios: List of (scenario_name, (low, high)) tuples for clicks/actions
        cvr: Conversion rate
        ramps: List of (month_label, ramp_decimal) tuples
    """
    rows = []
    for name, (low, high) in scenarios:
        row = {"Scenario": name}
        for month_label, ramp_pct in ramps:
            ramped = apply_ramp(low * cvr, high * cvr, ramp_pct)
            row[f"{month_label} leads"] = f"{num(ramped[0], 1)}–{num(ramped[1], 1)}"
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================
# Keyword Planner ingestion
# =========================================================
MONTH_RE = re.compile(r"^(?:Searches:\s*)?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", re.IGNORECASE)


def load_keyword_planner_file(uploaded_file) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Load and parse a Google Keyword Planner export.
    
    Supports:
    - XLSX files (with or without title rows)
    - CSV files (UTF-8, UTF-16, with comma or tab separators)
    
    Returns:
        - DataFrame with keyword data
        - Total average monthly searches
        - List of detected month columns (for seasonality)
    """
    filename = uploaded_file.name.lower()
    
    # Determine file type and load accordingly
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = _load_keyword_planner_excel(uploaded_file)
    elif filename.endswith('.csv') or filename.endswith('.tsv'):
        df = _load_keyword_planner_csv(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Please upload .xlsx, .xls, .csv, or .tsv")
    
    # Validate required columns
    if "Keyword" not in df.columns:
        raise ValueError("Missing 'Keyword' column. Is this a Keyword Planner export?")
    if "Avg. monthly searches" not in df.columns:
        raise ValueError("Missing 'Avg. monthly searches' column. Is this a Keyword Planner export?")
    
    # Convert search volume to numeric
    df["Avg. monthly searches"] = pd.to_numeric(df["Avg. monthly searches"], errors="coerce")
    
    # Detect month columns for seasonality
    month_cols = [
        c for c in df.columns
        if isinstance(c, str) and MONTH_RE.match(c.strip())
    ]
    
    for c in month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    total_avg = int(df["Avg. monthly searches"].fillna(0).sum())
    
    def month_sort_key(col: str) -> Tuple[int, int]:
        col = col.strip().replace("Searches:", "").strip()
        m, y = col.split()
        m_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        return (int(y), m_map[m.lower()])
    
    month_cols = sorted(month_cols, key=month_sort_key) if month_cols else []
    return df, total_avg, month_cols


def _load_keyword_planner_excel(uploaded_file) -> pd.DataFrame:
    """Load Keyword Planner from Excel file."""
    xl = pd.ExcelFile(uploaded_file)
    sheet = xl.sheet_names[0]
    raw = xl.parse(sheet)
    
    # Check if headers are already in the column names
    col_names = [str(c) for c in raw.columns]
    if "Keyword" in col_names and "Avg. monthly searches" in col_names:
        return raw.copy()
    
    # Look for header row in the data
    header_row_idx = None
    for i in range(min(20, len(raw))):
        row = raw.iloc[i].astype(str).tolist()
        if "Keyword" in row and "Avg. monthly searches" in row:
            header_row_idx = i
            break
    
    if header_row_idx is None:
        raise ValueError("Could not find header row with 'Keyword' and 'Avg. monthly searches'.")
    
    headers = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = headers
    df = df.reset_index(drop=True)
    return df


def _load_keyword_planner_csv(uploaded_file) -> pd.DataFrame:
    """Load Keyword Planner from CSV file, handling various encodings and formats."""
    
    # Read file content to detect encoding and format
    content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for re-reading
    
    # Try different encodings
    encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'latin-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            decoded = content.decode(encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError("Could not decode file. Please try saving as UTF-8 CSV.")
    
    # Detect separator (tab vs comma)
    first_lines = decoded.split('\n')[:5]
    tab_count = sum(line.count('\t') for line in first_lines)
    comma_count = sum(line.count(',') for line in first_lines)
    separator = '\t' if tab_count > comma_count else ','
    
    # Find the header row (may have title rows before it)
    header_row = None
    for i, line in enumerate(first_lines):
        if 'Keyword' in line and 'Avg. monthly searches' in line:
            header_row = i
            break
    
    if header_row is None:
        raise ValueError("Could not find header row with 'Keyword' and 'Avg. monthly searches'.")
    
    # Read with detected settings
    uploaded_file.seek(0)
    df = pd.read_csv(
        uploaded_file,
        encoding=encoding,
        sep=separator,
        skiprows=header_row
    )
    
    return df


def sum_month_searches(df: pd.DataFrame, month_col: str) -> float:
    """Sum searches for a specific month column"""
    return float(df[month_col].fillna(0).sum())


# =========================================================
# GSC ingestion (Queries report CSV)
# =========================================================
def load_gsc_queries_csv(uploaded_csv) -> pd.DataFrame:
    """
    Accepts a Google Search Console "Queries" export.
    Tries to detect columns flexibly.
    Output columns: query, clicks, impressions, position, ctr
    """
    df = pd.read_csv(uploaded_csv)

    # Normalize columns
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Query column candidates
    query_candidates = [c for c in df.columns if "query" in c.lower()]
    if not query_candidates:
        # Sometimes export uses "Top queries"
        query_candidates = [c for c in df.columns if "top" in c.lower() and "quer" in c.lower()]

    if not query_candidates:
        raise ValueError("Could not find a Query column in the GSC CSV. Expected something like 'Query' or 'Top queries'.")

    query_col = query_candidates[0]

    # Clicks, impressions, position
    def find_col_contains(substr: str) -> Optional[str]:
        for c in df.columns:
            if substr in c.lower():
                return c
        return None

    clicks_col = find_col_contains("click")
    impr_col = find_col_contains("impression")
    pos_col = find_col_contains("position")

    if clicks_col is None or impr_col is None:
        raise ValueError("Could not find Clicks/Impressions columns in the GSC CSV.")

    out = pd.DataFrame({
        "query": df[query_col].astype(str),
        "clicks": pd.to_numeric(df[clicks_col], errors="coerce").fillna(0),
        "impressions": pd.to_numeric(df[impr_col], errors="coerce").fillna(0),
    })

    if pos_col is not None:
        out["position"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        out["position"] = math.nan

    # Compute CTR from clicks/impressions
    out["ctr"] = out.apply(
        lambda r: (r["clicks"] / r["impressions"]) if r["impressions"] > 0 else 0.0,
        axis=1
    )

    # Filter rows with zero impressions
    out = out[out["impressions"] > 0].copy()

    return out


# =========================================================
# Assumptions export
# FIX: Allows exporting assumptions for proposals
# =========================================================
def build_assumptions_dict(
    cvr: float,
    business_hours_factor: float,
    # PPC
    con_cpc_low: float = None, con_cpc_high: float = None,
    con_is_low: float = None, con_is_high: float = None,
    con_ctr_low: float = None, con_ctr_high: float = None,
    agg_cpc_low: float = None, agg_cpc_high: float = None,
    agg_is_low: float = None, agg_is_high: float = None,
    agg_ctr_low: float = None, agg_ctr_high: float = None,
    # Organic
    serp_leakage: float = None,
    org_con_low: float = None, org_con_high: float = None,
    org_exp_low: float = None, org_exp_high: float = None,
    org_agg_low: float = None, org_agg_high: float = None,
    net_new_factor: float = None,
    org_ramps: Dict[str, float] = None,
    # Maps
    maps_con_low: float = None, maps_con_high: float = None,
    maps_exp_low: float = None, maps_exp_high: float = None,
    maps_agg_low: float = None, maps_agg_high: float = None,
    maps_qual: float = None,
    maps_ramps: Dict[str, float] = None,
) -> Dict:
    """Build a dictionary of all assumptions for export"""
    return {
        "generated_at": datetime.now().isoformat(),
        "shared": {
            "cvr_pct": pct_str(cvr, 1),
            "business_hours_factor_pct": pct_str(business_hours_factor, 0),
        },
        "ppc": {
            "conservative_bidding": {
                "cpc_range": f"${con_cpc_low:.2f}–${con_cpc_high:.2f}" if con_cpc_low else None,
                "impression_share_range": f"{pct_str(con_is_low, 0)}–{pct_str(con_is_high, 0)}" if con_is_low else None,
                "ctr_range": f"{pct_str(con_ctr_low, 0)}–{pct_str(con_ctr_high, 0)}" if con_ctr_low else None,
            },
            "aggressive_bidding": {
                "cpc_range": f"${agg_cpc_low:.2f}–${agg_cpc_high:.2f}" if agg_cpc_low else None,
                "impression_share_range": f"{pct_str(agg_is_low, 0)}–{pct_str(agg_is_high, 0)}" if agg_is_low else None,
                "ctr_range": f"{pct_str(agg_ctr_low, 0)}–{pct_str(agg_ctr_high, 0)}" if agg_ctr_low else None,
            },
        },
        "organic": {
            "serp_leakage_factor_pct": pct_str(serp_leakage, 0) if serp_leakage else None,
            "net_new_coverage_factor_pct": pct_str(net_new_factor, 0) if net_new_factor else None,
            "ctr_scenarios": {
                "conservative": f"{pct_str(org_con_low, 0)}–{pct_str(org_con_high, 0)}" if org_con_low else None,
                "expected": f"{pct_str(org_exp_low, 0)}–{pct_str(org_exp_high, 0)}" if org_exp_low else None,
                "aggressive": f"{pct_str(org_agg_low, 0)}–{pct_str(org_agg_high, 0)}" if org_agg_low else None,
            },
            "ramps": org_ramps,
        },
        "maps": {
            "action_qualification_rate_pct": pct_str(maps_qual, 0) if maps_qual else None,
            "action_share_scenarios": {
                "conservative": f"{pct_str(maps_con_low, 0)}–{pct_str(maps_con_high, 0)}" if maps_con_low else None,
                "expected": f"{pct_str(maps_exp_low, 0)}–{pct_str(maps_exp_high, 0)}" if maps_exp_low else None,
                "aggressive": f"{pct_str(maps_agg_low, 0)}–{pct_str(maps_agg_high, 0)}" if maps_agg_low else None,
            },
            "ramps": maps_ramps,
        },
    }


# =========================================================
# App UI
# =========================================================
# Only set page config if running standalone (not through app.py wrapper)
# This check prevents the "set_page_config can only be called once" error
try:
    st.set_page_config(page_title="Local Growth Estimator", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # Page config already set by wrapper

st.title("Local Growth Estimator (PPC + Organic + Maps)")
st.caption(
    "Everything here is **ranges + assumptions**. This is for planning and proposals — **not guarantees**."
)

# =========================================================
# TOP-LEVEL DATA UPLOADS
# FIX: GSC upload moved here for visibility (was buried in Organic tab)
# =========================================================
st.subheader("📁 Data Sources")

upload_col1, upload_col2 = st.columns(2, gap="large")

with upload_col1:
    st.markdown("#### Keyword Planner")
    st.caption("Required for PPC and Maps. Optional for Organic if you have GSC.")
    uploaded_kw = st.file_uploader(
        "Upload Keyword Planner export",
        type=["xlsx", "xls", "csv", "tsv"],
        key="kw_uploader",
        help="Export from Google Keyword Planner (.xlsx, .csv, or .tsv)"
    )

    df_kw = None
    total_avg_searches = None
    month_cols: List[str] = []

    if uploaded_kw:
        try:
            df_kw, total_avg_searches, month_cols = load_keyword_planner_file(uploaded_kw)
            st.success(f"✅ Loaded: {len(df_kw):,} keywords | **{total_avg_searches:,}** avg monthly searches")
            if month_cols:
                st.caption(f"Seasonality columns detected: {len(month_cols)} months")
            with st.expander("Preview keywords (first 25)"):
                st.dataframe(df_kw.head(25), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read Keyword Planner file: {e}")
            df_kw = None

with upload_col2:
    st.markdown("#### Google Search Console")
    st.caption("Optional. Provides site-specific baseline for Organic estimates.")
    gsc_csv = st.file_uploader(
        "Upload GSC Queries export",
        type=["csv", "tsv"],
        key="gsc_uploader",
        help="Export the 'Queries' report from Google Search Console (.csv)"
    )

    gsc_df = None
    gsc_summary: Dict[str, float] = {}

    if gsc_csv:
        try:
            gsc_df = load_gsc_queries_csv(gsc_csv)
            
            # GSC Date Range selector - CRITICAL for correct monthly estimates
            st.markdown("##### 📅 What date range does this export cover?")
            gsc_date_range = st.selectbox(
                "GSC Export Date Range",
                options=GSCDateRange.get_options(),
                index=4,  # Default to "Last 12 months" since that's common
                key="gsc_date_range",
                help="GSC exports show cumulative totals for the selected period. We'll divide by the appropriate number to get monthly averages."
            )
            gsc_period_divisor = GSCDateRange.get_divisor(gsc_date_range)
            
            # Calculate raw totals from file
            raw_total_impr = gsc_df["impressions"].sum()
            raw_total_clicks = gsc_df["clicks"].sum()
            
            # Convert to monthly averages
            monthly_impr = raw_total_impr / gsc_period_divisor
            monthly_clicks = raw_total_clicks / gsc_period_divisor
            
            gsc_summary = {
                "queries": float(gsc_df["query"].nunique()),
                "impressions": float(monthly_impr),  # Now monthly average
                "clicks": float(monthly_clicks),      # Now monthly average
                "ctr": float(raw_total_clicks / raw_total_impr) if raw_total_impr > 0 else 0.0,
                "avg_position": float(gsc_df["position"].mean()) if "position" in gsc_df.columns else math.nan,
                "raw_impressions": float(raw_total_impr),  # Keep raw for reference
                "raw_clicks": float(raw_total_clicks),
                "date_range": gsc_date_range,
                "period_divisor": gsc_period_divisor,
            }
            st.success(
                f"✅ Loaded: {int(gsc_summary['queries']):,} queries | "
                f"**{int(monthly_impr):,} impressions/mo** | "
                f"**{int(monthly_clicks):,} clicks/mo** | "
                f"CTR ~ {pct_str(gsc_summary['ctr'], 1)}"
            )
            st.caption(f"📊 Raw totals ({gsc_date_range}): {int(raw_total_impr):,} impressions, {int(raw_total_clicks):,} clicks → divided by {gsc_period_divisor:.2f} for monthly avg")
            with st.expander("Preview GSC data (top 25 by impressions)"):
                st.dataframe(
                    gsc_df.sort_values("impressions", ascending=False).head(25),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Could not read GSC CSV: {e}")
            gsc_df = None

# Show data status
has_kw = df_kw is not None
has_gsc = gsc_df is not None

if not has_kw and not has_gsc:
    st.warning("⚠️ Upload at least one data source to begin. Keyword Planner enables PPC + Maps. GSC enables site-specific Organic estimates.")

st.divider()

# =========================================================
# SHARED INPUTS
# =========================================================
st.subheader("⚙️ Shared Settings")

shared_col1, shared_col2 = st.columns([1.25, 1], gap="large")

with shared_col1:
    spend_text = st.text_area(
        "Monthly spend tiers (PPC only)",
        value="500, 1000, 1500, 2000, 3000, 4000",
        height=90,
        help="Comma-separated list of monthly spend levels to model"
    )
    spend_tiers = parse_spend_tiers(spend_text)
    if spend_tiers:
        st.caption(f"Will model: {', '.join([money(s) for s in spend_tiers])}")

with shared_col2:
    cvr = percent_input(
        label="Lead conversion rate (CVR) %",
        default_pct=5,
        recommended_min_pct=2,
        recommended_max_pct=8,
        hard_min_pct=0,
        hard_max_pct=50,
        step=0.5,
        key="cvr_pct",
        help_text="Percent of clicks/actions that become a qualified lead (call/form/chat).",
        explanation_md=(
            "**What it is:** The percent of clicks/actions that become a qualified lead.\n\n"
            "**How to use it:**\n"
            "- If you have no tracking, start conservative (3-5%).\n"
            "- If calls go unanswered, your real CVR is lower.\n"
            "- Well-optimized landing pages can achieve 8-12%.\n\n"
            "**Example:** 5% means 100 clicks → ~5 leads."
        )
    )

    business_hours_factor = percent_input(
        label="Business-hours demand factor %",
        default_pct=35,
        recommended_min_pct=25,
        recommended_max_pct=45,
        hard_min_pct=0,
        hard_max_pct=100,
        step=1,
        key="biz_hours_pct",
        help_text="Keyword volumes are 24/7. This approximates how much demand occurs during open hours.",
        explanation_md=(
            "**What it is:** Keyword volumes represent the full week (24/7). "
            "If you operate 9–5 M–F, you capture only a share.\n\n"
            "**How to use it:**\n"
            "- Start around **35%** for 9–5 M–F businesses.\n"
            "- Increase if your niche is daytime-heavy (B2B services).\n"
            "- Decrease if evenings/weekends dominate (entertainment, restaurants).\n\n"
            "This affects PPC, Organic (Keyword Planner mode), and Maps."
        )
    )

# Seasonality mode
seasonality_mode = False
selected_months: List[str] = []
if month_cols:
    seasonality_mode = st.checkbox("📅 Enable Seasonality Mode (month-by-month)", value=True)
    if seasonality_mode:
        default_months = month_cols[-3:] if len(month_cols) >= 3 else month_cols
        selected_months = st.multiselect(
            "Months to model",
            options=month_cols,
            default=default_months
        )
else:
    st.caption("💡 Seasonality Mode will appear after uploading a Keyword Planner file with monthly columns.")

st.divider()

# =========================================================
# TABS
# =========================================================
tab_ppc, tab_org, tab_maps, tab_summary = st.tabs(["📌 PPC", "🌿 Organic", "📍 Maps", "🧾 Summary"])


# =========================================================
# TAB: PPC
# FIX: Renamed postures, added validation, removed st.stop()
# =========================================================
with tab_ppc:
    st.subheader("PPC Estimator (Search only)")

    if not has_kw:
        st.error("⚠️ Upload a Keyword Planner XLSX (above) to use the PPC estimator.")
        st.info("PPC estimates require market demand data from Keyword Planner.")
        # FIX: No st.stop() - just don't render the rest
    else:
        st.caption("PPC output is capped by demand: searches × business-hours × impression share × CTR.")

        # FIX: Collapsible posture sections to reduce overwhelm
        # FIX: Renamed from "Mid-page" to "Conservative bidding" for clarity
        
        c1, c2 = st.columns(2, gap="large")

        with c1:
            with st.expander("### 🐢 Conservative Bidding Posture", expanded=True):
                st.caption(
                    "Lower bids = lower CPCs but less impression share. "
                    "Use this to model budget-conscious campaigns."
                )
                con_cpc_low = money_input(
                    label="CPC low ($)",
                    default_value=4.0,
                    recommended_min=3.0,
                    recommended_max=7.0,
                    hard_min=0.01,
                    hard_max=500.0,
                    step=0.25,
                    key="con_cpc_low",
                    help_text="Lower bound blended CPC for conservative bidding.",
                    explanation_md="Use your realistic blended CPC. If this is too low, click estimates become fantasy."
                )
                con_cpc_high = money_input(
                    label="CPC high ($)",
                    default_value=6.0,
                    recommended_min=4.0,
                    recommended_max=10.0,
                    hard_min=0.01,
                    hard_max=500.0,
                    step=0.25,
                    key="con_cpc_high",
                    help_text="Upper bound blended CPC for conservative bidding.",
                    explanation_md="Worst-case CPC for this posture."
                )
                # FIX: Cross-field validation
                cpc_con_valid = validate_range_pair(con_cpc_low, con_cpc_high, "CPC low", "CPC high", "Conservative")

                con_is_low = percent_input(
                    label="Impression share low %",
                    default_pct=10,
                    recommended_min_pct=8,
                    recommended_max_pct=20,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=1,
                    key="con_is_low",
                    help_text="Lower bound of how often your ad shows when eligible.",
                    explanation_md=(
                        "**Impression share** = percent of eligible auctions where you appear.\n\n"
                        "Conservative bidding usually means lower impression share (10-25%)."
                    )
                )
                con_is_high = percent_input(
                    label="Impression share high %",
                    default_pct=18,
                    recommended_min_pct=12,
                    recommended_max_pct=30,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=1,
                    key="con_is_high",
                    help_text="Upper bound impression share for conservative bidding.",
                    explanation_md="Optimistic impression share for conservative bidding."
                )
                is_con_valid = validate_range_pair(con_is_low, con_is_high, "IS low", "IS high", "Conservative")

                con_ctr_low = percent_input(
                    label="CTR low %",
                    default_pct=4,
                    recommended_min_pct=3,
                    recommended_max_pct=7,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=0.5,
                    key="con_ctr_low",
                    help_text="Lower bound click-through rate.",
                    explanation_md="CTR is the percent of impressions that become clicks."
                )
                con_ctr_high = percent_input(
                    label="CTR high %",
                    default_pct=6,
                    recommended_min_pct=4,
                    recommended_max_pct=10,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=0.5,
                    key="con_ctr_high",
                    help_text="Upper bound click-through rate.",
                    explanation_md="Optimistic CTR for conservative bidding."
                )
                ctr_con_valid = validate_range_pair(con_ctr_low, con_ctr_high, "CTR low", "CTR high", "Conservative")

        with c2:
            with st.expander("### 🚀 Aggressive Bidding Posture", expanded=True):
                st.caption(
                    "Higher bids = higher CPCs but more impression share and better positions. "
                    "Use this to model growth-focused campaigns."
                )
                agg_cpc_low = money_input(
                    label="CPC low ($)",
                    default_value=7.0,
                    recommended_min=5.0,
                    recommended_max=12.0,
                    hard_min=0.01,
                    hard_max=500.0,
                    step=0.25,
                    key="agg_cpc_low",
                    help_text="Lower bound blended CPC for aggressive bidding.",
                    explanation_md="Aggressive bidding usually costs more per click."
                )
                agg_cpc_high = money_input(
                    label="CPC high ($)",
                    default_value=10.0,
                    recommended_min=7.0,
                    recommended_max=20.0,
                    hard_min=0.01,
                    hard_max=500.0,
                    step=0.25,
                    key="agg_cpc_high",
                    help_text="Upper bound blended CPC for aggressive bidding.",
                    explanation_md="Worst-case CPC for aggressive bidding."
                )
                cpc_agg_valid = validate_range_pair(agg_cpc_low, agg_cpc_high, "CPC low", "CPC high", "Aggressive")

                agg_is_low = percent_input(
                    label="Impression share low %",
                    default_pct=25,
                    recommended_min_pct=15,
                    recommended_max_pct=45,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=1,
                    key="agg_is_low",
                    help_text="Lower bound impression share for aggressive bidding.",
                    explanation_md="If budget is tight, even aggressive bids may not deliver high impression share."
                )
                agg_is_high = percent_input(
                    label="Impression share high %",
                    default_pct=40,
                    recommended_min_pct=25,
                    recommended_max_pct=60,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=1,
                    key="agg_is_high",
                    help_text="Upper bound impression share for aggressive bidding.",
                    explanation_md="Optimistic impression share if you're bidding hard."
                )
                is_agg_valid = validate_range_pair(agg_is_low, agg_is_high, "IS low", "IS high", "Aggressive")

                agg_ctr_low = percent_input(
                    label="CTR low %",
                    default_pct=7,
                    recommended_min_pct=5,
                    recommended_max_pct=12,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=0.5,
                    key="agg_ctr_low",
                    help_text="Lower bound CTR for aggressive bidding.",
                    explanation_md="Better ad positions typically have higher CTR."
                )
                agg_ctr_high = percent_input(
                    label="CTR high %",
                    default_pct=10,
                    recommended_min_pct=7,
                    recommended_max_pct=18,
                    hard_min_pct=0,
                    hard_max_pct=100,
                    step=0.5,
                    key="agg_ctr_high",
                    help_text="Upper bound CTR for aggressive bidding.",
                    explanation_md="Optimistic CTR for aggressive bidding."
                )
                ctr_agg_valid = validate_range_pair(agg_ctr_low, agg_ctr_high, "CTR low", "CTR high", "Aggressive")

        # Check all validations passed
        ppc_inputs_valid = all([
            cpc_con_valid, is_con_valid, ctr_con_valid,
            cpc_agg_valid, is_agg_valid, ctr_agg_valid
        ])

        st.divider()

        if not ppc_inputs_valid:
            st.error("⛔ Please fix the invalid ranges above before viewing estimates.")
        else:
            st.markdown(confidence_badge(Confidence.MEDIUM))
            st.caption("Demand-based estimate from Keyword Planner + your assumptions. Not a promise.")

            def build_ppc_table(total_searches: float, posture: str) -> Tuple[pd.DataFrame, Tuple[float, float]]:
                """Build PPC estimate table for a given posture"""
                if posture == "conservative":
                    cap_low, cap_high = compute_click_caps(
                        total_searches=total_searches,
                        business_hours_factor=business_hours_factor,
                        impr_share_low=con_is_low,
                        impr_share_high=con_is_high,
                        ctr_low=con_ctr_low,
                        ctr_high=con_ctr_high,
                    )
                    cpc_low, cpc_high = con_cpc_low, con_cpc_high
                else:
                    cap_low, cap_high = compute_click_caps(
                        total_searches=total_searches,
                        business_hours_factor=business_hours_factor,
                        impr_share_low=agg_is_low,
                        impr_share_high=agg_is_high,
                        ctr_low=agg_ctr_low,
                        ctr_high=agg_ctr_high,
                    )
                    cpc_low, cpc_high = agg_cpc_low, agg_cpc_high

                rows = []
                for spend in spend_tiers:
                    (clicks_lo, clicks_hi), (leads_lo, leads_hi), (cpl_best, cpl_worst) = compute_ranges_with_caps(
                        spend=spend,
                        cpc_low=cpc_low,
                        cpc_high=cpc_high,
                        clicks_cap_low=cap_low,
                        clicks_cap_high=cap_high,
                        cvr=cvr,
                    )
                    rows.append({
                        "Spend": money(spend),
                        "Clicks (low–high)": f"{num(clicks_lo, 0)}–{num(clicks_hi, 0)}",
                        "Leads (low–high)": f"{num(leads_lo, 1)}–{num(leads_hi, 1)}",
                        "CPL (best–worst)": f"{money(cpl_best)}–{money(cpl_worst)}",
                    })

                return pd.DataFrame(rows), (cap_low, cap_high)

            def render_ppc(total_searches: float, label: str, period_key: str = "avg"):
                """Render PPC estimates for a given search volume"""
                st.markdown(f"### {label}")
                con_df, con_caps = build_ppc_table(total_searches, "conservative")
                agg_df, agg_caps = build_ppc_table(total_searches, "aggressive")

                st.info(
                    f"**Market click caps** (prevents over-promising): "
                    f"Conservative **{con_caps[0]:.0f}–{con_caps[1]:.0f}** clicks/mo | "
                    f"Aggressive **{agg_caps[0]:.0f}–{agg_caps[1]:.0f}** clicks/mo"
                )

                a, b = st.columns(2, gap="large")
                with a:
                    st.markdown("#### 🐢 Conservative Bidding")
                    st.dataframe(con_df, use_container_width=True, hide_index=True)
                    download_table_button(
                        con_df,
                        f"ppc_conservative_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "📥 Download Conservative",
                        key=f"dl_ppc_con_{period_key}"
                    )
                with b:
                    st.markdown("#### 🚀 Aggressive Bidding")
                    st.dataframe(agg_df, use_container_width=True, hide_index=True)
                    download_table_button(
                        agg_df,
                        f"ppc_aggressive_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "📥 Download Aggressive",
                        key=f"dl_ppc_agg_{period_key}"
                    )

            # Render for selected months or average
            if seasonality_mode and selected_months:
                month_tabs = st.tabs(selected_months)
                for t, month in zip(month_tabs, selected_months):
                    with t:
                        total = sum_month_searches(df_kw, month)
                        month_key = month.replace(" ", "_").replace(":", "")
                        render_ppc(total, f"{month} (searches: {total:,.0f})", month_key)
            else:
                render_ppc(float(total_avg_searches), f"Average month (searches: {total_avg_searches:,.0f})", "avg_month")

            # Assumptions expander
            with st.expander("🧠 Assumptions used (PPC)"):
                st.markdown(
                    f"**Shared**\n"
                    f"- CVR: **{pct_str(cvr, 1)}**\n"
                    f"- Business-hours demand factor: **{pct_str(business_hours_factor, 0)}**\n\n"
                    f"**Conservative Bidding**\n"
                    f"- CPC: **${con_cpc_low:.2f}–${con_cpc_high:.2f}**\n"
                    f"- Impression share: **{pct_str(con_is_low, 0)}–{pct_str(con_is_high, 0)}**\n"
                    f"- CTR: **{pct_str(con_ctr_low, 0)}–{pct_str(con_ctr_high, 0)}**\n\n"
                    f"**Aggressive Bidding**\n"
                    f"- CPC: **${agg_cpc_low:.2f}–${agg_cpc_high:.2f}**\n"
                    f"- Impression share: **{pct_str(agg_is_low, 0)}–{pct_str(agg_is_high, 0)}**\n"
                    f"- CTR: **{pct_str(agg_ctr_low, 0)}–{pct_str(agg_ctr_high, 0)}**\n"
                )

            st.caption(
                "⚠️ Disclaimer: Estimates depend on auction dynamics, ad quality, landing pages, "
                "tracking setup, and sales handling. These are not guarantees."
            )


# =========================================================
# TAB: Organic
# FIX: Removed st.stop(), GSC upload moved to top
# =========================================================
with tab_org:
    st.subheader("Organic Estimator (SEO)")
    st.caption("Uses Keyword Planner (market demand) and/or GSC (site-specific baseline).")

    # Determine mode
    if has_gsc and has_kw:
        org_mode = DataMode.BOTH
    elif has_gsc:
        org_mode = DataMode.GSC_ONLY
    elif has_kw:
        org_mode = DataMode.KW_ONLY
    else:
        org_mode = DataMode.NONE

    if org_mode == DataMode.NONE:
        st.error("⚠️ Upload at least one data source (above) to use the Organic estimator.")
        st.info("GSC provides site-specific estimates. Keyword Planner provides market-based estimates.")
    else:
        # Show mode indicator
        if org_mode == DataMode.BOTH:
            st.markdown(confidence_badge(Confidence.HIGH))
            st.caption("Using GSC for baseline + market expansion potential from Keyword Planner.")
        elif org_mode == DataMode.GSC_ONLY:
            st.markdown(confidence_badge(Confidence.MEDIUM))
            st.caption("Using GSC for site-specific baseline. No market ceiling from Keyword Planner.")
        else:
            st.markdown(confidence_badge(Confidence.LOW))
            st.caption("Using Keyword Planner market demand + scenario CTR ranges. No site-specific baseline.")

        st.divider()

        # Inputs
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Organic Scenario Settings")

            # SERP leakage (only for KW mode)
            serp_leakage = percent_input(
                label="SERP leakage factor %",
                default_pct=80,
                recommended_min_pct=60,
                recommended_max_pct=90,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="serp_leakage_pct",
                help_text="Used only when estimating from Keyword Planner (not GSC).",
                explanation_md=(
                    "**What it is:** In many local SERPs, Ads/Maps/LSA/instant answers reduce organic clicks.\n\n"
                    "**When it matters:** Only in **Keyword Planner mode** (market-based estimation).\n"
                    "**When it does NOT matter:** In **GSC mode**, impressions already reflect the real SERP.\n\n"
                    "**Typical:** 60–90% depending on SERP crowding."
                )
            )

            st.markdown("#### CTR Scenarios")

            org_con_low = percent_input(
                label="Conservative CTR low %",
                default_pct=1,
                recommended_min_pct=1,
                recommended_max_pct=3,
                hard_min_pct=0,
                hard_max_pct=50,
                step=0.5,
                key="org_con_low",
                help_text="Conservative organic CTR assumption.",
                explanation_md="Conservative = small slice of demand. Typical 1–3% for local non-brand."
            )
            org_con_high = percent_input(
                label="Conservative CTR high %",
                default_pct=3,
                recommended_min_pct=2,
                recommended_max_pct=5,
                hard_min_pct=0,
                hard_max_pct=50,
                step=0.5,
                key="org_con_high",
                help_text="Upper bound for conservative CTR.",
                explanation_md="Upper edge of conservative assumption."
            )
            org_con_valid = validate_range_pair(org_con_low, org_con_high, "Conservative low", "Conservative high", "Organic CTR")

            org_exp_low = percent_input(
                label="Expected CTR low %",
                default_pct=3,
                recommended_min_pct=3,
                recommended_max_pct=7,
                hard_min_pct=0,
                hard_max_pct=60,
                step=0.5,
                key="org_exp_low",
                help_text="Expected CTR assumption.",
                explanation_md="Expected = solid page-1 presence. Typical 3–7%."
            )
            org_exp_high = percent_input(
                label="Expected CTR high %",
                default_pct=7,
                recommended_min_pct=5,
                recommended_max_pct=12,
                hard_min_pct=0,
                hard_max_pct=60,
                step=0.5,
                key="org_exp_high",
                help_text="Upper bound for expected CTR.",
                explanation_md="Upper edge of expected assumption."
            )
            org_exp_valid = validate_range_pair(org_exp_low, org_exp_high, "Expected low", "Expected high", "Organic CTR")

            org_agg_low = percent_input(
                label="Aggressive CTR low %",
                default_pct=7,
                recommended_min_pct=7,
                recommended_max_pct=15,
                hard_min_pct=0,
                hard_max_pct=80,
                step=0.5,
                key="org_agg_low",
                help_text="Aggressive CTR assumption.",
                explanation_md="Aggressive = top-of-SERP dominance. Typical 7–15% if truly dominant."
            )
            org_agg_high = percent_input(
                label="Aggressive CTR high %",
                default_pct=15,
                recommended_min_pct=10,
                recommended_max_pct=25,
                hard_min_pct=0,
                hard_max_pct=80,
                step=0.5,
                key="org_agg_high",
                help_text="Upper bound for aggressive CTR.",
                explanation_md="Upper edge of aggressive assumption."
            )
            org_agg_valid = validate_range_pair(org_agg_low, org_agg_high, "Aggressive low", "Aggressive high", "Organic CTR")

            # Net-new factor (only for BOTH mode)
            net_new_factor = percent_input(
                label="Net-new coverage factor (BOTH mode only) %",
                default_pct=10,
                recommended_min_pct=5,
                recommended_max_pct=15,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="net_new_factor_pct",
                help_text="If you have both GSC + Keyword Planner, this models expansion beyond current visibility.",
                explanation_md=(
                    "**What it is:** Models **additional** clicks from **new queries** not yet captured in GSC.\n\n"
                    "**Keep it conservative** unless you know the site can expand aggressively into new query territory."
                )
            )

        with right:
            st.markdown("### Time Ramp (SEO)")
            st.caption("SEO results take time. This models how leads ramp up over months.")

            ramp_m1 = percent_input(
                label="Month 1 ramp %",
                default_pct=15,
                recommended_min_pct=10,
                recommended_max_pct=25,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="org_ramp_m1",
                help_text="Percent of steady-state expected in month 1.",
                explanation_md="Month 1 is usually setup/indexing/testing. SEO is not instant."
            )
            ramp_m2 = percent_input(
                label="Month 2 ramp %",
                default_pct=30,
                recommended_min_pct=20,
                recommended_max_pct=45,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="org_ramp_m2",
                help_text="Percent of steady-state expected in month 2.",
                explanation_md="Month 2 often shows early traction if execution is good."
            )
            ramp_m3 = percent_input(
                label="Month 3 ramp %",
                default_pct=60,
                recommended_min_pct=40,
                recommended_max_pct=80,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="org_ramp_m3",
                help_text="Percent of steady-state expected in month 3.",
                explanation_md="Month 3 is where compounding can start showing up."
            )
            ramp_m6 = percent_input(
                label="Month 6+ ramp %",
                default_pct=100,
                recommended_min_pct=80,
                recommended_max_pct=100,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="org_ramp_m6",
                help_text="Percent of steady-state by month 6+.",
                explanation_md="Steady-state for this scenario model."
            )

        org_inputs_valid = all([org_con_valid, org_exp_valid, org_agg_valid])

        st.divider()

        if not org_inputs_valid:
            st.error("⛔ Please fix the invalid ranges above before viewing estimates.")
        else:
            org_ramps = [
                ("Month 1", ramp_m1),
                ("Month 2", ramp_m2),
                ("Month 3", ramp_m3),
                ("Month 6+", ramp_m6),
            ]

            def render_organic_for_search_total(label: str, kw_total_searches: Optional[float], period_key: str = "avg"):
                """Render organic estimates for a given context"""
                st.markdown(f"### {label}")

                # Calculate scenarios based on mode
                if has_gsc:
                    gsc_impr = float(gsc_summary["impressions"])
                    gsc_clicks = float(gsc_summary["clicks"])
                    date_range = gsc_summary.get("date_range", "Unknown")
                    st.info(
                        f"**GSC baseline (monthly avg):** {int(gsc_impr):,} impressions/mo | "
                        f"{int(gsc_clicks):,} clicks/mo | CTR ~ {pct_str(gsc_summary['ctr'], 1)}\n\n"
                        f"_Source: {date_range} export, converted to monthly average_"
                    )

                    con_clicks = organic_clicks_from_gsc(gsc_impr, org_con_low, org_con_high)
                    exp_clicks = organic_clicks_from_gsc(gsc_impr, org_exp_low, org_exp_high)
                    agg_clicks = organic_clicks_from_gsc(gsc_impr, org_agg_low, org_agg_high)
                    base_note = "GSC-based (site-specific): clicks = monthly impressions × scenario CTR."
                else:
                    con_clicks = organic_clicks_from_keyword_demand(
                        kw_total_searches, business_hours_factor, org_con_low, org_con_high, serp_leakage
                    )
                    exp_clicks = organic_clicks_from_keyword_demand(
                        kw_total_searches, business_hours_factor, org_exp_low, org_exp_high, serp_leakage
                    )
                    agg_clicks = organic_clicks_from_keyword_demand(
                        kw_total_searches, business_hours_factor, org_agg_low, org_agg_high, serp_leakage
                    )
                    base_note = "Market-based (Keyword Planner): clicks = demand × biz-hours × CTR × leakage."

                # Steady-state table
                scenarios = [
                    (Scenario.CONSERVATIVE, con_clicks),
                    (Scenario.EXPECTED, exp_clicks),
                    (Scenario.AGGRESSIVE, agg_clicks),
                ]
                steady_df = build_scenario_table(scenarios, cvr, "Clicks/mo", "Leads/mo")
                st.dataframe(steady_df, use_container_width=True, hide_index=True)
                st.caption(base_note)
                download_table_button(
                    steady_df,
                    f"organic_scenarios_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "📥 Download Scenarios",
                    key=f"dl_org_scenarios_{period_key}"
                )

                # Ramp table
                st.markdown("#### Time Ramp (leads over time)")
                ramp_df = build_ramp_table(scenarios, cvr, org_ramps)
                st.dataframe(ramp_df, use_container_width=True, hide_index=True)
                download_table_button(
                    ramp_df,
                    f"organic_ramp_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "📥 Download Ramp",
                    key=f"dl_org_ramp_{period_key}"
                )

                # BOTH mode: net-new expansion
                if org_mode == DataMode.BOTH and kw_total_searches is not None:
                    st.markdown("#### Market Expansion (beyond current GSC footprint)")
                    st.caption("Conservative estimate of incremental clicks from queries not yet captured.")

                    kw_expected_clicks = organic_clicks_from_keyword_demand(
                        kw_total_searches, business_hours_factor, org_exp_low, org_exp_high, serp_leakage
                    )
                    net_new_clicks = (
                        kw_expected_clicks[0] * net_new_factor,
                        kw_expected_clicks[1] * net_new_factor
                    )

                    net_new_df = pd.DataFrame([
                        {
                            "Metric": "Net-new clicks/mo (low-high)",
                            "Value": f"{num(net_new_clicks[0], 0)}-{num(net_new_clicks[1], 0)}",
                        },
                        {
                            "Metric": "Net-new leads/mo (low-high)",
                            "Value": f"{num(net_new_clicks[0] * cvr, 1)}-{num(net_new_clicks[1] * cvr, 1)}",
                        }
                    ])
                    st.dataframe(net_new_df, use_container_width=True, hide_index=True)
                    download_table_button(
                        net_new_df,
                        f"organic_expansion_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "📥 Download Expansion",
                        key=f"dl_org_exp_{period_key}"
                    )

            # Render for selected months or average
            if has_kw and seasonality_mode and selected_months:
                month_tabs = st.tabs(selected_months)
                for t, month in zip(month_tabs, selected_months):
                    with t:
                        total = sum_month_searches(df_kw, month)
                        month_key = month.replace(" ", "_").replace(":", "")
                        render_organic_for_search_total(f"{month} (searches: {total:,.0f})", total, month_key)
            else:
                kw_total = float(total_avg_searches) if has_kw else None
                label = f"Average month (searches: {total_avg_searches:,.0f})" if has_kw else "GSC-only (no Keyword Planner ceiling)"
                render_organic_for_search_total(label, kw_total, "avg_month")

            # Assumptions expander
            with st.expander("🧠 Assumptions used (Organic)"):
                st.markdown(
                    f"**Shared**\n"
                    f"- CVR: **{pct_str(cvr, 1)}**\n"
                    f"- Business-hours demand factor: **{pct_str(business_hours_factor, 0)}**\n\n"
                    f"**Mode-specific**\n"
                    f"- SERP leakage factor (KW mode only): **{pct_str(serp_leakage, 0)}**\n"
                    f"- Net-new coverage factor (BOTH mode only): **{pct_str(net_new_factor, 0)}**\n\n"
                    f"**CTR Scenarios**\n"
                    f"- Conservative: **{pct_str(org_con_low, 0)}–{pct_str(org_con_high, 0)}**\n"
                    f"- Expected: **{pct_str(org_exp_low, 0)}–{pct_str(org_exp_high, 0)}**\n"
                    f"- Aggressive: **{pct_str(org_agg_low, 0)}–{pct_str(org_agg_high, 0)}**\n\n"
                    f"**Time Ramp**\n"
                    f"- Month 1: **{pct_str(ramp_m1, 0)}** | Month 2: **{pct_str(ramp_m2, 0)}** | "
                    f"Month 3: **{pct_str(ramp_m3, 0)}** | Month 6+: **{pct_str(ramp_m6, 0)}**\n"
                )

            st.caption(
                "⚠️ Disclaimer: Organic results depend on competition, site quality, content, links, "
                "reputation, SERP layout, and time. Estimates are not guarantees."
            )


# =========================================================
# TAB: Maps
# FIX: Removed st.stop(), added validation
# =========================================================
with tab_maps:
    st.subheader("Maps Estimator (Google Business Profile)")

    if not has_kw:
        st.error("⚠️ Upload a Keyword Planner XLSX (above) to use the Maps estimator.")
        st.info("Maps estimates require market demand data from Keyword Planner.")
    else:
        st.caption("Maps model uses market demand × action share. Outputs are ranges, not promises.")
        st.markdown(confidence_badge(Confidence.LOW))
        st.caption("💡 To increase confidence: Upload GBP Insights data (feature coming soon).")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Maps Action Share Scenarios")

            maps_con_low = percent_input(
                label="Conservative action share low %",
                default_pct=1,
                recommended_min_pct=1,
                recommended_max_pct=3,
                hard_min_pct=0,
                hard_max_pct=50,
                step=0.5,
                key="maps_con_low",
                help_text="Low visibility / inconsistent pack presence.",
                explanation_md="Conservative typically 1–3% for many local markets."
            )
            maps_con_high = percent_input(
                label="Conservative action share high %",
                default_pct=3,
                recommended_min_pct=2,
                recommended_max_pct=5,
                hard_min_pct=0,
                hard_max_pct=50,
                step=0.5,
                key="maps_con_high",
                help_text="Upper bound for conservative maps action share.",
                explanation_md="Upper edge of conservative."
            )
            maps_con_valid = validate_range_pair(maps_con_low, maps_con_high, "Conservative low", "Conservative high", "Maps")

            maps_exp_low = percent_input(
                label="Expected action share low %",
                default_pct=3,
                recommended_min_pct=3,
                recommended_max_pct=10,
                hard_min_pct=0,
                hard_max_pct=70,
                step=0.5,
                key="maps_exp_low",
                help_text="Decent and improving pack presence.",
                explanation_md="Expected often 3–10% depending on dominance and proximity."
            )
            maps_exp_high = percent_input(
                label="Expected action share high %",
                default_pct=10,
                recommended_min_pct=7,
                recommended_max_pct=15,
                hard_min_pct=0,
                hard_max_pct=70,
                step=0.5,
                key="maps_exp_high",
                help_text="Upper bound for expected maps action share.",
                explanation_md="Upper edge of expected."
            )
            maps_exp_valid = validate_range_pair(maps_exp_low, maps_exp_high, "Expected low", "Expected high", "Maps")

            maps_agg_low = percent_input(
                label="Aggressive action share low %",
                default_pct=10,
                recommended_min_pct=10,
                recommended_max_pct=20,
                hard_min_pct=0,
                hard_max_pct=90,
                step=0.5,
                key="maps_agg_low",
                help_text="Strong, consistent top 3 visibility.",
                explanation_md="Aggressive only if truly dominant in the service area."
            )
            maps_agg_high = percent_input(
                label="Aggressive action share high %",
                default_pct=25,
                recommended_min_pct=15,
                recommended_max_pct=30,
                hard_min_pct=0,
                hard_max_pct=90,
                step=0.5,
                key="maps_agg_high",
                help_text="Upper bound for aggressive maps action share.",
                explanation_md="Upper edge of aggressive."
            )
            maps_agg_valid = validate_range_pair(maps_agg_low, maps_agg_high, "Aggressive low", "Aggressive high", "Maps")

        with right:
            st.markdown("### Action → Lead Conversion")

            maps_qual = percent_input(
                label="Maps action qualification rate %",
                default_pct=70,
                recommended_min_pct=50,
                recommended_max_pct=85,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="maps_qual",
                help_text="Percent of actions (calls/clicks/directions) that become qualified leads.",
                explanation_md=(
                    "Not every action is a real lead. This converts actions into qualified leads.\n\n"
                    "- If the business misses calls → reduce this.\n"
                    "- If they get lots of wrong-fit inquiries → reduce this.\n"
                    "- Well-staffed businesses with good reviews → keep at 70-85%."
                )
            )

            st.markdown("### Time Ramp (Maps)")

            maps_ramp_m1 = percent_input(
                label="Month 1 ramp %",
                default_pct=20,
                recommended_min_pct=10,
                recommended_max_pct=35,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="maps_ramp_m1",
                help_text="Percent of steady-state in month 1.",
                explanation_md="Maps can move faster than SEO, but still ramps."
            )
            maps_ramp_m3 = percent_input(
                label="Month 3 ramp %",
                default_pct=60,
                recommended_min_pct=40,
                recommended_max_pct=80,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="maps_ramp_m3",
                help_text="Percent of steady-state by month 3.",
                explanation_md="Month 3 often shows compounding if execution is consistent."
            )
            maps_ramp_m6 = percent_input(
                label="Month 6+ ramp %",
                default_pct=100,
                recommended_min_pct=80,
                recommended_max_pct=100,
                hard_min_pct=0,
                hard_max_pct=100,
                step=1,
                key="maps_ramp_m6",
                help_text="Percent of steady-state by month 6+.",
                explanation_md="Steady-state for this scenario model."
            )

        maps_inputs_valid = all([maps_con_valid, maps_exp_valid, maps_agg_valid])

        st.divider()

        if not maps_inputs_valid:
            st.error("⛔ Please fix the invalid ranges above before viewing estimates.")
        else:
            maps_ramps = [
                ("Month 1", maps_ramp_m1),
                ("Month 3", maps_ramp_m3),
                ("Month 6+", maps_ramp_m6),
            ]

            def actions_to_leads(actions_low_high: Tuple[float, float]) -> Tuple[float, float]:
                return (actions_low_high[0] * maps_qual, actions_low_high[1] * maps_qual)

            def render_maps(total_searches: float, label: str, period_key: str = "avg"):
                """Render Maps estimates for a given search volume"""
                st.markdown(f"### {label}")

                con_actions = maps_actions_from_demand(total_searches, business_hours_factor, maps_con_low, maps_con_high)
                exp_actions = maps_actions_from_demand(total_searches, business_hours_factor, maps_exp_low, maps_exp_high)
                agg_actions = maps_actions_from_demand(total_searches, business_hours_factor, maps_agg_low, maps_agg_high)

                # Steady-state table (actions → leads, not clicks → leads)
                steady_df = pd.DataFrame([
                    {
                        "Scenario": Scenario.CONSERVATIVE,
                        "Actions/mo (low-high)": f"{num(con_actions[0], 0)}-{num(con_actions[1], 0)}",
                        "Qualified leads/mo (low-high)": f"{num(actions_to_leads(con_actions)[0], 1)}-{num(actions_to_leads(con_actions)[1], 1)}"
                    },
                    {
                        "Scenario": Scenario.EXPECTED,
                        "Actions/mo (low-high)": f"{num(exp_actions[0], 0)}-{num(exp_actions[1], 0)}",
                        "Qualified leads/mo (low-high)": f"{num(actions_to_leads(exp_actions)[0], 1)}-{num(actions_to_leads(exp_actions)[1], 1)}"
                    },
                    {
                        "Scenario": Scenario.AGGRESSIVE,
                        "Actions/mo (low-high)": f"{num(agg_actions[0], 0)}-{num(agg_actions[1], 0)}",
                        "Qualified leads/mo (low-high)": f"{num(actions_to_leads(agg_actions)[0], 1)}-{num(actions_to_leads(agg_actions)[1], 1)}"
                    },
                ])
                st.dataframe(steady_df, use_container_width=True, hide_index=True)
                download_table_button(
                    steady_df,
                    f"maps_scenarios_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "📥 Download Scenarios",
                    key=f"dl_maps_scenarios_{period_key}"
                )

                # Ramp table
                st.markdown("#### Time Ramp (Maps leads over time)")
                ramp_rows = []
                for name, actions in [(Scenario.CONSERVATIVE, con_actions), (Scenario.EXPECTED, exp_actions), (Scenario.AGGRESSIVE, agg_actions)]:
                    leads = actions_to_leads(actions)
                    row = {"Scenario": name}
                    for month_label, ramp_pct in maps_ramps:
                        ramped = apply_ramp(leads[0], leads[1], ramp_pct)
                        row[f"{month_label} leads"] = f"{num(ramped[0], 1)}-{num(ramped[1], 1)}"
                    ramp_rows.append(row)
                ramp_df = pd.DataFrame(ramp_rows)
                st.dataframe(ramp_df, use_container_width=True, hide_index=True)
                download_table_button(
                    ramp_df,
                    f"maps_ramp_{period_key}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "📥 Download Ramp",
                    key=f"dl_maps_ramp_{period_key}"
                )

            # Render for selected months or average
            if seasonality_mode and selected_months:
                month_tabs = st.tabs(selected_months)
                for t, month in zip(month_tabs, selected_months):
                    with t:
                        total = sum_month_searches(df_kw, month)
                        month_key = month.replace(" ", "_").replace(":", "")
                        render_maps(total, f"{month} (searches: {total:,.0f})", month_key)
            else:
                render_maps(float(total_avg_searches), f"Average month (searches: {total_avg_searches:,.0f})", "avg_month")

            # Assumptions expander
            with st.expander("🧠 Assumptions used (Maps)"):
                st.markdown(
                    f"**Shared**\n"
                    f"- Business-hours demand factor: **{pct_str(business_hours_factor, 0)}**\n"
                    f"- Action qualification rate: **{pct_str(maps_qual, 0)}**\n\n"
                    f"**Action Share Scenarios**\n"
                    f"- Conservative: **{pct_str(maps_con_low, 0)}–{pct_str(maps_con_high, 0)}**\n"
                    f"- Expected: **{pct_str(maps_exp_low, 0)}–{pct_str(maps_exp_high, 0)}**\n"
                    f"- Aggressive: **{pct_str(maps_agg_low, 0)}–{pct_str(maps_agg_high, 0)}**\n\n"
                    f"**Time Ramp**\n"
                    f"- Month 1: **{pct_str(maps_ramp_m1, 0)}** | Month 3: **{pct_str(maps_ramp_m3, 0)}** | "
                    f"Month 6+: **{pct_str(maps_ramp_m6, 0)}**\n"
                )

            st.caption(
                "⚠️ Disclaimer: Maps depends on proximity, categories, reviews, competition, and ongoing GBP activity. "
                "Estimates are not guarantees."
            )


# =========================================================
# TAB: Summary
# FIX: Added channel overlap disclaimer, better spend context
# =========================================================
with tab_summary:
    st.subheader("Executive Summary (All Channels)")
    st.caption("One view. Still ranges. Still not a promise.")

    if not has_kw and not has_gsc:
        st.error("⚠️ Upload Keyword Planner and/or GSC (above) to generate a summary.")
    else:
        # Choose summary month
        if has_kw and seasonality_mode and selected_months:
            chosen = selected_months[-1]
            total_searches_summary = sum_month_searches(df_kw, chosen)
            label = chosen
        elif has_kw:
            total_searches_summary = float(total_avg_searches)
            label = "Average month"
        else:
            total_searches_summary = None
            label = "GSC-only"

        # Confidence badge
        if has_kw and has_gsc:
            st.markdown(confidence_badge(Confidence.HIGH))
        elif has_gsc:
            st.markdown(confidence_badge(Confidence.MEDIUM))
        else:
            st.markdown(confidence_badge(Confidence.LOW))

        st.markdown(f"### Summary for: **{label}**")

        # FIX: Channel overlap disclaimer
        st.warning(
            "⚠️ **Channel Overlap Notice:** PPC, Organic, and Maps are not mutually exclusive. "
            "A single searcher may see your ad, organic listing, AND Maps pack. "
            "**Total leads across channels are NOT purely additive** — expect some overlap."
        )

        # Build summary rows
        rows = []

        # PPC summary - now with spend context
        if has_kw:
            # Use aggressive posture caps as the "ceiling"
            ppc_agg_cap_low, ppc_agg_cap_high = compute_click_caps(
                total_searches=total_searches_summary,
                business_hours_factor=business_hours_factor,
                impr_share_low=agg_is_low,
                impr_share_high=agg_is_high,
                ctr_low=agg_ctr_low,
                ctr_high=agg_ctr_high,
            )
            ppc_leads_cap = (ppc_agg_cap_low * cvr, ppc_agg_cap_high * cvr)

            # Also show a representative spend tier estimate
            if spend_tiers:
                mid_tier = spend_tiers[len(spend_tiers) // 2]  # Middle tier
                (_, _), (mid_leads_lo, mid_leads_hi), _ = compute_ranges_with_caps(
                    spend=mid_tier,
                    cpc_low=agg_cpc_low,
                    cpc_high=agg_cpc_high,
                    clicks_cap_low=ppc_agg_cap_low,
                    clicks_cap_high=ppc_agg_cap_high,
                    cvr=cvr,
                )
                ppc_note = f"At {money(mid_tier)}/mo (aggressive): {num(mid_leads_lo, 1)}–{num(mid_leads_hi, 1)} leads. Market ceiling shown."
            else:
                ppc_note = "Market ceiling (demand cap at unlimited budget)."

            rows.append({
                "Channel": "PPC (Aggressive ceiling)",
                "Traffic/Actions (low-high)": f"{num(ppc_agg_cap_low, 0)}-{num(ppc_agg_cap_high, 0)} clicks",
                "Leads (low-high)": f"{num(ppc_leads_cap[0], 1)}-{num(ppc_leads_cap[1], 1)}",
                "Notes": ppc_note,
            })

        # Organic summary
        if has_gsc:
            gsc_impr = float(gsc_summary["impressions"])
            org_clicks_low, org_clicks_high = organic_clicks_from_gsc(gsc_impr, org_exp_low, org_exp_high)
            org_note = "GSC-based expected scenario."
        elif has_kw:
            org_clicks_low, org_clicks_high = organic_clicks_from_keyword_demand(
                total_searches_summary, business_hours_factor, org_exp_low, org_exp_high, serp_leakage
            )
            org_note = "Keyword Planner-based expected scenario."
        else:
            org_clicks_low, org_clicks_high = 0.0, 0.0
            org_note = "No data."

        org_leads_low, org_leads_high = (org_clicks_low * cvr, org_clicks_high * cvr)
        rows.append({
            "Channel": "Organic (Expected)",
            "Traffic/Actions (low-high)": f"{num(org_clicks_low, 0)}-{num(org_clicks_high, 0)} clicks",
            "Leads (low-high)": f"{num(org_leads_low, 1)}-{num(org_leads_high, 1)}",
            "Notes": org_note,
        })

        # Maps summary
        if has_kw:
            maps_actions_low, maps_actions_high = maps_actions_from_demand(
                total_searches_summary, business_hours_factor, maps_exp_low, maps_exp_high
            )
            maps_leads_low, maps_leads_high = (maps_actions_low * maps_qual, maps_actions_high * maps_qual)
            rows.append({
                "Channel": "Maps (Expected)",
                "Traffic/Actions (low-high)": f"{num(maps_actions_low, 0)}-{num(maps_actions_high, 0)} actions",
                "Leads (low-high)": f"{num(maps_leads_low, 1)}-{num(maps_leads_high, 1)}",
                "Notes": "Demand x action share x qualification rate.",
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Export assumptions button
        st.divider()
        st.markdown("### 📤 Export")

        # Build assumptions dict with available data
        assumptions_export = {
            "generated_at": datetime.now().isoformat(),
            "summary_period": label,
            "shared": {
                "cvr": pct_str(cvr, 1),
                "business_hours_factor": pct_str(business_hours_factor, 0),
            },
            "estimates": {
                row["Channel"]: {
                    "traffic": row["Traffic/Actions (low-high)"],
                    "leads": row["Leads (low-high)"],
                    "notes": row["Notes"],
                }
                for row in rows
            },
        }

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Assumptions (JSON)",
                data=json.dumps(assumptions_export, indent=2),
                file_name=f"growth_estimate_assumptions_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )
        with col2:
            # CSV export of summary table using clean export
            download_table_button(
                summary_df,
                f"growth_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                "📥 Download Summary (CSV)",
                key="dl_summary"
            )

        st.info(
            "💡 **Reminder:** This tool outputs **estimates** based on uploaded data and selected assumptions. "
            "Use it to plan and communicate ranges — **not to guarantee outcomes**."
        )
