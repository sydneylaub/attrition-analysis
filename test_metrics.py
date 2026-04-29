import pandas as pd
import pytest

from load_data import clean_employee_data
from metrics import (
    attrition_by_department,
    attrition_by_overtime,
    attrition_rate,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    """
    6 employees across 3 departments.

    Leavers: employees 1, 3, 5  (attrition rate = 50%)

    Department breakdown:
      Sales       (3 employees): 2 leavers → 66.67%
      Engineering (2 employees): 1 leaver  → 50.00%
      HR          (1 employee):  0 leavers →  0.00%

    Overtime breakdown:
      Yes (3 employees): 3 leavers → 100.00%
      No  (3 employees): 0 leavers →   0.00%

    Income:
      Leavers avg: (4000 + 5000 + 7000) / 3 = 5333.33
      Stayers avg: (6000 + 8000 + 3000) / 3 = 5666.67

    Satisfaction:
      Level 1 (2 employees): 2 leavers → 100.00%
      Level 2 (2 employees): 0 leavers →   0.00%
      Level 3 (1 employee):  1 leaver  → 100.00%
      Level 4 (1 employee):  0 leavers →   0.00%
    """
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "Sales", "Engineering", "Engineering", "HR"],
            "age": [30, 35, 40, 25, 28, 33],
            "monthly_income": [4000.0, 6000.0, 5000.0, 8000.0, 7000.0, 3000.0],
            "job_satisfaction": [1, 2, 3, 4, 1, 2],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "travel_frequency": ["Frequent", "Rarely", "Occasional", "Rarely", "Frequent", "Rarely"],
            "years_at_company": [2, 5, 10, 3, 1, 7],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# ---------------------------------------------------------------------------
# clean_employee_data
# ---------------------------------------------------------------------------


def test_clean_raises_on_missing_columns():
    df = pd.DataFrame({"employee_id": [1], "department": ["Sales"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_fills_null_department(sample_df):
    sample_df.loc[0, "department"] = None
    result = clean_employee_data(sample_df)
    assert result.loc[0, "department"] == "Unknown"


def test_clean_strips_whitespace_from_department(sample_df):
    sample_df.loc[0, "department"] = "  Sales  "
    result = clean_employee_data(sample_df)
    assert result.loc[0, "department"] == "Sales"


def test_clean_fills_null_overtime(sample_df):
    sample_df.loc[0, "overtime"] = None
    result = clean_employee_data(sample_df)
    assert result.loc[0, "overtime"] == "No"


def test_clean_fills_null_job_satisfaction(sample_df):
    sample_df.loc[0, "job_satisfaction"] = None
    result = clean_employee_data(sample_df)
    assert result.loc[0, "job_satisfaction"] == 3


def test_clean_fills_null_income_with_median(sample_df):
    sample_df.loc[0, "monthly_income"] = None
    result = clean_employee_data(sample_df)
    # median of the remaining non-null values: [6000, 5000, 8000, 7000, 3000] → 6000.0
    assert result.loc[0, "monthly_income"] == 6000.0


def test_clean_normalizes_attrition_to_title_case(sample_df):
    sample_df["attrition"] = ["yes", "NO", "YES", "no", "yes", "NO"]
    result = clean_employee_data(sample_df)
    assert set(result["attrition"].unique()) == {"Yes", "No"}


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------


def test_attrition_rate(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_no_leavers(sample_df):
    sample_df["attrition"] = "No"
    assert attrition_rate(sample_df) == 0.0


def test_attrition_rate_all_leavers(sample_df):
    sample_df["attrition"] = "Yes"
    assert attrition_rate(sample_df) == 100.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------


def test_attrition_by_department_values(sample_df):
    result = attrition_by_department(sample_df)

    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 3
    assert sales["leavers"] == 2
    assert sales["attrition_rate"] == 66.67

    eng = result[result["department"] == "Engineering"].iloc[0]
    assert eng["employees"] == 2
    assert eng["leavers"] == 1
    assert eng["attrition_rate"] == 50.0

    hr = result[result["department"] == "HR"].iloc[0]
    assert hr["employees"] == 1
    assert hr["leavers"] == 0
    assert hr["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert set(result.columns) == {"department", "employees", "leavers", "attrition_rate"}


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------


def test_attrition_by_overtime_values(sample_df):
    result = attrition_by_overtime(sample_df)

    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    assert yes_row["employees"] == 3
    assert yes_row["leavers"] == 3
    assert yes_row["attrition_rate"] == 100.0

    no_row = result[result["overtime"] == "No"].iloc[0]
    assert no_row["employees"] == 3
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert set(result.columns) == {"overtime", "employees", "leavers", "attrition_rate"}


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------


def test_average_income_by_attrition(sample_df):
    result = average_income_by_attrition(sample_df)

    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    assert yes_avg == round((4000 + 5000 + 7000) / 3, 2)

    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert no_avg == round((6000 + 8000 + 3000) / 3, 2)


def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert set(result.columns) == {"attrition", "avg_monthly_income"}


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------


def test_satisfaction_summary_attrition_rate_per_group(sample_df):
    result = satisfaction_summary(sample_df)

    row1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert row1["total_employees"] == 2
    assert row1["leavers"] == 2
    assert row1["attrition_rate"] == 100.0

    row2 = result[result["job_satisfaction"] == 2].iloc[0]
    assert row2["total_employees"] == 2
    assert row2["leavers"] == 0
    assert row2["attrition_rate"] == 0.0

    row3 = result[result["job_satisfaction"] == 3].iloc[0]
    assert row3["total_employees"] == 1
    assert row3["leavers"] == 1
    assert row3["attrition_rate"] == 100.0

    row4 = result[result["job_satisfaction"] == 4].iloc[0]
    assert row4["total_employees"] == 1
    assert row4["leavers"] == 0
    assert row4["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)


def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert set(result.columns) == {"job_satisfaction", "total_employees", "leavers", "attrition_rate"}
