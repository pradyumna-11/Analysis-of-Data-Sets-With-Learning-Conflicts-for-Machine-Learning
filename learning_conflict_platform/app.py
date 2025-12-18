import streamlit as st
import pandas as pd

from core.validator import validate_dataset
from core.normalization import min_max_normalize
from core.cleaning import compute_total_conflict
from core.modeling import rmse
from core.visualization import conflict_histogram
from core.preprocessing import encode_categorical_features, handle_missing_values


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Learning Conflict Analysis Platform",
    layout="wide"
)

st.title("Learning Conflict Analysis Platform")
st.markdown(
    """
    Upload a **supervised regression dataset** to identify and remove learning conflicts.
    The platform validates datasets, preprocesses real-world issues,
    and generates performance reports automatically.
    """
)

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(file)
        st.session_state.processed = False

    df = st.session_state.df

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)

    validation_placeholder = st.empty()

    # =========================
    # MISSING VALUE HANDLING
    # =========================
    if df.isnull().any().any():
        st.warning("Missing values detected in the dataset.")

        missing_option = st.radio(
            "How would you like to handle missing values?",
            [
                "-- Select an option --",
                "Impute missing values (mean/mode) – recommended",
                "Drop rows with missing values"
            ],
            index=0
        )

        if missing_option == "-- Select an option --":
            st.stop()

        elif missing_option == "Impute missing values (mean/mode) – recommended":
            st.session_state.df = handle_missing_values(df, target)
            st.success("Missing values imputed successfully.")
            st.session_state.processed = False
            st.rerun()

        elif missing_option == "Drop rows with missing values":
            st.session_state.df = df.dropna()
            st.success("Rows with missing values removed.")
            st.session_state.processed = False
            st.rerun()

    df = st.session_state.df

    # =========================
    # VALIDATION & PREPROCESSING
    # =========================
    if not st.session_state.processed:
        valid, message = validate_dataset(df, target)

        if not valid:
            validation_placeholder.warning(message)

            if "Non-numeric features detected" in message:
                st.subheader("⚠️ Categorical Feature Handling")

                option = st.radio(
                    "How would you like to handle categorical features?",
                    [
                        "-- Select an option --",
                        "Drop categorical features (simpler, faster)",
                        "Encode categorical features (recommended)"
                    ],
                    index=0
                )

                if option == "-- Select an option --":
                    st.stop()

                elif option == "Drop categorical features (simpler, faster)":
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    if target not in numeric_cols:
                        st.error("Target column would be removed. Encoding is required.")
                        st.stop()

                    st.session_state.df = df[numeric_cols]
                    st.session_state.processed = True
                    st.rerun()

                elif option == "Encode categorical features (recommended)":
                    st.session_state.df = encode_categorical_features(df, target)
                    st.session_state.processed = True
                    st.rerun()

            else:
                st.stop()

    # =========================
    # FINAL VALIDATION
    # =========================
    df = st.session_state.df
    valid, message = validate_dataset(df, target)

    if not valid:
        st.error(message)
        st.stop()

    validation_placeholder.empty()
    st.success(message)

    features = [c for c in df.columns if c != target]

    st.divider()

    # =========================
    # RUN ANALYSIS
    # =========================
    if st.button("▶ Run Learning Conflict Analysis"):
        with st.spinner("Running conflict analysis... Please wait."):

            df_norm = min_max_normalize(df, features, target)
            df_conflict = compute_total_conflict(df_norm, features, target)

            rmse_before = rmse(df_conflict, features, target)

            remove_n = min(20, len(df_conflict) // 10)
            cleaned = (
                df_conflict
                .sort_values("total_conflict", ascending=False)
                .iloc[remove_n:]
            )

            rmse_after = rmse(cleaned, features, target)

        st.subheader("📊 Results Summary")

        col1, col2 = st.columns(2)
        col1.metric("RMSE Before Cleaning", round(rmse_before, 4))
        col2.metric("RMSE After Cleaning", round(rmse_after, 4))

        st.subheader("📈 Conflict Distribution")
        st.pyplot(conflict_histogram(df_conflict))

        st.subheader("⬇ Download Outputs")

        st.download_button(
            "Download Cleaned Dataset",
            cleaned.to_csv(index=False),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download Conflict-Scored Dataset",
            df_conflict.to_csv(index=False),
            file_name="conflict_scored_dataset.csv",
            mime="text/csv"
        )

        st.success("Analysis completed successfully 🎉")
