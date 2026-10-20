
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Ames House Price Prediction")
st.write("Estimate a house sale price using a trained Gradient Boosting model.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"
DATA_PATH = BASE_DIR / "data" / "train.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    reference_data = load_reference_data()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load the model or training data: {e}")
    st.stop()


# Use the same feature names and order as training.
feature_names = list(model.feature_names_in_)
X_reference = reference_data.drop(columns=["SalePrice", "Id"], errors="ignore")

# Match the reference columns to the model's expected input columns.
# The notebook's training features should be the source of truth.
if set(feature_names).issubset(X_reference.columns):
    X_reference = X_reference[feature_names]
else:
    st.error(
        "Training CSV columns do not match the model's expected features. "
        "We need to check the training preprocessing before proceeding."
    )
    st.stop()


# Create sensible default values for every feature
default_values = {}

for column in feature_names:
    if pd.api.types.is_numeric_dtype(X_reference[column]):
        # Use the median for numerical features
        default_values[column] = X_reference[column].median()
    else:
        # Use the most common value for categorical features
        common_values = X_reference[column].mode(dropna=True)

        if not common_values.empty:
            default_values[column] = common_values.iloc[0]
        else:
            default_values[column] = "None"

# Create a one-row DataFrame with all model features
default_input = pd.DataFrame(
    [default_values],
    columns=feature_names
)


def apply_absence_rules(input_df):
    """Set related features consistently when a house lacks a feature."""

    # No garage
    if "GarageCars" in input_df.columns and input_df["GarageCars"].iloc[0] == 0:
        for column in [
            "GarageArea", "GarageYrBlt", "GarageFinish",
            "GarageQual", "GarageCond", "GarageType"
        ]:
            if column in input_df.columns:
                if pd.api.types.is_numeric_dtype(input_df[column]):
                    input_df.loc[input_df.index[0], column] = 0
                else:
                    input_df.loc[input_df.index[0], column] = "None"

        # Garage exists: ensure garage area is sensible
    if "GarageCars" in input_df.columns and input_df["GarageCars"].iloc[0] > 0:
        if "GarageArea" in input_df.columns and input_df["GarageArea"].iloc[0] == 0:
            input_df.loc[input_df.index[0], "GarageArea"] = 200

        for column in ["GarageFinish", "GarageQual", "GarageCond", "GarageType"]:
            if column in input_df.columns and input_df[column].iloc[0] == "None":
                common_values = X_reference[column].mode(dropna=True)
                if not common_values.empty:
                    input_df.loc[input_df.index[0], column] = common_values.iloc[0]

    # No basement
    if "TotalBsmtSF" in input_df.columns and input_df["TotalBsmtSF"].iloc[0] == 0:
        for column in [
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
            "BsmtFullBath", "BsmtHalfBath",
            "BsmtFullBath", "BsmtHalfBath",
            "BsmtQual", "BsmtCond", "BsmtExposure",
            "BsmtFinType1", "BsmtFinType2"
        ]:
            if column in input_df.columns:
                if pd.api.types.is_numeric_dtype(input_df[column]):
                    input_df.loc[input_df.index[0], column] = 0
                else:
                    input_df.loc[input_df.index[0], column] = "None"

    # No fireplace
    if "Fireplaces" in input_df.columns and input_df["Fireplaces"].iloc[0] == 0:
        if "FireplaceQu" in input_df.columns:
            input_df.loc[input_df.index[0], "FireplaceQu"] = "None"

    
    # Basement exists: replace missing categorical defaults
    if "TotalBsmtSF" in input_df.columns and input_df["TotalBsmtSF"].iloc[0] > 0:
        for column in [
            "BsmtQual", "BsmtCond", "BsmtExposure",
            "BsmtFinType1", "BsmtFinType2"
        ]:
            if column in input_df.columns and input_df[column].iloc[0] == "None":
                common_values = X_reference[column].mode(dropna=True)
                if not common_values.empty:
                    input_df.loc[input_df.index[0], column] = common_values.iloc[0]

    # Fireplace exists: replace missing quality default
    if "Fireplaces" in input_df.columns and input_df["Fireplaces"].iloc[0] > 0:
        if "FireplaceQu" in input_df.columns and input_df["FireplaceQu"].iloc[0] == "None":
            common_values = X_reference["FireplaceQu"].mode(dropna=True)
            if not common_values.empty:
                input_df.loc[input_df.index[0], "FireplaceQu"] = common_values.iloc[0]


    return input_df


st.subheader("Enter House Details")

# Get valid categorical options from the training data.
neighborhoods = sorted(
    X_reference["Neighborhood"].dropna().unique().tolist()
)
house_styles = sorted(
    X_reference["HouseStyle"].dropna().unique().tolist()
)
kitchen_qualities = ["Ex", "Gd", "TA", "Fa", "Po"]
exterior_qualities = ["Ex", "Gd", "TA", "Fa", "Po"]

with st.form("house_details"):
    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider(
            "Overall Quality (1–10)", 1, 10, 6
        )
        overall_cond = st.slider(
            "Overall Condition (1–10)",
            min_value=1,
            max_value=10,
            value=5
        )
        living_area = st.number_input(
            "Above-ground Living Area (sq ft)",
            100, 10000, 1500, step=50
        )
        garage_cars = st.number_input(
            "Garage Capacity (cars)", 0, 5, 2
        )
        year_built = st.number_input(
            "Year Built", 1872, 2010, 2000
        )
        year_remod = st.number_input(
            "Year Remodeled", 1950, 2010, 2000
        )
        bedrooms = st.number_input(
            "Bedrooms", 0, 8, 3
        )
        full_baths = st.number_input(
            "Full Bathrooms", 0, 4, 2
        )
        garage_area = st.number_input(
            "Garage Area (sq ft)", 0, 1500, 400, step=25
        )
        fireplaces = st.number_input(
            "Fireplaces", 0, 4, 1
        )
        wood_deck = st.number_input(
            "Wood Deck Area (sq ft)", 0, 1000, 100, step=25
        )

    with col2:
        basement_area = st.number_input(
            "Total Basement Area (sq ft)",
            0, 5000, 800, step=50
        )
        first_floor = st.number_input(
            "First-floor Area (sq ft)",
            0, 4000, 1000, step=50
        )
        second_floor = st.number_input(
            "Second-floor Area (sq ft)",
            0, 4000, 500, step=50
        )
        lot_area = st.number_input(
            "Lot Area (sq ft)", 500, 100000, 8000, step=100
        )
        lot_frontage = st.number_input(
            "Lot Frontage (feet)",
            min_value=0,
            max_value=500,
            value=60,
            step=5
        )
        total_rooms = st.number_input(
            "Total Rooms", 2, 14, 6
        )
        open_porch = st.number_input(
            "Open Porch Area (sq ft)", 0, 500, 50, step=10
        )
        kitchen_qual = st.selectbox(
            "Kitchen Quality", kitchen_qualities, index=2
        )
        exterior_qual = st.selectbox(
            "Exterior Quality", exterior_qualities, index=2
        )
        neighborhood = st.selectbox(
            "Neighborhood", neighborhoods
        )
        house_style = st.selectbox(
            "House Style", house_styles
        )

    predict = st.form_submit_button("Predict House Price")


if predict:
    if first_floor + second_floor > living_area:
        st.warning(
            "The combined first-floor and second-floor areas "
            "should not exceed the above-ground living area."
        )

    elif garage_cars > 0 and garage_area == 0:
        st.warning(
            "Garage capacity is greater than zero, "
            "but garage area is zero. Please enter a valid garage area."
        )

    elif year_remod < year_built:
        st.warning(
            "The remodeling year cannot be earlier than the construction year."
        )

    else:
        # Keep your existing prediction code inside this else block
        input_df = default_input.copy()

        updates = {
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "GrLivArea": living_area,
            "GarageCars": garage_cars,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "TotalBsmtSF": basement_area,
            "1stFlrSF": first_floor,
            "2ndFlrSF": second_floor,
            "LotArea": lot_area,
            "LotFrontage": lot_frontage,
            "BedroomAbvGr": bedrooms,
            "FullBath": full_baths,
            "GarageArea": garage_area,
            "Fireplaces": fireplaces,
            "WoodDeckSF": wood_deck,
            "TotRmsAbvGrd": total_rooms,
            "OpenPorchSF": open_porch,
            "KitchenQual": kitchen_qual,
            "ExterQual": exterior_qual,
            "Neighborhood": neighborhood,
            "HouseStyle": house_style,
    }

        for feature, value in updates.items():
            if feature in input_df.columns:
                input_df.loc[input_df.index[0], feature] = value

        try:
            input_df = apply_absence_rules(input_df)
            predicted_price = model.predict(input_df)[0]
            st.subheader("House Details Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            summary_col1.metric("Overall Quality", overall_qual)
            summary_col2.metric("Living Area", f"{living_area:,} sq ft")
            summary_col3.metric("Garage Capacity", f"{garage_cars} cars")

            st.subheader("Estimated Sale Price")
            st.metric("Predicted price", f"${predicted_price:,.2f}")
            st.caption(
                "This is a machine learning estimate, not a guaranteed sale price. "
                "The prediction depends on the features entered and default values "
                "used for characteristics not entered by the user."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

