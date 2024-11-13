import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

# Load and prepare data
df = pd.read_csv(r"C:\Projects\ml_project_final\data\APY.csv")
new_df = df[df.iloc[:,0] == 'Karnataka']
new_df = new_df.applymap(lambda x: x.strip() if type(x) == str else float(str(x).strip()))
new_df.columns = [i.strip() for i in new_df.columns]
Q1 = new_df.Yield.quantile(0.25)
Q3 = new_df.Yield.quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5 * IQR                                          
high = Q3 + 1.5 * IQR
out_df = new_df[(new_df.Yield < high)]
districts = pd.get_dummies(out_df.District)
crops = pd.get_dummies(out_df.Crop)
seasons = pd.get_dummies(out_df.Season)
X = out_df[['Crop_Year', 'Area']]
X = pd.concat([X, districts, crops, seasons], axis=1)
Y = out_df['Yield']
x, y = map(lambda i: i.to_numpy(), [X, Y])
x_vals, x_cats = x[:, :2], x[:, 2:]
scaler = StandardScaler()
scaled_x_vals = scaler.fit_transform(x_vals)
scaled_x = np.concatenate([scaled_x_vals, x_cats], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)
scaled_x_train, scaled_x_test, y_train, y_test = train_test_split(scaled_x, y, random_state=42, test_size=0.25)

svr = SVR(kernel='rbf', C=10)
svr.fit(scaled_x_train, y_train)
y_pred = svr.predict(scaled_x_test)
mae = mean_absolute_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)

# Streamlit app
st.title("Future Crop Yield Prediction")

st.write("Please input the values to get the crop yield prediction for future years")

# Use Streamlit's form to group inputs and the button together
with st.form(key='prediction_form'):
    crop_year = st.number_input("Crop Year", min_value=int(new_df['Crop_Year'].min()), max_value=2050, value=2024)
    area = st.number_input("Area", min_value=float(new_df['Area'].min()), max_value=float(new_df['Area'].max()))

    district = st.selectbox("District", options=districts.columns)
    crop = st.selectbox("Crop", options=crops.columns)
    season = st.selectbox("Season", options=seasons.columns)

    submit_button = st.form_submit_button(label='Predict')

# Handle form submission
if submit_button:
    # Convert inputs to the required format
    input_data = np.zeros(2 + len(districts.columns) + len(crops.columns) + len(seasons.columns))
    input_data[0] = crop_year
    input_data[1] = area
    input_data[2 + districts.columns.get_loc(district)] = 1
    input_data[2 + len(districts.columns) + crops.columns.get_loc(crop)] = 1
    input_data[2 + len(districts.columns) + len(crops.columns) + seasons.columns.get_loc(season)] = 1

    # Scale the input data
    scaled_input_data = np.concatenate([scaler.transform([input_data[:2]]).flatten(), input_data[2:]])

    prediction = svr.predict([scaled_input_data])[0]
    st.write(f"Predicted Crop Yield for {crop_year}: {prediction:.2f}")
