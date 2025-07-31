import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import numpy as np

model_data = load("artifacts/model_data.joblib")
model = model_data['model']
scaler = model_data['scaler']
cols_to_scale = model_data['cols_to_scale']
features = model_data['features']

def prepare_df(input_data):
    df = pd.DataFrame([input_data])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]
    return df

def calculate_credit_score(df,base_score=300,scale_length=600):
    y = np.dot(df.values,model.coef_.T)+model.intercept_
    # sigmoid function
    default_probability = 1/(1+np.exp(-y))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score)

    return default_probability.flatten()[0], int(credit_score[0]), rating


def predict(age,income,loan_amount,loan_tenure_months,
            Avg_DPD,deliquent_loan_months,credit_utilization_ratio,
            number_of_open_accounts,residence_type,loan_purpose,loan_type):

    input_data = {
        'age': age,
        'loan_to_income_ratio': loan_amount/income if income>0 else 0,
        'loan_tenure_months': loan_tenure_months,
        'Avg_DPD': Avg_DPD,
        'deliquent_loan_months': deliquent_loan_months,
        'credit_utilization_ratio': credit_utilization_ratio,
        'number_of_open_accounts': number_of_open_accounts,
        'loan_type_Unsecured': 1 if loan_type=='Unsecured' else 0,
        'residence_type_Owned': 1 if residence_type=='Owned' else 0,
        'residence_type_Rented': 1 if residence_type=='Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose=='Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose=='Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose=='Personal' else 0,

        # additional dummy features just for scaling purpose
        'sanction_amount':1,
        'processing_fee':1,
        'gst':1,
        'net_disbursement':1,
        'principal_outstanding':1,
        'bank_balance_at_application':1,
        'number_of_dependants':1,
        'years_at_current_address':1,
        'number_of_closed_accounts':1,
        'enquiry_count':1,
    }

    df = prepare_df(input_data)
    probability, credit_score, rating = calculate_credit_score(df)
    return probability,credit_score,rating
