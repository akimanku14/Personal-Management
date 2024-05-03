import numpy as np
import torch
import torch.nn as nn
import joblib
from flask import *
import os
from werkzeug.utils import secure_filename
import pandas as pd


app = Flask(__name__)

##################################
import colored 

hr_df = pd.read_csv('./HR.csv')
hr_df.columns = [col.lower() for col in hr_df.columns]
if 'sales' in hr_df.columns.tolist():
    hr_df.rename(columns={'sales': 'department'}, inplace=True)

if 'promotion_last_years' in hr_df.columns.tolist():
    hr_df.rename(columns={'promotion_last_years': 'promotion_last_years'}, inplace=True)

if 'average_monthly_hours' in hr_df.columns.tolist():
    hr_df.rename(columns={'average_monthly_hours': 'average_monthly_hours'}, inplace=True)

scaler = joblib.load('scaler.pkl')

valid_columns_list = hr_df.columns.tolist()
valid_columns_list.remove('left')   # These are our valid columns list 

# Setup the device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"  # Activate if you want to use cpu only.


class ClassificationModel(nn.Module):
    def __init__(self, n_features):
        super(ClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def scale(df, scaler=scaler):
    df = scaler.transform(df)
    df = torch.tensor(df.astype(np.float32)).to(device)
    return df

def columns_compare(input_csv, valid_columns = valid_columns_list):
        # Load the CSV files into DataFrames  df1 = input dataframe that the user uploads 
        df1 = pd.read_csv(input_csv)
        # Get the column sets for both DataFrames
        columns_set1 = set(df1.columns)   #columns of our  uploaded csv file
        columns_set2 = set(valid_columns) #columns that we had before of our hr csv file 
        # Compare the column sets
        if columns_set1 != columns_set2: 
            return True
        return False 
        
def validate_columns(df, valid_columns_list=valid_columns_list):
    # Convert DataFrame columns and the valid list to sets for comparison
    current_columns_set = set(df.columns)
    valid_columns_set = set(valid_columns_list)

    # Check if the sets are exactly the same
    if current_columns_set != valid_columns_set:
        # Find columns that are in the DataFrame but not in the valid list (unexpected)
        unexpected_columns = current_columns_set - valid_columns_set
        # Find columns that are in the valid list but not in the DataFrame (missing)
        missing_columns = valid_columns_set - current_columns_set

        # Create error message with details about unexpected and missing columns
        error_message = "DataFrame column mismatch:"
        if unexpected_columns:
            error_message += f" Unexpected columns: {unexpected_columns}."
        if missing_columns:
            error_message += f" Missing columns: {missing_columns}."

        # Raise an error with the detailed message
        raise ValueError(error_message)
def align_dfs_and_check_dtypes(df1, df2):
   
    # Check if the input DataFrame has the same columns
    if not df1.columns.equals(df2.columns):
        validate_columns(df1, valid_columns_list=valid_columns_list)

    # Check data types of common columns
    common_columns = df1.columns.intersection(df2.columns)
    dtype_mismatches = []

    for column in common_columns:
        df1[column] = df1[column].astype(df2[column].dtype)
        if df1[column].dtype != df2[column].dtype:
            dtype_mismatches.append(
                f"\texpected dtype of {df2[column].dtype} for column '{column}' but instead got {df1[column].dtype}."
            )

    # If there are any mismatches, raise a ValueError
    if dtype_mismatches:
        raise ValueError("Data type mismatch found: \n" + "\n".join(dtype_mismatches))
    return df1

#######Used to check if there are any missing values
def align(input_csv,hr_df): 
    df1 = pd.read_csv(input_csv)
    df2 = hr_df
    if not df1.columns.equals(df2.columns):
        validate_columns(df1, valid_columns_list=valid_columns_list)

    # Check data types of common columns
    common_columns = df1.columns.intersection(df2.columns)
    dtype_mismatches = []

    for column in common_columns:
        df1[column] = df1[column].astype(df2[column].dtype)
        if df1[column].dtype != df2[column].dtype:
            dtype_mismatches.append(
                f"\texpected dtype of {df2[column].dtype} for column '{column}' but instead got {df1[column].dtype}."
            )
    # If there are any mismatches, raise a ValueError
    if dtype_mismatches:
        return True 
    return False 


def salary_column(df1, df2=hr_df):
    df2 = df2.drop(columns='left')
    #print("df2", df2,"valid_columns",valid_columns_list)
    validate_columns(df1, valid_columns_list)

    # Check if the salary column has any invalid values
    valid_salaries = df2["salary"].unique().tolist() # List of valid department names
    invalid_mask = ~df1['salary'].isin(valid_salaries) # Create a mask where True indicates invalid values
    invalid_indices = df1[invalid_mask].index.tolist() # Display invalid rows
    if invalid_indices:
        return True 
    else:
        salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df1['salary'] = df1['salary'].map(salary_mapping)
        df2['salary'] = df2['salary'].map(salary_mapping)
        return False 

def preprocess_validate_data(df1, df2=hr_df):
    df2 = df2.drop(columns='left')
    #print("df2", df2,"valid_columns",valid_columns_list)
    validate_columns(df1, valid_columns_list )

    # Check if the salary column has any invalid values
    valid_salaries = df2["salary"].unique().tolist() # List of valid department names
    invalid_mask = ~df1['salary'].isin(valid_salaries) # Create a mask where True indicates invalid values
    invalid_indices = df1[invalid_mask].index.tolist() # Display invalid rows
    if invalid_indices:
        raise ValueError(f"{colored('salary', 'blue')} column can only take values: {colored(valid_salaries, 'blue')} \n Please check the salary column of the invalid row indices in the uploaded csv! \n {colored('invalid row indices:', 'blue')} {invalid_indices}")

    else:
        salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df1['salary'] = df1['salary'].map(salary_mapping)
        df2['salary'] = df2['salary'].map(salary_mapping)

    # Check if the department column has any invalid values
    valid_departments = df2["department"].unique().tolist() # List of valid department names
    invalid_mask = ~df1['department'].isin(valid_departments) # Create a mask where True indicates invalid values
    invalid_indices = df1[invalid_mask].index.tolist() # Display invalid rows
    if invalid_indices:
        raise ValueError(f"{colored('department', 'blue')} column can only take values: {colored(valid_departments, 'blue')} \n Please check the salary column of the invalid row indices in the uploaded csv! \n {colored('invalid row indices:', 'blue')} {invalid_indices}")

    # Check if the input DataFrame has the same columns and dtypes.
    df1 = align_dfs_and_check_dtypes(df1, df2)

    df1 = pd.get_dummies(df1, columns=['department'])
    df2 = pd.get_dummies(df2, columns=['department'])

    # Reindex df1 to match the columns of df2, filling new columns with 0
    df1 = df1.reindex(columns=df2.columns, fill_value=0)
    df1 = df1[df2.columns]

    # Again check if the input DataFrame has the same columns and dtypes.
    df1 = align_dfs_and_check_dtypes(df1, df2)

    # Check numeric columns for value ranges #This we can't use because if we have to use that user is restricted to add values less than 0.36
    '''for column in df1.select_dtypes(include=['float64', 'int64']).columns:
        if column in df2.columns:
            min_val = df2[column].min()    
            max_val = df2[column].max()
            if not df1[column].between(min_val, max_val).all():
                raise ValueError(f"Values in column '{column}' are out of the range {min_val} to {max_val}.")'''

    # Check object type columns for matching unique values
    for column in df1.select_dtypes(include=['object']).columns:
        if column in df2.columns:
            unique_values_hr = set(df2[column].unique())
            unique_values_input = set(df1[column].unique())
            if unique_values_hr != unique_values_input:
                print(f"unique_values_hr: {unique_values_hr} \t unique_values_input: {unique_values_input}")
                raise ValueError(f"Values in column '{column}' do not match the unique values of the HR DataFrame.")

    # Last check before proceeding
    if not (len(df1.columns)==len(df2.columns)):
        raise ValueError("length mismatch")

    df1 = scaler.transform(df1)
    df1 = torch.tensor(df1, dtype=torch.float32).to(device)
    return df1

#A problem here we have 
def predict_csv_file(input_csv_path="input.csv", model_path='model_classification.pth'):
    input_df = pd.read_csv(input_csv_path)
    # Validate the input DataFrame
    input_df_preprocessed = preprocess_validate_data(input_df, hr_df)
    model = ClassificationModel(input_df_preprocessed.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_df_preprocessed)
        #predicted_prob = output.squeeze().item()
        pred = output
        #print(f"Leaving probability of %{100 * predicted_prob:.0f}")
        input_df['left'] = pred
    return input_df

def predict_csv(input_csv_path="input.csv", model_path='model_classification.pth'):

    input_df = pd.read_csv(input_csv_path)

    # Validate the input DataFrame
    input_df_preprocessed = preprocess_validate_data(input_df, hr_df)

    # Load the model
    model = ClassificationModel(input_df_preprocessed.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_df_preprocessed)
        predicted_prob = output.squeeze().item()

        print(f"Leaving probability of %{100 * predicted_prob:.0f}")

    return render_template('prediction.html',pred="Leaving probability of {:.0f}".format(100 * predicted_prob) +"%")
        # TODO: line 193 does not return a valid response,
        #  however we can see the predicted probabilities
        #  on the terminal with line 191 print. We need to fix it.

#uploading csv file
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'


# Define route for the home page
@app.route('/')
def home():
    return render_template('prediction.html')


# Define route for prediction 
@app.route('/predict', methods=['POST'])
def predict():
    # Check if all required fields are filled
    required_fields = ['satisfaction_level', 'last_evaluation', 'average_monthly_hours', 'number_project',
                       'time_spend_company', 'work_accident', 'promotion_last_years', 'department', 'salary']
    #for field in required_fields:
        #if field not in request.form or not request.form[field]:
            #return render_template('prediction.html', error_empty_feilds='Please fill in all required fields.')

    final_input = pd.DataFrame(columns=valid_columns_list)
    # Get the input values from the form
    satisfaction_level_str = request.form['satisfaction_level']
    try:
        satisfaction_level = float(satisfaction_level_str)
        if not 0 <= satisfaction_level <= 1:
            raise ValueError("Satisfaction level must be between 0 and 1.")
        if not satisfaction_level_str.replace('.', '', 1).isdigit():
            raise ValueError("Satisfaction level must be a numeric value.")
    except ValueError: 
            return render_template('prediction.html',
               error_satisfaction_level='Please enter a valid numeric satisfaction level between 0 and 1.')
    final_input.at[0, 'satisfaction_level'] = satisfaction_level

    last_evaluation_str = request.form['last_evaluation']
    try:
        last_evaluation = float(last_evaluation_str)
        if not 0 <= last_evaluation <= 1:
            raise ValueError("Last Evaluation must be between 0 and 1.")
        if not satisfaction_level_str.replace('.', '', 1).isdigit():
            raise ValueError("Last Evaluation must be a numeric value.")
    except ValueError:
        return render_template('prediction.html',
                               error_last_evaluation='Please enter a valid numeric Last Evaluation between 0 and 1.')
    final_input.at[0, 'last_evaluation'] = last_evaluation

    number_project_str = request.form['number_project']
    try:
        number_project = float(number_project_str)
        if not number_project == int(number_project):
            raise ValueError("Please enter a whole number as number of projects")
        if not number_project >= 0:
            raise ValueError("Enter a non-negative integer as project number")
    except ValueError:
        return render_template('prediction.html',
                               error_number_project='Please enter a non-negative integer for the number of project.')
    final_input.at[0, 'number_project'] = number_project

    average_monthly_hours_str = request.form['average_monthly_hours']
    try:
        average_monthly_hours = float(average_monthly_hours_str)
        if not average_monthly_hours >= 0:
            raise ValueError("Please enter a value which is non-negative for Average Monthly Hours.")
    except ValueError:
        return render_template('prediction.html',
                               error_average_monthly_hours='Please enter a value which is non-negative for Average Monthly Hours.')
    final_input.at[0, 'average_monthly_hours'] = average_monthly_hours

    time_spend_company_str = request.form['time_spend_company']
    try:
        time_spend_company = float(time_spend_company_str)
        if not time_spend_company >= 0:
            raise ValueError("Time Spend in Company must be non-negative number")
    except ValueError:
        return render_template('prediction.html',
                               error_time_spend_company='Time Spend in Company must be non-negative number.')
    final_input.at[0, 'time_spend_company'] = time_spend_company

    work_accident_str = request.form['work_accident']
    try:
        work_accident = float(work_accident_str)
        if not (work_accident == 0 or work_accident == 1):
            raise ValueError('Work accident must be binary , either 1 or 0')
    except ValueError:
        return render_template('prediction.html',
                               error_work_accident='Please enter either 1 if you had a work accident or o if you did not have any.')
    final_input.at[0, 'work_accident'] = work_accident

    promotion_last_years_str = request.form['promotion_last_years']
    try:
        promotion_last_years = float(promotion_last_years_str)
        if not (promotion_last_years == 0 or promotion_last_years == 1):
            raise ValueError('Promotion must be binary , either 1 or 0')
    except ValueError:
        return render_template('prediction.html',
                               error_promotion_last_years='Please enter either 0 or 1 for the field Promotion in the last years.')
    final_input.at[0, 'promotion_last_years'] = promotion_last_years

    department = request.form['department']
    final_input.at[0, 'department'] = str(department)

    salary = request.form['salary']
    final_input.at[0, 'salary'] = str(salary)

    # convert final_input into csv
    final_input.to_csv("temp_input.csv", index=False)

    # Call this function to make prediction on csv.
    return predict_csv(input_csv_path="temp_input.csv", model_path='model_classification.pth')


@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        f = request.files.get('file')

        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
                                                          data_filename)

        return render_template('prediction2.html')
    return render_template("prediction.html")


@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None) 
    if columns_compare(data_file_path, valid_columns=valid_columns_list)==False or (align(data_file_path,hr_df) )==True:
        uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
        output_df = predict_csv_file(input_csv_path=data_file_path, model_path='model_classification.pth')   
        uploaded_df_html = output_df.to_html()
        return render_template('show_csv_data.html', data_var=uploaded_df_html) 
    else: 
        return render_template("prediction2.html",error= "Please enter a valid csv file. Download the template attached and fulfil the requirements")
     


@app.route('/download')
def download_file():
    p = "/Users/akishakaurmanku/Desktop/Personal management/version1/staticFiles/uploads/final2.csv"
    return send_file(p, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
