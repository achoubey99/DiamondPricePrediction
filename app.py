from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained encoders and model
with open('grid_search_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("column_transformer.pkl", "rb") as f:
    loaded_transformed_col = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/getData', methods=['POST'])
def getData():
    print('inside get data')
    try:
        # Check the content type
        if request.content_type == "application/json":
            form_data = request.json  # Parse JSON input
        else:
            form_data = request.form.to_dict()  # Parse form-encoded input

        print("Received Data:", form_data)

        # Convert input to DataFrame
        input_df = pd.DataFrame([form_data])  # Single record as a row
        print('after input df')

        # Define the columns
        obj_cols = ['cut', 'color', 'clarity']
        num_cols = ['depth', 'table', 'x', 'y', 'z', 'carat']

        # Preprocess categorical and numerical data
        categorical_data = [form_data[x] for x in obj_cols]

        #input_df[num_cols] = input_df[num_cols].apply(pd.to_numeric, errors='coerce')  # Handle non-numeric values
        numerical_data = [float(form_data[x]) for x in num_cols]

        categorical_data.extend(numerical_data)

        # Combine processed columns
        input_data = [categorical_data]
        

        input_df = pd.DataFrame(input_data, columns = obj_cols + num_cols)
        
        processed_input = loaded_transformed_col.transform(input_df)

        # Making prediction prediction
        prediction = model.predict(processed_input)
        
        # Return prediction
        result = "Predicted Price " + "= " + "$" + f"{round(float(prediction[0]),2)}"
        return render_template('index.html', prediction_text = result )
        

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
