from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained encoders and model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

with open('ordinal_encoder.pkl', 'rb') as file:
    ordinal_encoder = pickle.load(file)

with open("column_transformer.pkl", "rb") as f:
    loaded_transformed_col = pickle.load(f)


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

        # Validate input
        for col in obj_cols + num_cols:
            if col not in input_df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # Preprocess categorical and numerical data
        categorical_data = [form_data[x] for x in obj_cols]

        print('categorical data',categorical_data)
        #input_df[num_cols] = input_df[num_cols].apply(pd.to_numeric, errors='coerce')  # Handle non-numeric values
        numerical_data = [float(form_data[x]) for x in num_cols]
        
        print('numerical data',numerical_data)

        categorical_data.extend(numerical_data)

        print('input list', categorical_data)

        # Combine processed columns
        input_data = [categorical_data]
        print('input array', input_data)

        input_df = pd.DataFrame(input_data, columns = obj_cols + num_cols)
        
        if len(input_data[0]) != 9:
            raise ValueError("Input data must have 9 features.", input_df)
        print(input_df)
        processed_input = loaded_transformed_col.transform(input_df)
        # processed_input = pd.DataFrame(
        #     data=np.hstack([categorical_transformed, numerical_transformed]),
        #     columns=obj_cols + num_cols
        # )

        print('processing done')

        # Make prediction
        prediction = model.predict(processed_input)
        if prediction.size == 0:
            raise ValueError("Prediction is empty")

        print('prediction done')
        
        # Return prediction
        return jsonify({"predicted_price": float(prediction[0])}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
