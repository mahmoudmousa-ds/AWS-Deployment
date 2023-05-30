from flask import Flask, render_template, request
import joblib
import pandas as pd
import preprocessing as pre


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Page1.html')


@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("INSIDE PREDICT >>>>>>>>>>>>")
    input_data = [x for x in request.form.values()]
    preds = pre.predict_output(input_data)

    # Get the predicted values as a list
    is_closed = preds[0]
    if is_closed == 0:
        status = 'closed'
    else:
        status = preds[1] 

    print(is_closed, "   ", status)
    # Render the results template with the predicted values
    return render_template('submit.html', n= [is_closed, status])


if __name__ == '__main__':
    app.run(debug=True)
