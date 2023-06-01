import pandas as pd
import datetime
import pickle

def predict_output(b_data):
    
    return pipeline_out(preprocess_data(b_data))

def pipeline_out(X):
    output = []
    binary_pipeline = pickle.load(open("models_rfc.pkl", 'rb'))
    output.append(binary_pipeline.predict(X))

    multiclass_pipeline = pickle.load(open("models_gb_clf.pkl", 'rb'))
    output.append(multiclass_pipeline.predict(X))
    
    return output

def preprocess_data(raw_data):
    output={}

    #preprocess lng raw_data[0]
    output['lng'] = float(raw_data[12])

    #preprocess founded_at raw_data[0]
    var1 = datetime.datetime.strptime(raw_data[0], "%Y-%m-%d")
    var1 = var1.year
    output['founded_at']= float(var1)

    #Invested companies no preprocessing add to output raw_data[2]
    output['invested_companies'] = float(raw_data[4])

    # milestones no preprocessing needed
    output['milestones'] = float(raw_data[8])

    #preprocess updated_at raw_data[2]
    var2 = datetime.datetime.strptime(raw_data[2], "%Y-%m-%d")
    var2 = var2.year
    output['updated_at'] = float(var2)

    # ROI no preprocessing needed
    output['ROI'] = float(raw_data[10])

    # relationships no preprocessing needed
    output['relationships'] = float(raw_data[9])

    # investment rounds no preprocessing needed
    output['investment_rounds'] = float(raw_data[5])

    #preprocess created_at raw_data[1]
    var3 = datetime.datetime.strptime(raw_data[1], "%Y-%m-%d")
    var3 = var3.year
    output['created_at'] = float(var3)

    # latitude no preprocessing needed
    output['lat'] = float(raw_data[13])

    # funding rounds no preprocessing needed
    output['funding_rounds'] = float(raw_data[6])

    # funding total usd no preprocessing needed
    output['funding_total_usd'] = float(raw_data[7])

    # Remove that from the Fourm DONE!!!
    # Calculate that as per preprocessed in the final pipeline notebook
    output['funding_per_round']= float(output['funding_total_usd'] / output['funding_rounds'])

    # Calculate Active days per comppany
    # Remove it from Fourm DONE!!!
    # Calculate closed_at to use it as var4
    var4 = datetime.datetime.strptime(raw_data[3], "%Y-%m-%d")
    var4 = float(var4.year)
    output['active_days_of_company'] = float(var4 - output['founded_at']) 

    # closed_at change into #years (add it in its order as per X in Pipeline)
    output['closed_at'] = var4

    # Change that due to Label Encoder
    output['city'] = int(raw_data[15])

    # Change that due to Label Encoder
    output['category_code'] = int(raw_data[11])

    # Change that due to Label Encoder
    output['country_code'] = int(raw_data[16])

    # Change that due to label Encoder
    output['state_code'] = int(raw_data[17])

    # Change that due to Label Encoder (DONE)
    output['region'] = int(raw_data[14])
    
    return pd.DataFrame([output.values()], columns=output.keys())
