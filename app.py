#!/usr/local/bin/flask/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, jsonify
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
os.chdir("/Users/JohnnyChiu/Desktop/檔案總管/2016專案/R/1227_素材分析/Advertising-Creative-Analysis/")

from creative_function import *
from initDataFrame import *

insightPath='../facebook_ad_report_ver3/insightData/age_gender_json_0217'
googlevisionPath='../facebook_ad_report_ver3/insightData/with-x-y-google-vision_0217.json'

mydata = initDataFrame(insightPath, googlevisionPath)
mydata=metric_generator(mydata)

gv_df_label = gv_data_reader(googlevisionPath)



@app.route('/best_ad/v1.0/<campaign_id>', methods=['GET'])
def best_ad(campaign_id):
    campaign_ids=campaign_id.split(',')
    campaign_ids = map(int, campaign_ids)
    a=mydata[mydata.campaign_id.isin(campaign_ids)]
    b=find_best_ad(a)
    c=b[['ad_id','ranking']].to_dict(orient='records')

    return jsonify(c)


@app.route('/recommendation/v1.0/<campaign_id>', methods=['GET'])
def recommendation(campaign_id):
    campaign_ids=campaign_id.split(',')
    campaign_ids = map(int, campaign_ids)
    
    campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
    campaign_data=metric_generator(campaign_data)
    
    feature_and_importance_female = find_feature_and_importance('gender','female',campaign_data,gv_df_label)
    feature_and_importance_male = find_feature_and_importance('gender','male',campaign_data,gv_df_label)
    feature_and_importance_unknown = find_feature_and_importance('gender','unknown',campaign_data,gv_df_label)
    
    feature_and_importance_1824 = find_feature_and_importance('age','18-24',campaign_data,gv_df_label)
    feature_and_importance_2534 = find_feature_and_importance('age','25-34',campaign_data,gv_df_label)
    feature_and_importance_3544 = find_feature_and_importance('age','35-44',campaign_data,gv_df_label)
    feature_and_importance_4554 = find_feature_and_importance('age','45-54',campaign_data,gv_df_label)
    feature_and_importance_5564 = find_feature_and_importance('age','55-64',campaign_data,gv_df_label)
    feature_and_importance_65 = find_feature_and_importance('age','65+',campaign_data,gv_df_label)
    
    df_list=[feature_and_importance_female,feature_and_importance_male,feature_and_importance_unknown,
    feature_and_importance_1824,feature_and_importance_2534,feature_and_importance_3544,
    feature_and_importance_4554,feature_and_importance_5564,feature_and_importance_65]
    
    result_df = pd.DataFrame(columns=['segment','value','recommend'])
    
    for index,df in enumerate(df_list):
        temp=df[['feature','value','percentage']]
        temp.set_index(temp.feature, inplace = True)
        del temp['feature']
        temp['priority']=range(1, temp.shape[0]+1)
        temp_dict=temp.to_dict(orient='index')
        
        result_df.loc[index]=pd.Series({'segment':df.columns[0],'value':df.iloc[0][0],'recommend':temp_dict})
    
    result_json=result_df.to_dict(orient='records')   

    return jsonify(result_json)
    
@app.route('/best_ad_by_segment/v1.0/<campaign_id>', methods=['GET'])
def best_ad_by_segment(campaign_id):
    campaign_ids=campaign_id.split(',')
    campaign_ids = map(int, campaign_ids)
        
    campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
    campaign_data=metric_generator(campaign_data)
    
    best_ad_gender=find_best_ad_by_segment(campaign_data,'gender')
    best_ad_age=find_best_ad_by_segment(campaign_data,'age')
    
    gender_df=pd.melt(best_ad_gender,id_vars=['ad_id','impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
    age_df = pd.melt(best_ad_age,id_vars=['ad_id','impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
    df_adid = pd.concat([gender_df, age_df])
    df_adid = df_adid[['feature','value','ad_id']]
    df_adid = df_adid.reset_index(drop=True)
    
    result_df = pd.DataFrame(columns=['segment','value','ad_id','feature'])
    
    for index,ad_id in enumerate(df_adid['ad_id']):
        ad_feature=find_ad_feature(campaign_data, gv_df_label, [ad_id])
        ad_feature=ad_feature[['feature','value']].T
        ad_feature.columns = ad_feature.iloc[0]
        ad_feature.drop(ad_feature.index[0:1], inplace=True)
        ad_feature_dict=ad_feature.to_dict(orient='records') #automatically deduplicate
        
        result_df.loc[index]=pd.Series({'segment':df_adid['feature'][index],'value':df_adid['value'][index],
                                  'ad_id':df_adid['ad_id'][index],'feature':ad_feature_dict})
    
    result_df['ad_id'] = map(int, result_df['ad_id'])
    result_json=result_df.to_dict(orient='records')
    return jsonify(result_json)

if __name__ == '__main__':
    app.run(debug=True)


