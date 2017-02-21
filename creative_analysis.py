os.chdir("/Users/JohnnyChiu/Desktop/檔案總管/2016專案/R/1227_素材分析/Advertising-Creative-Analysis/")
os.getcwd()

import pandas as pd
import os
import numpy as np
from creative_function import *
from initDataFrame import *



######################
##### 讀檔案進來 ##### 
######################

mydata = initDataFrame('../facebook_ad_report_ver3/insightData/age_gender_json_0217',
                '../facebook_ad_report_ver3/insightData/with-x-y-google-vision_0217.json')
mydata['account_name'] = mydata['account_name'].str.strip()



######################################################################
###########################
##### etungo analysis ##### 
###########################

etungo_df=mydata[mydata.account_id == 1618494121752703]
etungo_df=brand_column_generator(etungo_df,'大同|etungo')
#pd.isnull(etungo_df).any(axis=0)
etungo_df=metric_generator(etungo_df)




########################################
#####find the best and worst ad ########
########################################

best_ad=find_best_ad(etungo_df)

#temp=best_ad[['ad_id','ranking']].to_dict(orient='records')



####################################################
#####find the best ad and feature for each group ###
####################################################

best_ad_gender=find_best_ad_by_segment(etungo_df,'gender')
best_ad_age=find_best_ad_by_segment(etungo_df,'age')

#find_ad_feature(etungo_df, [6055843154405])
#best_ad_gender_feature=find_ad_feature(etungo_df, best_ad_gender.ad_id.unique().tolist())
#best_ad_age_feature=find_ad_feature(etungo_df, best_ad_age.ad_id.unique().tolist())


#################################
#####filter analyzing columns ###
#################################
### ad related: title_brand, sub_title_brand, ad_content_brand, title_length_interval, sub_title_length_interval, ad_content_length_interval, call_to_action
### ad performance related: age, gender, impression, click
### google vision related: faceCount, majorColor, textInImage, logoInImage, adult, medical, spoof, violence, image category

etungo_df_analysis=column_selector(etungo_df)



#############################################
#####find the best feature for each group ###
#############################################
#etungo_df_analysis=metric_generator(etungo_df_analysis)
#final_feature=find_feature(etungo_df_analysis,'gender')

#final_feature.to_csv('final_feature.csv', sep=',', encoding='utf-8')
#etungo_df_analysis.head(10) 
#test=final_feature[final_feature.variable=='title_brand']
#test.head(20)

###################################################
#####find the feature importance for each group ###
###################################################

#importance_df=find_importance(etungo_df_analysis)




    
#######################################################
#####find the feature and importance for each group ###
#######################################################
#print_full(feature_and_importance_65)
#a=find_ad_feature(etungo_df, [6050712185805])
#print_full(a)

feature_and_importance_female = find_feature_and_importance(etungo_df_analysis,'gender','female')
feature_and_importance_male = find_feature_and_importance(etungo_df_analysis,'gender','male')
feature_and_importance_unknown = find_feature_and_importance(etungo_df_analysis,'gender','unknown')

feature_and_importance_1824 = find_feature_and_importance(etungo_df_analysis,'age','18-24')
feature_and_importance_2534 = find_feature_and_importance(etungo_df_analysis,'age','25-34')
feature_and_importance_3544 = find_feature_and_importance(etungo_df_analysis,'age','35-44')
feature_and_importance_4554 = find_feature_and_importance(etungo_df_analysis,'age','45-54')
feature_and_importance_5564 = find_feature_and_importance(etungo_df_analysis,'age','55-64')
feature_and_importance_65 = find_feature_and_importance(etungo_df_analysis,'age','65+')

campaign_id='6060850543605,6059224129805,6059889803605,6059892762805,6059893654805,6057193158805,6055407354005,6053312447405,6051265864405,6051266212805,6049374949605,6035884102605,6035882865005,6034201296805,6034192331405,6033466101805,6032523746205,6033118372005,6033118372205,6032205731205,6032523746005,6032205730605,6031518441405,6031616546205,6031518442005,6031518441605,6031518441205,6031518441005,6032906211805,6032906682605,6032907243605,6032261374205,6032627603405,6032260651005,6032626832005,6032321919605,6032411410005,6032321128805,6032200009405,6031742821405,6031796276605,6031722773205'
campaign_ids=campaign_id.split(',')
campaign_ids = map(int, campaign_ids)
    
campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
campaign_data=metric_generator(campaign_data)
campaign_data_analysis=column_selector(campaign_data)
    
feature_and_importance_female = find_feature_and_importance(campaign_data_analysis,'gender','female')
feature_and_importance_male = find_feature_and_importance(campaign_data_analysis,'gender','male')
feature_and_importance_unknown = find_feature_and_importance(campaign_data_analysis,'gender','unknown')

feature_and_importance_1824 = find_feature_and_importance(campaign_data_analysis,'age','18-24')
feature_and_importance_2534 = find_feature_and_importance(campaign_data_analysis,'age','25-34')
feature_and_importance_3544 = find_feature_and_importance(campaign_data_analysis,'age','35-44')
feature_and_importance_4554 = find_feature_and_importance(campaign_data_analysis,'age','45-54')
feature_and_importance_5564 = find_feature_and_importance(campaign_data_analysis,'age','55-64')
feature_and_importance_65 = find_feature_and_importance(campaign_data_analysis,'age','65+')
    
df_list=[feature_and_importance_female,feature_and_importance_male,feature_and_importance_unknown,
feature_and_importance_1824,feature_and_importance_2534,feature_and_importance_3544,
feature_and_importance_4554,feature_and_importance_5564,feature_and_importance_65]

result_df = pd.DataFrame(columns=['segment','value','recommend'])
    
for index,df in enumerate(df_list):
    temp=df[['feature','value','percentage']]
    temp=temp[temp.percentage.notnull()]
    temp.set_index(temp.feature, inplace = True)
    del temp['feature']
    temp['priority']=range(1, temp.shape[0]+1)
    temp_dict=temp.to_dict(orient='index')
    
    result_df.loc[index]=pd.Series({'segment':df.columns[0],'value':df.iloc[0][0],'recommend':temp_dict})




######################################################################
###########################
##### bandai analysis ##### 
###########################

bandai_df=mydata[mydata.account_id == 1539145383020911]
bandai_df=brand_column_generator(bandai_df,'萬代|bandai')
#pd.isnull(bandai_df).any(axis=0)
bandai_df=metric_generator(bandai_df)







########################################
#####find the best and worst ad ########
########################################

best_ad=find_best_ad(bandai_df)



####################################################
#####find the best ad and feature for each group ###
####################################################

best_ad_gender=find_best_ad_by_segment(bandai_df,'gender')
best_ad_age=find_best_ad_by_segment(bandai_df,'age')


#################################
#####filter analyzing columns ###
#################################
### ad related: title_brand, sub_title_brand, ad_content_brand, title_length_interval, sub_title_length_interval, ad_content_length_interval, call_to_action
### ad performance related: age, gender, impression, click
### google vision related: faceCount, majorColor, textInImage, logoInImage, adult, medical, spoof, violence, image category

bandai_df_analysis=column_selector(bandai_df)


#######################################################
#####find the feature and importance for each group ###
#######################################################
#print_full(feature_and_importance_4554)
#a=find_ad_feature(bandai_df, [6054156195798])
#print_full(a)

feature_and_importance_female = find_feature_and_importance(bandai_df_analysis,'gender','female')
feature_and_importance_male = find_feature_and_importance(bandai_df_analysis,'gender','male')
feature_and_importance_unknown = find_feature_and_importance(bandai_df_analysis,'gender','unknown')

feature_and_importance_1824 = find_feature_and_importance(bandai_df_analysis,'age','18-24')
feature_and_importance_2534 = find_feature_and_importance(bandai_df_analysis,'age','25-34')
feature_and_importance_3544 = find_feature_and_importance(bandai_df_analysis,'age','35-44')
feature_and_importance_4554 = find_feature_and_importance(bandai_df_analysis,'age','45-54')

 
######################################################################
###########################
##### mamaway analysis ##### 
###########################

mamaway_df=mydata[mydata.account_id == 1157415984328478]
mamaway_df=brand_column_generator(mamaway_df,'媽媽餵|mamaway')
#pd.isnull(mamaway_df).any(axis=0)
mamaway_df=metric_generator(mamaway_df)


a=mamaway_df[mamaway_df.ad_id==23842537195520136]


########################################
#####find the best and worst ad ########
########################################

best_ad=find_best_ad(mamaway_df)



########################################
#####find the best ad for each group ###
########################################

best_ad_gender=find_best_ad_by_segment(mamaway_df,'gender')
best_ad_age=find_best_ad_by_segment(mamaway_df,'age')





