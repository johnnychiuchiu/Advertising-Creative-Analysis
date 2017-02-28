os.chdir("/Users/JohnnyChiu/Desktop/檔案總管/2016專案/R/1227_素材分析/Advertising-Creative-Analysis/")
os.getcwd()

import pandas as pd
import os
import numpy as np
from creative_function import *
from initDataFrame import *

impression_threshold=1000
label_threshold=100
  


######################
##### 讀檔案進來 ##### 
######################
insightPath='../facebook_ad_report_ver3/insightData/age_gender_json_0217'
googlevisionPath='../facebook_ad_report_ver3/insightData/with-x-y-google-vision_0217.json'

mydata = initDataFrame(insightPath, googlevisionPath)
mydata['account_name'] = mydata['account_name'].str.strip()

gv_df_label = gv_data_reader(googlevisionPath)


######################################################################
###########################
##### etungo analysis ##### 
###########################

etungo_df=mydata[mydata.account_id == 1618494121752703]
etungo_df=brand_column_generator(etungo_df,'大同|etungo')
#pd.isnull(etungo_df).any(axis=0)
etungo_df=metric_generator(etungo_df)
etungo_df=image_label_generator(etungo_df,gv_df_label,label_threshold)



########################################
#####find the best and worst ad ########
########################################

best_ad=find_best_ad(etungo_df)



####################################################
#####find the best ad and feature for each group ###
####################################################

best_ad_gender=find_best_ad_by_segment(etungo_df,'gender')
best_ad_age=find_best_ad_by_segment(etungo_df,'age')

a=find_ad_feature(etungo_df, gv_df_label, [6055843154405])
#best_ad_gender_feature=find_ad_feature(etungo_df, gv_df_label, best_ad_gender.ad_id.unique().tolist())




##############################################
#####filter and generate analyzing columns ###
##############################################
### ad related: title_brand, sub_title_brand, ad_content_brand, title_length_interval, sub_title_length_interval, ad_content_length_interval, call_to_action
### ad performance related: age, gender, impression, click
### google vision related: faceCount, majorColor, textInImage, logoInImage, adult, medical, spoof, violence, image category

etungo_df_analysis=column_selector(etungo_df,gv_df_label)
#etungo_df_analysis.to_csv('etungo_df_analysis.csv', sep=',', encoding='utf-8')
#pd.isnull(etungo_df_analysis).any(axis=0)



#############################################
#####find the best feature for each group ###
#############################################

final_feature=find_feature(etungo_df_analysis,'age','18-24',etungo_df,gv_df_label)
#final_feature.to_csv('final_feature.csv', sep=',', encoding='utf-8')

final_label=find_label_feature(etungo_df[etungo_df.age=='18-24'],gv_df_label,'age')
#final_label=find_label_feature(etungo_df[etungo_df.gender=='female'],gv_df_label,'gender')

top_label=find_top_label(df,gv_df_label)

###################################################
#####find the feature importance for each group ###
###################################################

importance_df=find_importance(etungo_df_analysis,'age','18-24',etungo_df,gv_df_label)

    
#######################################################
#####find the feature and importance for each group ###
#######################################################
#print_full(feature_and_importance_65)
#a=find_ad_feature(etungo_df, [6050712185805])
#print_full(a)
  

feature_and_importance_female = find_feature_and_importance(etungo_df_analysis,'gender','female',etungo_df,gv_df_label)
feature_and_importance_male = find_feature_and_importance(etungo_df_analysis,'gender','male',etungo_df,gv_df_label)
feature_and_importance_unknown = find_feature_and_importance(etungo_df_analysis,'gender','unknown',etungo_df,gv_df_label)

feature_and_importance_1824 = find_feature_and_importance(etungo_df_analysis,'age','18-24',etungo_df,gv_df_label)
feature_and_importance_2534 = find_feature_and_importance(etungo_df_analysis,'age','25-34',etungo_df,gv_df_label)
feature_and_importance_3544 = find_feature_and_importance(etungo_df_analysis,'age','35-44',etungo_df,gv_df_label)
feature_and_importance_4554 = find_feature_and_importance(etungo_df_analysis,'age','45-54',etungo_df,gv_df_label)
feature_and_importance_5564 = find_feature_and_importance(etungo_df_analysis,'age','55-64',etungo_df,gv_df_label)
feature_and_importance_65 = find_feature_and_importance(etungo_df_analysis,'age','65+',etungo_df,gv_df_label)




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






