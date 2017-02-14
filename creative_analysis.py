os.chdir("/Users/JohnnyChiu/Desktop/檔案總管/2016專案/R/1227_素材分析/Advertising-Creative-Analysis/")
os.getcwd()

import pandas as pd
import os
import numpy as np
from creative_function import *


######################
##### 讀檔案進來 ##### 
######################

myjson = pd.read_json('../facebook_ad_report_ver3/age_gender_json_0214')
myjson['account_name'] = myjson['account_name'].str.strip()
#myjson.account_id.value_counts()
#myjson.account_name.value_counts()



######################################################################
###########################
##### etungo analysis ##### 
###########################

etungo_df=myjson[myjson.account_id == 1618494121752703]
etungo_df=etungo_df[(etungo_df.ad_type=='Ad with an image') | (etungo_df.ad_type=='Ad Use existing post')]
#pd.isnull(etungo_df).any(axis=0)


########################################
#####find the best ad for each group ###
########################################

best_ad_age_gender=find_best_ad(etungo_df)

df_6034192350805=find_ad_feature(etungo_df, [6034192350805])
df_adid=find_ad_feature(etungo_df, best_ad_age_gender.ad_id.unique().tolist())



#################################
#####create analyzing columns ###
#################################
### ad related: title_brand, sub_title_brand, ad_content_brand, title_length_interval, sub_title_length_interval, ad_content_length_interval, call_to_action
### ad performance related: age, gender, impression, click
### google vision related: faceCount, majorColor, textInImage, logoInImage, adult, medical, spoof, violence, image category

analysis_column_generator(etungo_df,'title','subtitle','message','大同|etungo')
etungo_df_analysis=etungo_df.loc[:,['gender','age','ad_type','bidding_type','call_to_action','campaign_goal',
'impression','link_clicks','spend','page_category','title_brand', 'title_length_interval',
'sub_title_brand','sub_title_length_interval','ad_content_brand','ad_content_length_interval']]


#############################################
#####find the best feature for each group ###
#############################################

final_feature=find_feature(etungo_df_analysis)

best_feature_female_1824=find_feature(etungo_df_analysis[(etungo_df_analysis.gender=='female') & (etungo_df_analysis.age=='18-24')])
print_full(best_feature_female_1824)
#final_feature.to_csv('final_feature.csv', sep=',', encoding='utf-8')
#etungo_df_analysis.head(10) 
#test=final_feature[final_feature.variable=='title_brand']
#test.head(20)

###################################################
#####find the feature importance for each group ###
###################################################

importance_df=find_importance(etungo_df_analysis)


#######################################################
#####find the feature and importance for each group ###
#######################################################

feature_and_importance_female_1824 = find_feature_and_importance(etungo_df_analysis,'female','18-24')
feature_and_importance_female_2534 = find_feature_and_importance(etungo_df_analysis,'female','25-34')
feature_and_importance_female_3544 = find_feature_and_importance(etungo_df_analysis,'female','35-44')
feature_and_importance_female_4554 = find_feature_and_importance(etungo_df_analysis,'female','45-54')
feature_and_importance_female_5564 = find_feature_and_importance(etungo_df_analysis,'female','55-64')
feature_and_importance_female_65 = find_feature_and_importance(etungo_df_analysis,'female','65+')

feature_and_importance_male_1824 = find_feature_and_importance(etungo_df_analysis,'male','18-24')
feature_and_importance_male_2534 = find_feature_and_importance(etungo_df_analysis,'male','25-34')
feature_and_importance_male_3544 = find_feature_and_importance(etungo_df_analysis,'male','35-44')
feature_and_importance_male_4554 = find_feature_and_importance(etungo_df_analysis,'male','45-54')
feature_and_importance_male_5564 = find_feature_and_importance(etungo_df_analysis,'male','55-64')
feature_and_importance_male_65 = find_feature_and_importance(etungo_df_analysis,'male','65+')

feature_and_importance_unknown_1824 = find_feature_and_importance(etungo_df_analysis,'unknown','18-24')
feature_and_importance_unknown_2534 = find_feature_and_importance(etungo_df_analysis,'unknown','25-34')
feature_and_importance_unknown_3544 = find_feature_and_importance(etungo_df_analysis,'unknown','35-44')
feature_and_importance_unknown_4554 = find_feature_and_importance(etungo_df_analysis,'unknown','45-54')
feature_and_importance_unknown_5564 = find_feature_and_importance(etungo_df_analysis,'unknown','55-64')
feature_and_importance_unknown_65 = find_feature_and_importance(etungo_df_analysis,'unknown','65+')



#potential issue:如果用score拿來當importance計算的依據，每一個feature的importance都會差不多
#                因為score的計算方法已經把最大最小值的影響爆包含在內了
#next step
#1. 表情符號判定
#2. find_best_ad sccipt
# input:  
# output:
#3. find_best_feature sccipt
# input: 
# output:
#4. feature_importance sccipt
# input: 
# output:






