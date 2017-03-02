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
insightPath='../facebook_ad_report_ver3/insightData/age_gender_json_0217'
googlevisionPath='../facebook_ad_report_ver3/insightData/with-x-y-google-vision_0217.json'

mydata = initDataFrame(insightPath, googlevisionPath)
#mydata['account_name'] = mydata['account_name'].str.strip()
mydata=metric_generator(mydata)

gv_df_label = gv_data_reader(googlevisionPath)


######################################################################
###########################
##### etungo analysis ##### 
###########################

etungo_df=mydata[mydata.account_id == 1618494121752703]
#etungo_df=brand_column_generator(etungo_df,'大同|etungo')
#pd.isnull(etungo_df).any(axis=0)





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


#############################################
#####find the best feature for each group ###
#############################################

final_feature=find_feature('gender','female',etungo_df,gv_df_label)


final_label=find_label_feature(etungo_df[etungo_df.age=='18-24'],gv_df_label,'age')
#final_label=find_label_feature(etungo_df,gv_df_label,'age')
#top_label=find_top_label(etungo_df,gv_df_label)

#final_content_keyword=find_keyword_feature(etungo_df[etungo_df.age=='18-24'],'content','age')
#top_keyword=find_top_keyword(etungo_df,'content')

campaign_id='6060850543605,6059224129805,6059889803605,6059892762805,6059893654805,6057193158805,6055407354005,6053312447405,6051265864405,6051266212805,6049374949605,6035884102605,6035882865005,6034201296805,6034192331405,6033466101805,6032523746205,6033118372005,6033118372205,6032205731205,6032523746005,6032205730605,6031518441405,6031616546205,6031518442005,6031518441605,6031518441205,6031518441005,6032906211805,6032906682605,6032907243605,6032261374205,6032627603405,6032260651005,6032626832005,6032321919605,6032411410005,6032321128805,6032200009405,6031742821405,6031796276605,6031722773205'
campaign_ids=campaign_id.split(',')
campaign_ids = map(int, campaign_ids)
    
campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
campaign_data=metric_generator(campaign_data)

best_ad_gender=find_best_ad_by_segment(campaign_data,'gender')
best_ad_age=find_best_ad_by_segment(campaign_data,'age')

gender_df=pd.melt(best_ad_gender,id_vars=['ad_id','impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
age_df=pd.melt(best_ad_age,id_vars=['ad_id','impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
df_adid = pd.concat([gender_df, age_df])
df_adid=df_adid[['feature','value','ad_id']]
df_adid = df_adid.reset_index(drop=True)
result_df = pd.DataFrame(columns=['segment','value','ad_id','feature'])

index=1
ad_id=6031616548405

ad_feature=find_ad_feature(campaign_data, gv_df_label, [ad_id])
ad_feature=ad_feature[['feature','value']].T
ad_feature.columns = ad_feature.iloc[0]
ad_feature.drop(ad_feature.index[0:1], inplace=True)
ad_feature_dict=ad_feature.to_dict(orient='records') #automatically deduplicate

result_df.loc[index]=pd.Series({'segment':df_adid['feature'][index],'value':df_adid['value'][index],
                          'ad_id':df_adid['ad_id'][index],'feature':ad_feature_dict})


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



###################################################
#####find the feature importance for each group ###
###################################################

importance_df=find_importance('gender','female',etungo_df,gv_df_label)

    
#######################################################
#####find the feature and importance for each group ###
#######################################################
#print_full(feature_and_importance_65)
#a=find_ad_feature(etungo_df, [6050712185805])
#print_full(a)
  

feature_and_importance_female = find_feature_and_importance('gender','female',etungo_df,gv_df_label)
feature_and_importance_male = find_feature_and_importance('gender','male',etungo_df,gv_df_label)
feature_and_importance_unknown = find_feature_and_importance('gender','unknown',etungo_df,gv_df_label)

feature_and_importance_1824 = find_feature_and_importance('age','18-24',etungo_df,gv_df_label)
feature_and_importance_2534 = find_feature_and_importance('age','25-34',etungo_df,gv_df_label)
feature_and_importance_3544 = find_feature_and_importance('age','35-44',etungo_df,gv_df_label)
feature_and_importance_4554 = find_feature_and_importance('age','45-54',etungo_df,gv_df_label)
feature_and_importance_5564 = find_feature_and_importance('age','55-64',etungo_df,gv_df_label)
feature_and_importance_65 = find_feature_and_importance('age','65+',etungo_df,gv_df_label)





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






