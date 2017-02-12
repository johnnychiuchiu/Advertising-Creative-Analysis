os.chdir("/Users/JohnnyChiu/Desktop/檔案總管/2016專案/R/1227_素材分析/Advertising-Creative-Analysis/")
os.getcwd()

import pandas as pd
import os


############################################################


#eval(parse("creative_function.R", encoding="UTF-8"))


######################
##### 讀檔案進來 ##### 
######################
#先把ad_informations.csv用sbulime轉utf8 with BOM, 之後用excel開另存成xlsx檔案，把original json的欄位刪掉，再用下面
ad_information= pd.read_excel("../facebook_ad_report_ver2/ad_informations.xlsx", sheetname='ad_informations')

ad_information=ad_information.iloc[:,0:11]

ad_age_gender= pd.read_csv("../facebook_ad_report_ver2/ad_performance_age_genders.csv")

google_vision= pd.read_csv('../facebook_ad_report_ver2/GoogleVisionModified.csv')
google_vision=google_vision.rename(columns = {'creativeId':'creative_id'})

account_manager_df= pd.read_excel("../facebook_ad_report_ver2/電商值案帳號.xlsx", sheetname='sheet1')
account_manager_df.columns=['account_name','account_manager']


###account data summary
summary_df=ad_information['account_name'].value_counts().sort


###########################
##### data manipulate ##### 
###########################
##ad_information
ad_information.account_name.unique()
account_manager_df.account_name

ad_information2=ad_information[ad_information.account_name.isin(account_manager_df.account_name)]
ad_information2=ad_information2[ad_information2.call_to_action != 'News Feed on desktop computers or Right column on desktop computers']
#ad_information2.columns
ad_information2=ad_information2.loc[:,['creative_id','ad_id','title','sub_title','ad_content','call_to_action']]
ad_information2.shape


##ad_age_gender
ad_age_gender2=ad_age_gender.loc[:,['account_name','account_id','ad_id','gender','age','impression','link_clicks']]


##google_vision
google_vision2=google_vision.iloc[:,3:12]



###合併
final_temp=pd.merge(ad_information2, google_vision2, on='creative_id', how='left')
final_temp2=pd.merge(ad_age_gender2,final_temp,on='ad_id',how='left')
# a lot of na columns, because the ad_id of ad_age_gender and ad_information cannot be matched.

final_df=final_temp2

######################################################################
###########################
##### etungo analysis ##### 
###########################
import numpy as np
etungo_df=final_df[final_df.account_name=='2016.06_大同_F']
etungo_df['cpm_random']=np.random.choice(range(60, 90), etungo_df.shape[0])/1000.0
etungo_df['spent']=etungo_df.impression*etungo_df.cpm_random
etungo_df.loc[:,['impression','link_clicks','spent']].head(30)
del etungo_df['cpm_random']

etungo_df.columns



########################################
#####find the best ad for each group ###
########################################

best_ad_age_gender=find_best_ad(etungo_df)

#################################
#####create analyzing columns ###
#################################
### ad related: title_brand, sub_title_brand, ad_content_brand, title_length_interval, sub_title_length_interval, ad_content_length_interval, call_to_action
### ad performance related: age, gender, impression, click
### google vision related: faceCount, majorColor, textInImage, logoInImage, adult, medical, spoof, violence, image category

analysis_column_generator(etungo_df,'title','sub_title','ad_content','大同|etungo')
etungo_df_analysis=etungo_df.loc[:,['gender','age','impression','link_clicks','spent','call_to_action','faceCount','majorColor',
'textInImage','adult','medical','spoof','violence','logoInImage','title_brand', 'title_length_interval',
'sub_title_brand','sub_title_length_interval','ad_content_brand','ad_content_length_interval']]




#############################################
#####find the best feature for each group ###
#############################################

final_feature=find_best_feature(etungo_df)
    

#############################################
#####find the best feature for each group ###
#############################################

importance_df=feature_importance(etungo_df_analysis)









