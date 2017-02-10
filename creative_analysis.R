library(XLConnect)
library(dplyr)
options(scipen=999)# disable scientific notation
eval(parse("creative_function.R", encoding="UTF-8"))

######################
##### 讀檔案進來 ##### 
######################
#先把ad_informations.csv用sbulime轉utf8 with BOM, 之後用excel開另存成xlsx檔案，把original json的欄位刪掉，再用下面
ad_information = readWorksheet(loadWorkbook('../facebook_ad_report_ver2/ad_informations.xlsx'), sheet = 'ad_informations', 
                               header = TRUE)
ad_information = ad_information %>% select(1:11) 

ad_age_gender<-read.csv('../facebook_ad_report_ver2/ad_performance_age_genders.csv')
ad_age_gender$account_id<-as.character(ad_age_gender$account_id)
ad_age_gender$creative_id<-as.character(ad_age_gender$creative_id)
ad_age_gender$ad_id<-as.character(ad_age_gender$ad_id)

google_vision<- read.csv('../facebook_ad_report_ver2/GoogleVisionModified.csv')
colnames(google_vision)[which(names(google_vision) == "creativeId")] <- "creative_id"


account_manager_df<-readWorksheet(loadWorkbook('../facebook_ad_report_ver2/電商值案帳號.xlsx'), sheet = '工作表1', header = TRUE)
colnames(account_manager_df)[which(names(account_manager_df) == "帳號")] <- "account_name"
colnames(account_manager_df)[which(names(account_manager_df) == "值案人")] <- "account_manager"

ad_url<-read.csv('../facebook_ad_report_ver2/ad_url.csv')
ad_url$creative_id<-as.character(ad_url$creative_id)

###account data summary
summary_df<- as.data.frame(sort(table(ad_information$account_name),decreasing = TRUE))



###########################
##### data manipulate ##### 
###########################
##ad_information
ad_information2<-ad_information %>% filter(account_name %in% unique(account_manager_df$account_name))
ad_information2<- ad_information2 %>% filter(call_to_action != 'News Feed on desktop computers or Right column on desktop computers ')
ad_information2<-ad_information2 %>% select(creative_id,ad_id:call_to_action)

##ad_age_gender
ad_age_gender2<- ad_age_gender %>% select(account_name,account_id,ad_id,gender,age,impression,link_clicks)

##google_vision
google_vision2<- google_vision %>% select(creative_id:logoInImage)
google_vision2$creative_id<- as.character(google_vision2$creative_id)

###合併
final_temp<- merge(ad_information2, google_vision2, by='creative_id', all.x = TRUE)

final_temp2<- merge(ad_age_gender2, final_temp, by='ad_id', all.x = TRUE)
# a lot of na columns, because the ad_id of ad_age_gender and ad_information cannot be matched.

# final_temp3<- final_temp2 %>% filter(!is.na(adult))
# 
# number_calculate<- final_temp2 %>% filter(!is.na(adult))
# creative_calculate<- number_calculate %>% group_by(account_name, creative_id) %>% summarise(count=n())
# creative_calculate<- creative_calculate %>% group_by(account_name) %>% summarise(count=n())
# account_calculate<-data.frame(table(final_temp3$account_name))
# 
# final_df<- final_temp3
# final_df$call_to_action<- as.factor(final_df$call_to_action)
# final_df$account_manager<- as.factor(final_df$account_manager)
final_df<- final_temp2

######################################################################
###########################
##### etungo analysis ##### 
###########################
etungo_df<- final_df %>% filter(account_name=='2016.06_大同_F')
etungo_df[etungo_df=="NULL"]<-NA




########################################
#####find the best ad for each group ###
########################################
best_ad_age_gender <- find_best_ad(etungo_df)


#################################
#####create analyzing columns ###
#################################
### ad related: title_brand, sub_title_brand, ad_content_brand, title_length_interval, sub_title_length_interval, ad_content_length_interval, call_to_action
### ad performance related: age, gender, impression, click
### google vision related: faceCount, majorColor, textInImage, logoInImage, adult, medical, spoof, violence, image category

etungo_df<- etungo_df %>% filter(!is.na(title))
etungo_df<- analysis_column_generator(etungo_df,'title','sub_title','ad_content',brand_keywords<-c('大同'))
etungo_df_analysis<- etungo_df %>% 
  select(gender:link_clicks, call_to_action:logoInImage, 
         title_brand, title_length_interval,
         sub_title_brand, sub_title_length_interval,
         ad_content_brand, ad_content_length_interval)



#############################################
#####find the best feature for each group ###
#############################################
final_feature<-find_best_feature(etungo_df_analysis)


#############################################
##### the group_by age plot #################(tony)
#############################################
library(rCharts)
# impression_threshold=1000
# age_color<- etungo_df %>% group_by(age, majorColor) %>% 
#   dplyr::summarise(total_impression=sum(impression), total_click=sum(link_clicks)) %>% 
#   filter(total_impression > impression_threshold) %>%
#   mutate(CTR=total_click/total_impression) %>%
#   group_by_(.dots=dots2) %>% slice(which.max(CTR)) 
# 








