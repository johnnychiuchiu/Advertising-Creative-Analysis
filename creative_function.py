import pandas as pd
import os
import numpy as np
import copy
from flask import jsonify
import json
############################################ print_full ############################################ 
##### goal: print the full pandas data frame
##### input
#####       x: a pandas data frame
##### output 
#####       print the full data frame


def print_full(x):
    with pd.option_context('display.max_rows', None, 'display.max_columns', x.shape[1] ):
        print(x)



############################################ webClickScore ############################################ 
##### input
#####       cpc
#####       ctr
#####       clicks
##### output 
#####       a pandas dataframe column 


def webClickScore(cpc, ctr, clicks):
    scoreSum =  (ctr - ctr.mean()) / (ctr.max() - ctr.min()) + (clicks - clicks.mean()) / (clicks.max() - clicks.min()) - (cpc - cpc.mean()) / (cpc.max() - cpc.min())
    return 70 + 40 * (scoreSum - scoreSum.mean()) / (scoreSum.max() - scoreSum.min())



############################################ metric_generator ############################################ 
##### goal: 
#####       1. replace link_clicks with link_clicks + 1
#####       2. generate 'score','cpc','ctr' columns by each row
#####       3. replace rows with abnormal value in CPC and CTR with mean value
##### input
#####       df: craetive table
##### output 
#####       a data frame that the targeted metrics, including link_clicks, cpc, ctr, is being replaced

def metric_generator(df):
    impression_threshold=10

    df['link_clicks']=df['link_clicks']+1
    df=df.assign(CTR=df.link_clicks/df.impression*1.0) 
    df=df.assign(CPC=df.spend/df.link_clicks*1.0) 
    df_filter=df[ (df.impression>impression_threshold) & (df.link_clicks!=1) & (df.CTR<1) ]
    
    cpc_mean=df_filter.CPC.mean()
    ctr_mean=df_filter.CTR.mean()
    
    df.CPC[ (df.impression<impression_threshold) | (df.link_clicks==1) | (df.CTR>1) ]= cpc_mean
    df.CTR[ (df.impression<impression_threshold) | (df.link_clicks==1) | (df.CTR>1) ]= ctr_mean
    df['score']=webClickScore(df.CPC, df.CTR, df.link_clicks)
    
    return df 



############################################ find_best_ad ############################################ 
##### input
#####       df: craetive table
##### output 
#####       a dataframe that provides the best ad for the targeted dimension, such as {gender, age}

def find_best_ad(df):
  
    impression_threshold=1000

    f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
    temp=df.groupby(['ad_id']).agg(f).reset_index()
    temp=temp[temp.impression>impression_threshold]
    temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
    temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0) 
        
    temp_max=temp[temp['score']==temp['score'].max()]
    temp_max.loc[:,('ranking')]='best'
    temp_min=temp[temp['score']==temp['score'].min()]
    temp_min.loc[:,('ranking')]='worst'
    result = pd.concat([temp_max, temp_min])
    my_round=lambda x: x.round(1)
    result.loc[:,['score']] = result.score.map(my_round)

    return result
    

    
############################################ find_best_ad_by_segment ############################################ 
##### input
#####       df: craetive table
#####       segment: a segment type, such as 'gender','age'
##### output 
#####       a dataframe that provides the best ad for the targeted dimension, such as {gender, age}

def find_best_ad_by_segment(df,segment):
    impression_threshold=1000
    
    f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
    temp=df.groupby([segment]+['ad_id']).agg(f).reset_index()
    temp=temp[temp.impression>impression_threshold]
    temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
    temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0) 
        
    idx=temp.groupby([segment])['score'].transform(max) == temp['score']
    result=temp[idx]
    my_round=lambda x: x.round(1)
    result.loc[:,['score']] = result.score.map(my_round)

    return result


############################################ brand_column_generator ############################################ 
##### input
#####       df: craetive table
#####       brand_keywords: a vector of the brand keywords
##### output 
#####       a table appended with brand related columns 

def brand_column_generator(df, brand_keywords):
    
    # title_brand
    df['title_brand']=df['title'].str.contains(brand_keywords.decode('utf-8'))
    
    # sub_title_brand
    df['subtitle_brand']=df['subtitle'].str.contains(brand_keywords.decode('utf-8'))
    
    # ad_content_brand
    df['content_brand']=df['content'].str.contains(brand_keywords.decode('utf-8'))
    
    return df



############################################ analysis_column_generator ############################################ 
##### input
#####       df: craetive table
#####       title_colname: the colname of title
#####       sub_title_colname: the colname of subtitle
#####       ad_content_colname: the colname of ad content
#####       brand_keywords: a vector of the brand keywords
##### output 
#####       a table with analysis columns that ready for analysis  
def analysis_column_generator(df,title_colname, sub_title_colname, ad_content_colname, brand_keywords):
    ###ad information related
    #title: 25 characters, sub_title: 30 characters, ad_content: 90 characters
    title_max_length=25
    sub_title_max_length=30
    ad_content_max_length=90
    
    
    # title_length
    title_group_names = ['Short(0-12)','Medium(13-25)','Long(>25)']
    df[title_colname] = df[title_colname].str.strip()
    df['title_length'] = df[title_colname].str.len()
    bins = [0, title_max_length*0.5, title_max_length, float("inf")]
    df['title_length_interval'] = pd.cut(df['title_length'], bins, labels=title_group_names)

    # title_brand
    df['title_brand']=df[title_colname].str.contains(brand_keywords.decode('utf-8'))
    
    # sub_title_length
    subtitle_group_names = ['Short(0-15)','Medium(16-30)','Long(>30)']
    df[sub_title_colname] = df[sub_title_colname].str.strip()
    df['sub_title_length'] = df[sub_title_colname].str.len()    
    bins = [0, sub_title_max_length*0.5, sub_title_max_length, float("inf")]
    df['sub_title_length_interval'] = pd.cut(df['sub_title_length'], bins, labels=subtitle_group_names)

    # sub_title_brand
    df['sub_title_brand']=df[sub_title_colname].str.contains(brand_keywords.decode('utf-8'))
    
    # ad_content_length
    ad_content_group_names = ['Short(0-45)','Medium(46-90)','Long(>90)']
    df[ad_content_colname] = df[ad_content_colname].str.strip()
    df['ad_content_length'] = df[ad_content_colname].str.len()    
    bins = [0, ad_content_max_length*0.5, ad_content_max_length, float("inf")]
    df['ad_content_length_interval'] = pd.cut(df['ad_content_length'], bins, labels=ad_content_group_names)

    # ad_content_brand
    df['ad_content_brand']=df[ad_content_colname].str.contains(brand_keywords.decode('utf-8'))
    
    return df



############################################ find_ad_feature ############################################ 
##### input
#####       df: a data frame which columns contains ad_id and columns ready for analysis, such as etungo_df
#####       ad_id: list of ad_id, such as [123,456]. For pandas dataframe, use df.ad_id.unique().tolist()
##### output 
#####       a dataframe contains the feature of this ad_id

def find_ad_feature(df,gv_df_label,ad_id):
    dimension=['gender','age']
    drop_columns=dimension+['account_id','account_name','ad_set_id','campaign_goal',
                        'campaign_id','content','creative_id','creative_url','fanpage',
                        'fanpage_industry','impression','interest',
                        'link_clicks','page_id','subtitle','title',
                        'spend','CPC','CTR','score']
    df=df.drop(drop_columns, axis=1)
    
    gv_df_label = gv_df_label[gv_df_label['ad_id'].isin(df.ad_id.unique())]
    drop_columns=get_label_name(df,gv_df_label,100)
    df=df.drop(drop_columns, axis=1)
    
    df_adid=df[df['ad_id'].isin(ad_id)]
    df_adid=df_adid.drop_duplicates()
    df_adid=pd.melt(df_adid,id_vars=['ad_id'],var_name='feature')
    
    gv_df_label_adid=gv_df_label[gv_df_label['ad_id'].isin(ad_id)]
    gv_df_label_adid=gv_df_label_adid.rename(columns = {'label':'value'})
    gv_df_label_adid['feature']='label'
    cols_sort=['ad_id','feature','value']
    gv_df_label_adid=gv_df_label_adid[cols_sort]
    
    df_adid=df_adid.append(gv_df_label_adid,ignore_index=True)
    
    df_adid['sort'] = pd.Categorical(df_adid['ad_id'], ad_id)
    df_adid=df_adid.sort_values(by="sort")
    del df_adid['sort']
    
    return df_adid 
    

############################################ column_selector ############################################ 
##### input
#####       df: a data frame which columns contains ad_id and columns ready for analysis 
##### output 
#####       a dataframe contains only analysis related columns

def column_selector(df,gv_df_label):
    result_df=copy.deepcopy(df)
    drop_columns=['account_id','account_name','ad_id','ad_set_id','campaign_goal','campaign_id',
                   'content','creative_id','creative_url','fanpage','fanpage_industry',
                    'interest','page_id','subtitle','title']
    result_df=result_df.drop(drop_columns, axis=1)
    
    drop_columns=get_label_name(df,gv_df_label,100)
    result_df=result_df.drop(drop_columns, axis=1)
    
    
    return result_df 
    

############################################ find_feature ############################################ 
##### input
#####       analysis_df: a table with all columns that are ready for analysis, with metric columns(such as CPC,CTR,score)
#####       segment: a segment type, such as 'gender','age'
##### output 
#####       a dataframe that provides the best feature for the targeted segment, such as {gender, age}
def find_feature(analysis_df,segment,value,df,gv_df_label):
    final_feature = pd.DataFrame()
    impression_threshold=1000
    label_threshold=100
    
    #gv_df_label_filtered = gv_df_label[gv_df_label['ad_id'].isin(df.ad_id.unique())]
    #drop_columns=get_label_name(df,gv_df_label,100)
    #analysis_df=analysis_df.drop(drop_columns, axis=1)
    
    analysis_df_filtered=analysis_df[analysis_df[segment]==value]
    df_filtered=df[df[segment]==value]

    if segment=='gender':
        analysis_df_filtered=analysis_df_filtered.drop(['age'], axis=1)
    elif segment=='age':   
        analysis_df_filtered=analysis_df_filtered.drop(['gender'], axis=1)

    for name in analysis_df_filtered:
        if name not in [segment]+['impression','link_clicks','spend','CTR','CPC','score']:
            print name
            f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
            temp=analysis_df_filtered.groupby([segment]+[name]).agg(f).reset_index()
            temp=temp[temp.impression>impression_threshold]
            temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
            temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0) 
                
            idx= temp.groupby([segment])['score'].transform(max) == temp['score']
            temp2=temp[idx]
            
            temp_reshape=pd.melt(temp2,id_vars=[segment]+['impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
            temp_reshape=temp_reshape.loc[:,[segment]+['feature','value','impression','link_clicks','spend','CTR','CPC','score']]
            
            final_feature=pd.concat([final_feature, temp_reshape])
    
    #merge the recommend label row back to final_feature
    final_label=find_label_feature(df_filtered,gv_df_label,label_threshold,segment)
    final_feature = pd.concat([final_feature, final_label])[['feature',segment,'value']] 
            
    final_feature=final_feature.sort_values(by=[segment], ascending=True)
    

    return final_feature

############################################ find_importance ############################################ 
##### goal: find the feature importance for each dimension
##### input
#####       analysis_df: a table with all columns that are ready for analysis
#####       segment: a segment type, such as 'gender','age'
##### output 
#####       a dataframe that provides the feature importance ranking for the targeted dimension, such as {gender, age}

def find_importance(analysis_df,segment,value,df,gv_df_label):
    impression_threshold=1000
    
    analysis_df=analysis_df[analysis_df[segment]==value]
    df=df[df[segment]==value]

    top_label=find_top_label(df,gv_df_label)
    analysis_df['label']=df[top_label]
    
    feature=copy.deepcopy(analysis_df)
    feature=feature.drop(['impression','link_clicks','spend','CTR','CPC','score'], axis=1)
    importance_df=pd.DataFrame({'feature':feature.columns,
                            'max_impression':0,'max_click':0,'max_spend':0,'max_ctr':0,'max_cpc':0,'max_score':0,
                            'min_impression':0,'min_click':0,'min_spend':0,'min_ctr':0,'min_cpc':0,'min_score':0,
                            'importance':0})
    
    for name in feature.columns:
        print name
        f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
        temp=analysis_df.groupby([name]).agg(f).reset_index()
        temp=temp[temp.impression>impression_threshold]
        temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
        temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0)
    
        if temp.shape[0]<=1:
            importance_df=importance_df[importance_df.feature!=name]
        else:
            max_score= temp.score.max()
            min_score= temp.score.min()
            importance_df.importance[importance_df.feature==name]= max_score-min_score
            importance_df.max_score[importance_df.feature==name]= max_score
            importance_df.min_score[importance_df.feature==name]= min_score
            importance_df.max_impression[importance_df.feature==name]= temp.impression[temp.score==max_score].values
            importance_df.min_impression[importance_df.feature==name]= temp.impression[temp.score==min_score].values
            importance_df.max_click[importance_df.feature==name]= temp.link_clicks[temp.score==max_score].values
            importance_df.min_click[importance_df.feature==name]= temp.link_clicks[temp.score==min_score].values
            importance_df.max_spend[importance_df.feature==name]= temp.spend[temp.score==max_score].values
            importance_df.min_spend[importance_df.feature==name]= temp.spend[temp.score==min_score].values
            importance_df.max_ctr[importance_df.feature==name]= temp.CTR[temp.score==max_score].values
            importance_df.min_ctr[importance_df.feature==name]= temp.CTR[temp.score==min_score].values
            importance_df.max_cpc[importance_df.feature==name]= temp.CPC[temp.score==max_score].values
            importance_df.min_cpc[importance_df.feature==name]= temp.CPC[temp.score==min_score].values
                
    importance_df=importance_df.sort_values(by=['importance'], ascending=False) 
    importance_df['percentage']=  importance_df.importance / importance_df.importance.sum()

    return importance_df



############################################ find_label_feature ############################################ 
##### goal: find the label feature for each dimension
##### input
#####       df: a table with all columns that are ready for analysis, it's different from find_importance's analysis_df
#####       segment: a segment type, such as 'gender','age'
##### output 
#####       a dataframe that provides the feature importance ranking for the targeted dimension, such as {gender, age}
def find_label_feature(df,gv_df_label,label_threshold,segment):
    result_df = pd.DataFrame(columns=[segment,'feature','value'])
    impression_threshold=1000
    
    for segment_value in df[segment].unique():
        print segment_value
        gv_df_label_filtered = gv_df_label[gv_df_label['ad_id'].isin(df[df[segment]==segment_value].ad_id.unique())]
        label_name=get_label_name(df,gv_df_label_filtered,label_threshold)
        importance_df=pd.DataFrame({'feature':pd.Series(label_name),
                            'max_impression':0,'max_click':0,'max_spend':0,'max_ctr':0,'max_cpc':0,'max_score':0,
                            'min_impression':0,'min_click':0,'min_spend':0,'min_ctr':0,'min_cpc':0,'min_score':0,
                            'importance':0})
        
        for name in label_name:
            print name
            f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
            temp=df[df[segment]==segment_value].groupby([name]).agg(f).reset_index()
            temp=temp[temp.impression>impression_threshold]
            temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
            temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0)
        
            if temp.shape[0]<=1:
                importance_df=importance_df[importance_df.feature!=name]
            else:
                max_score= temp.score.max()
                min_score= temp.score.min()
                importance_df.importance[importance_df.feature==name]= max_score-min_score
                importance_df.max_score[importance_df.feature==name]= max_score
                importance_df.min_score[importance_df.feature==name]= min_score
                importance_df.max_impression[importance_df.feature==name]= temp.impression[temp.score==max_score].values
                importance_df.min_impression[importance_df.feature==name]= temp.impression[temp.score==min_score].values
                importance_df.max_click[importance_df.feature==name]= temp.link_clicks[temp.score==max_score].values
                importance_df.min_click[importance_df.feature==name]= temp.link_clicks[temp.score==min_score].values
                importance_df.max_spend[importance_df.feature==name]= temp.spend[temp.score==max_score].values
                importance_df.min_spend[importance_df.feature==name]= temp.spend[temp.score==min_score].values
                importance_df.max_ctr[importance_df.feature==name]= temp.CTR[temp.score==max_score].values
                importance_df.min_ctr[importance_df.feature==name]= temp.CTR[temp.score==min_score].values
                importance_df.max_cpc[importance_df.feature==name]= temp.CPC[temp.score==max_score].values
                importance_df.min_cpc[importance_df.feature==name]= temp.CPC[temp.score==min_score].values
                    
        importance_df=importance_df.sort_values(by=['importance'], ascending=False) 
        importance_df['percentage']=  importance_df.importance / importance_df.importance.sum()
        
        row_data={segment:segment_value,'feature':'label','value':list(importance_df['feature'][0:5])}
        
        temp=pd.DataFrame.from_dict(row_data,orient='columns')
        temp=pd.DataFrame(temp.groupby([segment,'feature'])['value'].apply(list)).reset_index()
        result_df=result_df.append(temp,ignore_index=True)
    
    return result_df



############################################ find_top_label ############################################ 
##### goal: find the label importance for each dimension
##### input
#####       df: a table with all columns that are ready for analysis, it's different from find_importance's analysis_df
#####       segment: a segment type, such as 'gender','age'
##### output 
#####       a dataframe that provides the feature importance ranking for the targeted dimension, such as {gender, age}
def find_top_label(df,gv_df_label):
    label_threshold=100
    impression_threshold=1000
    gv_df_label_filtered = gv_df_label[gv_df_label['ad_id'].isin(df.ad_id.unique())]
    label_name=get_label_name(df,gv_df_label_filtered,label_threshold)
    importance_df=pd.DataFrame({'feature':pd.Series(label_name),
                        'max_impression':0,'max_click':0,'max_spend':0,'max_ctr':0,'max_cpc':0,'max_score':0,
                        'min_impression':0,'min_click':0,'min_spend':0,'min_ctr':0,'min_cpc':0,'min_score':0,
                        'importance':0})
    
    for name in label_name:
        print name
        f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
        temp=df.groupby([name]).agg(f).reset_index()
        temp=temp[temp.impression>impression_threshold]
        temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
        temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0)
    
        if temp.shape[0]<=1:
            importance_df=importance_df[importance_df.feature!=name]
        else:
            max_score= temp.score.max()
            min_score= temp.score.min()
            importance_df.importance[importance_df.feature==name]= max_score-min_score
            importance_df.max_score[importance_df.feature==name]= max_score
            importance_df.min_score[importance_df.feature==name]= min_score
            importance_df.max_impression[importance_df.feature==name]= temp.impression[temp.score==max_score].values
            importance_df.min_impression[importance_df.feature==name]= temp.impression[temp.score==min_score].values
            importance_df.max_click[importance_df.feature==name]= temp.link_clicks[temp.score==max_score].values
            importance_df.min_click[importance_df.feature==name]= temp.link_clicks[temp.score==min_score].values
            importance_df.max_spend[importance_df.feature==name]= temp.spend[temp.score==max_score].values
            importance_df.min_spend[importance_df.feature==name]= temp.spend[temp.score==min_score].values
            importance_df.max_ctr[importance_df.feature==name]= temp.CTR[temp.score==max_score].values
            importance_df.min_ctr[importance_df.feature==name]= temp.CTR[temp.score==min_score].values
            importance_df.max_cpc[importance_df.feature==name]= temp.CPC[temp.score==max_score].values
            importance_df.min_cpc[importance_df.feature==name]= temp.CPC[temp.score==min_score].values
                
    importance_df=importance_df.sort_values(by=['importance'], ascending=False).reset_index() 
    importance_df['percentage']=  importance_df.importance / importance_df.importance.sum()
    top_label=importance_df['feature'][0]
    
    return top_label





############################################ best_feature_and_importance ############################################ 
##### input
#####       analysis_df: a table with all columns that are ready for analysis
#####       segment: a segment type, such as 'gender','age'
#####       value: a target value for this segment
##### output 
#####       a dataframe that provides the best feature and importance for the targeted dimension, such as {gender, age}

def find_feature_and_importance(analysis_df,segment,value,df,gv_df_label):
    
    best_feature=find_feature(analysis_df,segment,value,df,gv_df_label)
    importance=find_importance(analysis_df,segment,value,df,gv_df_label)

    feature_and_importance=pd.merge(best_feature, importance, on=['feature'], how='left')
    feature_and_importance=feature_and_importance.sort_values(by='importance', ascending=False)
    
    feature_and_importance = feature_and_importance[[segment]+[
                                                    'feature',
                                                    'value',
                                                    'percentage',
                                                    'importance',
                                                    'max_click',
                                                    'max_cpc',
                                                    'max_ctr',
                                                    'max_impression',
                                                    'max_score',
                                                    'max_spend',
                                                    'min_click',
                                                    'min_cpc',
                                                    'min_ctr',
                                                    'min_impression',
                                                    'min_score',
                                                    'min_spend',]]
    my_round=lambda x: x.round(2)
    feature_and_importance.loc[:,['percentage']] = feature_and_importance.percentage.map(my_round)                                                
    
    return feature_and_importance
    


############################################ image_label_generator ########################
##### input
#####       df: a data frame that is filtered and ready for analysis
#####       label_threshold: the threshold for the label count, that is, the label count number
#####                         should exceend the threshold to be included in the new column.
#####       gv_df_label: a data frame contains 'ad_id' and 'label'
##### output 
#####       a data frame with label columns
def description_extract(mylist):
    return [item['description'] for item in mylist]


def flatten(l):
    return [item for sublist in l for item in sublist]

def get_label_name(df,gv_df_label,label_threshold):
    gv_df_label = gv_df_label[gv_df_label['ad_id'].isin(df.ad_id.unique())]
    
    label_flatten=flatten(gv_df_label['label'])
    label_name=pd.Series(label_flatten).value_counts()[pd.Series(label_flatten).value_counts() >label_threshold].index
    return label_name

def image_label_generator(df,label_threshold,gv_df_label):
    gv_df_label = gv_df_label[gv_df_label['ad_id'].isin(df.ad_id.unique())]
    
    label_name=get_label_name(df,gv_df_label,100)
    label_df=pd.DataFrame({'ad_id':gv_df_label['ad_id']})
    label_df=label_df.reset_index(drop=True)
    
    for label in label_name:
        label_df[label]=pd.Series([label in list for list in gv_df_label['label']])
    
      
    #drop_columns=['ad_type','creative_id','face_annotation','height','id','image_properties_annotation',
    #            'image_url','label_annotations','landmarking_confidence','logo_annotations',
    #            'safe_search_annotation','subtitle','text_annotations','title','width']
    #gv_df=gv_df.drop(drop_columns, axis=1)     
    #gv_df=gv_df.reset_index(drop=True)
        
    result_df = pd.merge(df,label_df, on='ad_id', how='left')
    return result_df    
    
    
############################################ gv_data_reader ########################
##### input
#####       googlevisionPath: the path to the gv dataset
##### output 
#####       a data frame contains 'ad_id' and 'label'
def gv_data_reader(googlevisionPath):
    gv_df = pd.read_json(googlevisionPath)
    gv_label=[description_extract(item) for item in gv_df['label_annotations'].apply(lambda m: json.loads(m) if m!="" else "")] # get the label
    result_df=pd.DataFrame({'ad_id':gv_df['ad_id'],'label':pd.Series(gv_label)})
    return result_df
    



########################################################################################
########################################### api function ###############################
########################################################################################

############################################ best_ad api ########################
##### input
#####       campaign id
#####           such as: http://localhost:5000/best_ad/v1.0/6060850543605,6059224129805,6059889803605,6059892762805,6059893654805,6057193158805,6055407354005,6053312447405,6051265864405,6051266212805,6049374949605,6035884102605,6035882865005,6034201296805,6034192331405,6033466101805,6032523746205,6033118372005,6033118372205,6032205731205,6032523746005,6032205730605,6031518441405,6031616546205,6031518442005,6031518441605,6031518441205,6031518441005,6032906211805,6032906682605,6032907243605,6032261374205,6032627603405,6032260651005,6032626832005,6032321919605,6032411410005,6032321128805,6032200009405,6031742821405,6031796276605,6031722773205
##### output 
#####       best and worst ad_id in json format
def best_ad(campaign_id):
    campaign_ids=campaign_id.split(',')
    campaign_ids = map(int, campaign_ids)
    campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
    result_df=find_best_ad(campaign_data)
    result_json=b[['ad_id','ranking']].to_dict(orient='records')

    return jsonify(result_json)



############################################ recommendation api ########################
##### input
#####       campaign id
#####           such as: http://localhost:5000/recommendation/v1.0/6060850543605,6059224129805,6059889803605,6059892762805,6059893654805,6057193158805,6055407354005,6053312447405,6051265864405,6051266212805,6049374949605,6035884102605,6035882865005,6034201296805,6034192331405,6033466101805,6032523746205,6033118372005,6033118372205,6032205731205,6032523746005,6032205730605,6031518441405,6031616546205,6031518442005,6031518441605,6031518441205,6031518441005,6032906211805,6032906682605,6032907243605,6032261374205,6032627603405,6032260651005,6032626832005,6032321919605,6032411410005,6032321128805,6032200009405,6031742821405,6031796276605,6031722773205
##### output 
#####       all segment and value recommendation in json format
def recommendation(campaign_id):
    campaign_ids=campaign_id.split(',')
    campaign_ids = map(int, campaign_ids)
    
    campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
    campaign_data=metric_generator(campaign_data)
    campaign_data=image_label_generator(campaign_data,100,gv_df_label)
    
    campaign_data_analysis=column_selector(campaign_data,gv_df_label)
    
    feature_and_importance_female = find_feature_and_importance(campaign_data_analysis,'gender','female',campaign_data,gv_df_label)
    feature_and_importance_male = find_feature_and_importance(campaign_data_analysis,'gender','male',campaign_data,gv_df_label)
    feature_and_importance_unknown = find_feature_and_importance(campaign_data_analysis,'gender','unknown',campaign_data,gv_df_label)
    
    feature_and_importance_1824 = find_feature_and_importance(campaign_data_analysis,'age','18-24',campaign_data,gv_df_label)
    feature_and_importance_2534 = find_feature_and_importance(campaign_data_analysis,'age','25-34',campaign_data,gv_df_label)
    feature_and_importance_3544 = find_feature_and_importance(campaign_data_analysis,'age','35-44',campaign_data,gv_df_label)
    feature_and_importance_4554 = find_feature_and_importance(campaign_data_analysis,'age','45-54',campaign_data,gv_df_label)
    feature_and_importance_5564 = find_feature_and_importance(campaign_data_analysis,'age','55-64',campaign_data,gv_df_label)
    feature_and_importance_65 = find_feature_and_importance(campaign_data_analysis,'age','65+',campaign_data,gv_df_label)
    
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



############################################ best_ad_by_segment api ########################
##### input
#####       campaign id
#####           such as: http://localhost:5000/best_ad_by_segment/v1.0/6060850543605,6059224129805,6059889803605,6059892762805,6059893654805,6057193158805,6055407354005,6053312447405,6051265864405,6051266212805,6049374949605,6035884102605,6035882865005,6034201296805,6034192331405,6033466101805,6032523746205,6033118372005,6033118372205,6032205731205,6032523746005,6032205730605,6031518441405,6031616546205,6031518442005,6031518441605,6031518441205,6031518441005,6032906211805,6032906682605,6032907243605,6032261374205,6032627603405,6032260651005,6032626832005,6032321919605,6032411410005,6032321128805,6032200009405,6031742821405,6031796276605,6031722773205
##### output 
#####       all segment and value recommendation in json format
def best_ad_by_segment(campaign_id):
    campaign_ids=campaign_id.split(',')
    campaign_ids = map(int, campaign_ids)
        
    campaign_data=mydata[mydata.campaign_id.isin(campaign_ids)]
    campaign_data=metric_generator(campaign_data)
    
    gv_df_label_filtered = gv_df_label[gv_df_label['ad_id'].isin(campaign_data.ad_id.unique())]
    
    best_ad_gender=find_best_ad_by_segment(campaign_data,'gender')
    best_ad_age=find_best_ad_by_segment(campaign_data,'age')
    
    gender_df=pd.melt(best_ad_gender,id_vars=['ad_id','impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
    age_df=pd.melt(best_ad_age,id_vars=['ad_id','impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
    df_adid = pd.concat([gender_df, age_df])
    df_adid=df_adid[['feature','value','ad_id']]
    df_adid = df_adid.reset_index(drop=True)
    result_df = pd.DataFrame(columns=['segment','value','ad_id','feature'])
    
    for index,ad_id in enumerate(df_adid['ad_id']):
        ad_feature=find_ad_feature(campaign_data, gv_df_label_filtered, [ad_id])
        ad_feature=ad_feature[['feature','value']].T
        ad_feature.columns = ad_feature.iloc[0]
        ad_feature.drop(ad_feature.index[0:1], inplace=True)
        ad_feature_dict=ad_feature.to_dict(orient='records')
        
        result_df.loc[index]=pd.Series({'segment':df_adid['feature'][index],'value':df_adid['value'][index],
                                  'ad_id':df_adid['ad_id'][index],'feature':ad_feature_dict})
    
    result_df['ad_id'] = map(int, result_df['ad_id'])
    result_json=result_df.to_dict(orient='records')
    return jsonify(result_json)







