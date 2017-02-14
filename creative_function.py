import pandas as pd
import os
import numpy as np
import copy

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


############################################ find_best_ad ############################################ 
##### input
#####       df: craetive table
##### output 
#####       a dataframe that provides the best ad for the targeted dimension, such as {gender, age}

def find_best_ad(df):
    dimension=['gender','age']
    impression_threshold=1000
    normal_impression_threshold=10

    df['link_clicks']=df['link_clicks']+1
    df=df.assign(CTR=df.link_clicks/df.impression*1.0) 
    df=df.assign(CPC=df.spend/df.link_clicks*1.0) 
    df_filter=df[ (df.impression>normal_impression_threshold) & (df.CTR<1) & (df.link_clicks!=1)]
    
    cpc_mean=df_filter.CPC.mean()
    ctr_mean=df_filter.CTR.mean()
    
    df.CPC[ (df.impression<normal_impression_threshold) | (df.link_clicks==1) | (df.CTR>1) ]= cpc_mean
    df.CTR[ (df.impression<normal_impression_threshold) | (df.link_clicks==1) | (df.CTR>1) ]= ctr_mean
    df['score']=webClickScore(df.CPC, df.CTR, df.link_clicks)
        
    f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
    temp=df.groupby(dimension+['ad_id']).agg(f).reset_index()
    temp=temp[temp.impression>impression_threshold]
    temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
    temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0) 
        
    idx=temp.groupby(['gender','age'])['score'].transform(max) == temp['score']
    temp2=temp[idx]
    temp2['score'] = temp2.score.round(1)
    return temp2


############################################ find_ad_feature ############################################ 
##### input
#####       df: a data frame which columns contains ad_id and columns ready for analysis 
#####       ad_id
##### output 
#####       a dataframe contains the feature of this ad_id

def find_ad_feature(df,ad_id):
    dimension=['gender','age']
    drop_columns=dimension+['account_id','account_name','ad_set_id','campaign_id','creative_id','image_link','interest','page_id','page_name','gender','age','impression','link_clicks','spend','title_brand','sub_title_brand','ad_content_length','title','subtitle','message']
    df=df.drop(drop_columns, axis=1)
    df_adid=df[df['ad_id'].isin(ad_id)]
    df_adid=df_adid.drop_duplicates()
    df_adid=pd.melt(df_adid,id_vars=['ad_id'])
    df_adid['sort'] = pd.Categorical(df_adid['ad_id'], ad_id)
    df_adid=df_adid.sort_values(by="sort")
    del df_adid['sort']
    
    return df_adid 
    
    


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



############################################ find_best_feature ############################################ 
##### input
#####       analysis_df: a table with all columns that are ready for analysis
##### output 
#####       a dataframe that provides the best feature for the targeted dimension, such as {gender, age}
def find_best_feature(analysis_df):
    final_feature = pd.DataFrame()
    dimension=['gender','age']
    impression_threshold=1000
    normal_impression_threshold=10
    
    analysis_df['link_clicks']=analysis_df['link_clicks']+1
    analysis_df=analysis_df.assign(CTR=analysis_df.link_clicks/analysis_df.impression*1.0) 
    analysis_df=analysis_df.assign(CPC=analysis_df.spend/analysis_df.link_clicks*1.0) 
    analysis_df_filter=analysis_df[ (analysis_df.impression>normal_impression_threshold) & (analysis_df.CTR<1) & (analysis_df.link_clicks!=1)]
        
    cpc_mean=analysis_df_filter.CPC.mean()
    ctr_mean=analysis_df_filter.CTR.mean()
        
    analysis_df.CPC[ (analysis_df.impression<normal_impression_threshold) | (analysis_df.link_clicks==1) | (analysis_df.CTR>1) ]= cpc_mean
    analysis_df.CTR[ (analysis_df.impression<normal_impression_threshold) | (analysis_df.link_clicks==1) | (analysis_df.CTR>1) ]= ctr_mean
    analysis_df['score']=webClickScore(analysis_df.CPC, analysis_df.CTR, analysis_df.link_clicks)
    
    for name in analysis_df:
        if name not in dimension+['impression','link_clicks','spend','CTR','CPC','score']:
            print name
            f = {'score': np.mean, 'impression': np.sum, 'link_clicks': np.sum, 'spend': np.sum }    
            temp=analysis_df.groupby(dimension+[name]).agg(f).reset_index()
            temp=temp[temp.impression>impression_threshold]
            temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
            temp=temp.assign(CPC=temp.spend/temp.link_clicks*1.0) 
                
            idx= temp.groupby(dimension)['score'].transform(max) == temp['score']
            temp2=temp[idx]
            
            temp_reshape=pd.melt(temp2,id_vars=dimension+['impression','link_clicks','spend','CTR','CPC','score'],var_name='feature')
            temp_reshape=temp_reshape.loc[:,dimension+['feature','value','impression','link_clicks','spend','CTR','CPC','score']]
            
            final_feature=pd.concat([final_feature, temp_reshape])
            
    final_feature=final_feature.sort_values(by=dimension, ascending=True)

    return final_feature

############################################ feature_importance ############################################ 
##### goal: find the feature importance for each dimension
##### input
#####       analysis_df: a table with all columns that are ready for analysis
##### output 
#####       a dataframe that provides the feature importance ranking for the targeted dimension, such as {gender, age}

def feature_importance(analysis_df):
    impression_threshold=1000
    normal_impression_threshold=10
    
    analysis_df['link_clicks']=analysis_df['link_clicks']+1
    analysis_df=analysis_df.assign(CTR=analysis_df.link_clicks/analysis_df.impression*1.0) 
    analysis_df=analysis_df.assign(CPC=analysis_df.spend/analysis_df.link_clicks*1.0) 
    analysis_df_filter=analysis_df[ (analysis_df.impression>normal_impression_threshold) & (analysis_df.CTR<1) & (analysis_df.link_clicks!=1)]
            
    cpc_mean=analysis_df_filter.CPC.mean()
    ctr_mean=analysis_df_filter.CTR.mean()
        
    analysis_df.CPC[ (analysis_df.impression<normal_impression_threshold) | (analysis_df.link_clicks==1) | (analysis_df.CTR>1) ]= cpc_mean
    analysis_df.CTR[ (analysis_df.impression<normal_impression_threshold) | (analysis_df.link_clicks==1) | (analysis_df.CTR>1) ]= ctr_mean
    analysis_df['score']=webClickScore(analysis_df.CPC, analysis_df.CTR, analysis_df.link_clicks)
    
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


############################################ best_feature_and_importance ############################################ 
##### input
#####       analysis_df: a table with all columns that are ready for analysis
##### output 
#####       a dataframe that provides the best feature and importance for the targeted dimension, such as {gender, age}

def best_feature_and_importance(analysis_df,gender,age):
    analysis_df=analysis_df[(analysis_df.gender==gender) & (analysis_df.age==age)]
    best_feature=find_best_feature(analysis_df)
    importance=feature_importance(analysis_df)
    
    feature_and_importance=pd.merge(best_feature, importance, on=['feature'], how='left')
    feature_and_importance=feature_and_importance.sort_values(by='importance', ascending=False)
    
    return feature_and_importance
    
