############################################ webClickScore ############################################ 
##### input
#####       cpc
#####       ctr
#####       clicks
##### output 
#####       a pandas dataframe column 


def webClickScore(cpc, ctr, clicks):
    scoreSum =  (ctr - ctr.mean()) / (ctr.max() - ctr.min()) + (clicks - clicks.mean()) / (clicks.max() - clicks.min()) - (cpc - cpc.mean()) / (cpc.max() - cpc.min())
    return 80 + 32 * (scoreSum - scoreSum.mean()) / (scoreSum.max() - scoreSum.min())


############################################ find_best_ad ############################################ 
##### input
#####       df: craetive table
##### output 
#####       a dataframe that provides the best ad for the targeted dimension, such as {gender, age}

def find_best_ad(df):
    dimension=['gender','age']
    impression_threshold=1000
    
    temp=df.groupby(dimension+['ad_id'])[['impression','link_clicks','spent']].sum().reset_index()
    temp=temp[temp.impression>impression_threshold]
    temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
    temp=temp.assign(CPC=temp.spent/temp.link_clicks*1.0) 
    temp.CPC=temp.CPC.replace(np.inf, np.nan)
    temp.CPC=temp.CPC.replace(np.nan, temp.CPC.max())
    temp['score']=webClickScore(temp.CPC, temp.CTR, temp.link_clicks)

    idx=temp.groupby(['gender','age'])['score'].transform(max) ==temp['score']
    temp2=temp[idx]
    return temp2



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
    title_max_length=72
    sub_title_max_length=104
    ad_content_max_length=90
    group_names = ['short','medium','long']
    
    # title_length
    df[title_colname] = df[title_colname].str.strip()
    df['title_length'] = df[title_colname].str.len()
    bins = [0, title_max_length*0.33, title_max_length*0.66, title_max_length]
    df['title_length_interval'] = pd.cut(df['title_length'], bins, labels=group_names)

    # title_brand
    df['title_brand']=df[title_colname].str.contains(brand_keywords.decode('utf-8'))
    
    # sub_title_length
    df[sub_title_colname] = df[sub_title_colname].str.strip()
    df['sub_title_length'] = df[sub_title_colname].str.len()    

    bins = [0, sub_title_max_length*0.33, sub_title_max_length*0.66, sub_title_max_length]
    df['sub_title_length_interval'] = pd.cut(df['sub_title_length'], bins, labels=group_names)

    # sub_title_brand
    df['sub_title_brand']=df[sub_title_colname].str.contains(brand_keywords.decode('utf-8'))
    
    # ad_content_length
    df[ad_content_colname] = df[ad_content_colname].str.strip()
    df['ad_content_length'] = df[ad_content_colname].str.len()    

    bins = [0, ad_content_max_length*0.33, ad_content_max_length*0.66, ad_content_max_length]
    df['ad_content_length_interval'] = pd.cut(df['ad_content_length'], bins, labels=group_names)

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
    for name in analysis_df:
        if name not in dimension+['impression','link_clicks','spent']:
            temp=analysis_df.groupby(dimension+[name])[['impression','link_clicks','spent']].sum().reset_index()
            temp=temp[temp.impression>impression_threshold]
            temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
            temp=temp.assign(CPC=temp.spent/temp.link_clicks*1.0) 
            temp.CPC=temp.CPC.replace(np.inf, np.nan)
            temp.CPC=temp.CPC.replace(np.nan, temp.CPC.max())
            temp['score']=webClickScore(temp.CPC, temp.CTR, temp.link_clicks)
            
            idx= temp.groupby(dimension)['score'].transform(max) == temp['score']
            temp2=temp[idx]
        
            temp_reshape=pd.melt(temp2,id_vars=dimension+['impression','link_clicks','spent','CTR','CPC','score'])
            temp_reshape=temp_reshape.loc[:,dimension+['variable','value','impression','link_clicks','spent','CTR','CPC','score']]
        
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
    import copy
    feature=copy.deepcopy(analysis_df)
    feature=feature.drop(['impression','link_clicks','spent'], axis=1)
    importance_df=pd.DataFrame({'feature':feature.columns,
                            'max_impression':0,'max_click':0,'max_spent':0,'max_ctr':0,'max_cpc':0,'max_score':0,
                            'min_impression':0,'min_click':0,'min_spent':0,'min_ctr':0,'min_cpc':0,'min_score':0,
                            'importance':0})
    impression_threshold=1000

    for name in feature.columns:
        print name
        temp=analysis_df.groupby([name])[['impression','link_clicks','spent']].sum().reset_index()
        temp=temp[temp.impression>impression_threshold]
        temp=temp.assign(CTR=temp.link_clicks/temp.impression*1.0) 
        temp=temp.assign(CPC=temp.spent/temp.link_clicks*1.0) 
        temp.CPC=temp.CPC.replace(np.inf, np.nan)
        temp.CPC=temp.CPC.replace(np.nan, temp.CPC.max())
        temp['score']=webClickScore(temp.CPC, temp.CTR, temp.link_clicks)
    
        if temp.shape[0]==1:
            importance_df=importance_df[importance_df.feature!=name]
        else:
#            max_score= temp.score.max()
#            min_score= temp.score.min()
#            importance_df.importance[importance_df.feature==name]= max_score-min_score
#            importance_df.max_score[importance_df.feature==name]= max_score
#            importance_df.min_score[importance_df.feature==name]= min_score
#            importance_df.max_impression[importance_df.feature==name]= temp.impression[temp.score==max_score].values
#            importance_df.min_impression[importance_df.feature==name]= temp.impression[temp.score==min_score].values
#            importance_df.max_click[importance_df.feature==name]= temp.link_clicks[temp.score==max_score].values
#            importance_df.min_click[importance_df.feature==name]= temp.link_clicks[temp.score==min_score].values
#            importance_df.max_spent[importance_df.feature==name]= temp.spent[temp.score==max_score].values
#            importance_df.min_spent[importance_df.feature==name]= temp.spent[temp.score==min_score].values
#            importance_df.max_ctr[importance_df.feature==name]= temp.CTR[temp.score==max_score].values
#            importance_df.min_ctr[importance_df.feature==name]= temp.CTR[temp.score==min_score].values
#            importance_df.max_cpc[importance_df.feature==name]= temp.CPC[temp.score==max_score].values
#            importance_df.min_cpc[importance_df.feature==name]= temp.CPC[temp.score==min_score].values
            
            max_score= temp.CTR.max()
            min_score= temp.CTR.min()
            importance_df.importance[importance_df.feature==name]= temp.score[temp.CTR==max_score].values-temp.score[temp.CTR==min_score].values
            importance_df.max_score[importance_df.feature==name]= temp.score[temp.CTR==max_score].values
            importance_df.min_score[importance_df.feature==name]= temp.score[temp.CTR==min_score].values
            importance_df.max_impression[importance_df.feature==name]= temp.impression[temp.CTR==max_score].values
            importance_df.min_impression[importance_df.feature==name]= temp.impression[temp.CTR==min_score].values
            importance_df.max_click[importance_df.feature==name]= temp.link_clicks[temp.CTR==max_score].values
            importance_df.min_click[importance_df.feature==name]= temp.link_clicks[temp.CTR==min_score].values
            importance_df.max_spent[importance_df.feature==name]= temp.spent[temp.CTR==max_score].values
            importance_df.min_spent[importance_df.feature==name]= temp.spent[temp.CTR==min_score].values
            importance_df.max_ctr[importance_df.feature==name]= temp.CTR[temp.CTR==max_score].values
            importance_df.min_ctr[importance_df.feature==name]= temp.CTR[temp.CTR==min_score].values
            importance_df.max_cpc[importance_df.feature==name]= temp.CPC[temp.CTR==max_score].values
            importance_df.min_cpc[importance_df.feature==name]= temp.CPC[temp.CTR==min_score].values
    
    importance_df=importance_df.sort_values(by=['importance'], ascending=False) 
    importance_df['percentage']=  importance_df.importance / importance_df.importance.sum()
       
    return importance_df    
  