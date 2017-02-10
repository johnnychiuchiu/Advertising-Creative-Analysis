############################################ webClickScore ############################################ 
##### input
#####       cpc
#####       ctr
#####       clicks
##### output 
#####       a pandas dataframe column 


def webClickScore(cpc, ctr, clicks):
    scoreSum = (cpc - cpc.mean()) / (cpc.max() - cpc.min()) + (ctr - ctr.mean()) / (ctr.max() - ctr.min()) + (clicks - clicks.mean()) / (clicks.max() - clicks.min())
    return 80 + 32 * (scoreSum - scoreSum.mean()) / (scoreSum.max() - scoreSum.min())


############################################ find_best_ad ############################################ 
##### input
#####       df: craetive table
##### output 
#####       a dataframe that provides the best ad for the targeted dimension, such as {gender, age}

def find_best_ad(df):
    dimension=['gender','age']
    impression_threshold=1000
    
    temp=etungo_df.groupby(dimension+['ad_id'])[['impression','link_clicks']].sum().reset_index()
    temp2=temp[temp.impression>impression_threshold]
    temp3=temp2.assign(CTR=temp2.link_clicks/temp2.impression*1.0) 
    idx= temp3.groupby(['gender','age'])['CTR'].transform(max) == temp3['CTR']
    temp4=temp3[idx]
    return temp4



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
        if name not in dimension+['impression','link_clicks']:
            temp=analysis_df.groupby(dimension+[name])[['impression','link_clicks']].sum().reset_index()
            temp2=temp[temp.impression>impression_threshold]
            temp3=temp2.assign(CTR=temp2.link_clicks/temp2.impression*1.0) 
            idx= temp3.groupby(dimension)['CTR'].transform(max) == temp3['CTR']
            temp4=temp3[idx]
        
            temp_reshape=pd.melt(temp4,id_vars=dimension+['impression','link_clicks','CTR'])
            temp_reshape=temp_reshape.loc[:,dimension+['variable','value','impression','link_clicks','CTR']]
        
            final_feature=pd.concat([final_feature, temp_reshape])
        
    final_feature=final_feature.sort_values(by=dimension, ascending=True)

    return final_feature


find_best_feature<- function(analysis_df){
  library(reshape)
  final_feature=data.frame()
  dimension<-c('gender','age')
  impression_threshold=1000
  
  for(name in colnames(analysis_df)){  # 
    if(!(name %in% c('gender','age','impression','link_clicks'))){
      grp_cols <- c(dimension, name) # Columns you want to group by
      dots <- lapply(grp_cols, as.symbol)# Convert character vector to list of symbols
      dots2 <- lapply(dimension, as.symbol)# Convert character vector to list of symbols
      
      temp_feature_df<-analysis_df %>% group_by_(.dots=dots) %>% 
        dplyr::summarise(total_impression=sum(impression), total_click=sum(link_clicks)) %>% 
        filter(total_impression > impression_threshold) %>%
        mutate(CTR=total_click/total_impression) %>%
        group_by_(.dots=dots2) %>% slice(which.max(CTR)) 
      
      temp_reshape <- melt(temp_feature_df, id=c(dimension,"total_impression",'total_click','CTR')) %>% 
        select(gender, age, variable, value, total_impression, total_click, CTR)
      
      final_feature<- rbind(final_feature,temp_reshape) 
    }
  }
  
  final_feature<- final_feature %>% arrange(gender, age)
  return(final_feature)
}





