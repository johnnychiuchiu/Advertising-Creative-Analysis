############################################ analysis_column_generator ############################################ 
##### input
#####       df: craetive table
#####       title_colname: the colname of title
#####       sub_title_colname: the colname of subtitle
#####       ad_content_colname: the colname of ad content
#####       brand_keywords: a vector of the brand keywords
##### output 
#####       a table with analysis columns that ready for analysis  
analysis_column_generator<- function(df,title_colname, sub_title_colname, ad_content_colname, brand_keywords){
  ###ad information related
  # title_length
  df$title_length<- nchar(df[,title_colname])
  df$title_length_interval <- cut(df$title_length, 
                                         breaks = c(0,as.numeric(quantile(df$title_length,0.33, na.rm = TRUE)),
                                                    as.numeric(quantile(df$title_length,0.66, na.rm = TRUE)),Inf), 
                                         labels = c("short","medium","long"), 
                                         right = FALSE) 
  
  
  # title_brand
  df$title_brand<- ifelse(grepl(brand_keywords,df[,title_colname]),'yes','no')
  
  # sub_title_length
  df$sub_title_length<- nchar(df[,sub_title_colname])
  df$sub_title_length_interval <- cut(df$sub_title_length, 
                                             breaks = c(0,as.numeric(quantile(df$sub_title_length, 0.33, na.rm = TRUE)),
                                                        as.numeric(quantile(df$sub_title_length, 0.66, na.rm = TRUE)),Inf), 
                                             labels = c("short","medium","long"), 
                                             right = FALSE) 
  
  # sub_title_brand
  df$sub_title_brand<- ifelse(grepl(brand_keywords,df[,sub_title_colname]),'yes','no')
  
  # ad_content_length
  df$ad_content_length<- nchar(df[,ad_content_colname])
  df$ad_content_length_interval <- cut(df$ad_content_length, 
                                              breaks = c(0,as.numeric(quantile(df$ad_content_length,0.33, na.rm = TRUE)),
                                                         as.numeric(quantile(df$ad_content_length,0.66, na.rm = TRUE)),Inf), 
                                              labels = c("short","medium","long"), 
                                              right = FALSE) 
  
  # ad_content_brand
  df$ad_content_brand<- ifelse(grepl(brand_keywords,df[,ad_content_colname]),'yes','no')
  return(df)
}

############################################ find_best_feature ############################################ 
##### input
#####       analysis_df: a table with all columns that are ready for analysis
##### output 
#####       a dataframe that provides the best feature for the targeted dimension, such as {gender, age}
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

############################################ find_best_ad ############################################ 
##### input
#####       df: craetive table
##### output 
#####       a dataframe that provides the best ad for the targeted dimension, such as {gender, age}
find_best_ad<- function(df){
  dimension<-c('gender','age')
  impression_threshold=1000
  
  grp_cols <- c(dimension,'ad_id') # Columns you want to group by
  dots <- lapply(grp_cols, as.symbol)# Convert character vector to list of symbols
  
  final_df<-df %>% group_by_(.dots=dots) %>% 
    dplyr::summarise(total_impression=sum(impression), total_click=sum(link_clicks)) %>% 
    filter(total_impression > impression_threshold) %>%
    mutate(CTR=total_click/total_impression) %>%
    group_by_(.dots=dots2) %>% slice(which.max(CTR)) 

  return(final_df)
}
