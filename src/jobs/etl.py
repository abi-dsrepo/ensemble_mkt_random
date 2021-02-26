import pandas as pd
import numpy as np
import os
from pandas import DataFrame

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
from src.utils.utils import get_logger


class ETL:
    def __init__(self) -> None:
        self.input =  os.getcwd()
        self.output = os.path.join(self.input, 'src/output')
        self.logger = get_logger('ETL')
        self.target = 'is_returning_customer'

    def load_data(self) -> DataFrame:
        """Loads the csv files

        Returns:
            [Dataframe]: [Returns label and orders data]
        """
        self.logger.info("Reading csv data..")
        dflabel=pd.read_csv(os.path.join(self.input, "src/data/machine_learning_challenge_labeled_data.csv.gz"))
        dforders=pd.read_csv(os.path.join(self.input, "src/data/machine_learning_challenge_order_data.csv.gz"))
        return dflabel, dforders.drop_duplicates(keep=False)

    @staticmethod
    def day_time(x):
        """Adds time of the day as a predictor variable

        Args:
            x ([Row]): [Rows of a dataframe]

        Returns:
            [str]: [Based on the time, returns night, day or evening]
        """
        if x['order_hour'] < 8:
            return 'night'
        elif (x['order_hour'] >=8) & (x['order_hour'] <16):
            return 'day'
        elif x['order_hour'] >= 16:
            return 'evening'

    def additional_cols(self, dforders):
        self.logger.info("Adding time related additional columns")
        dforders['Isnullrank']=np.where(dforders['customer_order_rank'].isnull(),1,0)
        dforders['Isnonzerovoucher']=np.where(dforders['voucher_amount']>0,1,0)
        
        dforders.order_date = pd.to_datetime(dforders.order_date)
        dforders['year'] = dforders['order_date'].dt.year
        dforders['month'] = dforders.order_date.apply(lambda x: x.strftime('%Y-%m'))
        dforders['week_number'] = dforders.order_date.dt.week
        dforders['week_day'] = dforders.order_date.dt.day_name()
        dforders['hour_class'] = dforders.apply(self.day_time, axis=1)
        return dforders


    def separate_by_year(self, dforders):
        self.logger.info("Adding year based columns..")
        dfnew=pd.get_dummies(dforders, columns=['hour_class'])
        dfnew['Is2017']=np.where(dfnew['year']==2017,1,0)
        dfnew['Is2016']=np.where(dfnew['year']==2016,1,0)
        dfnew['Is2015']=np.where(dfnew['year']==2015,1,0)
        return dfnew

    def etl_initial(self, dfnew, dflabel):
        self.logger.info("Performing initial ETL...")
        df1=dfnew.groupby('customer_id',as_index=False).agg(['min','max','median'])[['customer_order_rank','voucher_amount','amount_paid','delivery_fee']]
        df2=dfnew.groupby(['customer_id']).agg({'order_date': [np.min,np.max]})
        df3=dfnew.groupby('customer_id',as_index=False).agg(['sum'])[['customer_order_rank','Isnullrank','Is2017','Is2016','Is2015','is_failed','Isnonzerovoucher','voucher_amount','amount_paid','delivery_fee','hour_class_day','hour_class_evening','hour_class_night']]
        df4=dfnew.groupby('customer_id',as_index=False).agg(['nunique'])[['restaurant_id','city_id','payment_id','platform_id','transmission_id','year','week_day','month','order_date']] 
        df5=dfnew.groupby(['customer_id']).size().reset_index() 
        df5.columns=['customer_id','totaltranscations']
        dforderssummary=df1.merge(df2,on='customer_id').merge(df3,on='customer_id').merge(df4,on='customer_id').merge(df5,on='customer_id').merge(dflabel,on='customer_id')
        return dforderssummary

    def rename_cols(self, dforderssummary):
        self.logger.info("Renaming columns..")
        cols = dforderssummary.iloc[:,1:-2].columns
        newcols = []
        for x in cols: 
            newcols.append('_'.join(list(x)))
        newcols
        newdf = dforderssummary.copy(True)
        for i,j in zip(dforderssummary.iloc[:, 1:-2], newcols):
            newdf.rename(columns={i : j}, inplace=True)
        return newdf


    def etl_phase2(self, newdf):
        self.logger.info("Performing ETL phase 2...")
        newdf['daydiff']=(newdf['order_date_amax']-newdf['order_date_amin'])/np.timedelta64(1, 'D')
        newdf['samedaytransaction']=np.where((newdf.order_date_nunique == 1) & (newdf.totaltranscations>1),1,0)
        newdf['moretrans']=np.where(newdf.order_date_nunique < newdf.totaltranscations ,1,0)

        newdf['customer_order_rank_min'] = newdf['customer_order_rank_min'].replace(np.nan, 0)
        newdf['customer_order_rank_max'] = newdf['customer_order_rank_max'].replace(np.nan, 0)
        newdf['customer_order_rank_median'] = newdf['customer_order_rank_median'].replace(np.nan, 0)

        newdf['order_date_amax'] = pd.to_datetime(newdf.order_date_amax)
        maxdate = max(newdf.order_date_amax)
        mindate = min(newdf.order_date_amax)
        newdf['recenencyscore']= 1- ((maxdate - newdf['order_date_amax']).dt.days / int((maxdate-mindate).total_seconds()/(60*60*24)))

        newdf['Is2017binary']=np.where(newdf['Is2017_sum']>0,1,0)
        newdf['Is2016binary']=np.where(newdf['Is2016_sum']>0,1,0)
        newdf['Is2015binary']=np.where(newdf['Is2015_sum']>0,1,0)

        cols=['Is2015binary', 'Is2016binary', 'Is2017binary']
        newdf['yearinfo']=newdf[cols].dot(newdf[cols].columns + ';').str.rstrip(';')

        newdf['totaltransbinary']=np.where(newdf['totaltranscations']>1,1,0)
        newdf.order_date_amax = pd.to_datetime(newdf.order_date_amax)
        newdf['yearmax'] = newdf['order_date_amax'].dt.year

        newdf.order_date_amin = pd.to_datetime(newdf.order_date_amin)
        newdf['yearmin'] = newdf['order_date_amin'].dt.year
        return newdf

    def split_by_year(self, newdf):
        self.logger.info("Splitting by year...")
        df_2017=newdf[newdf.yearmax==2017]
        df_2016=newdf[newdf.yearmax==2016]
        df_2015=newdf[newdf.yearmax==2015]
        return df_2015, df_2016, df_2017


    def select_imp_cols(self, df):
        self.logger.info("Selecting only required columns..")
        vardf = df[[i for i in df.columns if i not in [self.target, 'recenencyscore']]].var().sort_values(ascending=False)
        select = vardf.shape[0]//2
        final_cols = list(vardf.iloc[:select].index) + ['recenencyscore', self.target]
        return final_cols


    def core(self, save_file=False):
        dflabel, dforders = self.load_data()
        dforders1 = self.additional_cols(dforders)
        dfnew = self.separate_by_year(dforders1)
        dforderssummary = self.etl_initial(dfnew, dflabel)
        newdf = self.rename_cols(dforderssummary)
        newdf1 = self.etl_phase2(newdf)
        #a, b, c = self.split_by_year(newdf1)
        final_cols = self.select_imp_cols(newdf1)
        if save_file:
            if not os.path.exists(self.output):
                os.makedirs(self.output)
            newdf1[final_cols].to_csv(os.path.join(self.output, 'etl_all_data.csv'), index=False)
            newdf1.to_csv(os.path.join(self.output, 'etl_all_columns.csv'), index=False)
        return newdf1[final_cols]
