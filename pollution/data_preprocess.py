import pandas as pd
import warnings

warnings.filterwarnings("ignore") #Hide messy Numpy warnings

"""
Part 1: merge air_quality data and feature data (data before 2018-01-31) 
"""

# dataframe_aq = pd.read_csv('data/raw_data/weather_data_kddCup/beijing_17_18_aq.csv')
# dataframe_meo = pd.read_csv('data/raw_data/weather_data_kddCup/beijing_17_18_meo.csv')
#
# print (dataframe_aq.stationId.unique())
# print (dataframe_meo.station_id.unique())
#
# # print(dataframe_meo.columns.values)
#
# dataframe_aq['stationId'] = dataframe_aq['stationId'].str[:-3] # drop last three charactors (_aq)
# dataframe_meo['stationId'] = dataframe_meo['station_id'].str[:-4] # drop last four charactors (_meo), rename column to 'stationId'
#
#
# new_df = pd.merge(dataframe_meo, dataframe_aq, on=['stationId','utc_time'])
# new_df = new_df.drop('station_id', axis=1)
#
# print (new_df.head(5))
#
# print (len(dataframe_aq))
# print (len(dataframe_meo))
#
# print (len(new_df))
# new_df.to_csv('data/preprocessed_data/bj_merged.csv', index=0)

"""
Part 2: Save merged data by different area 
"""

data_all = pd.read_csv('data/preprocessed_data/bj_merged.csv')
# print (data_all.stationId.unique())

shunyi_data = data_all.loc[data_all['stationId'] == 'shunyi']
miyun_data = data_all.loc[data_all['stationId'] == 'miyun']
huairou_data = data_all.loc[data_all['stationId'] == 'huairou']
pinggu_data = data_all.loc[data_all['stationId'] == 'pinggu']
tongzhou_data = data_all.loc[data_all['stationId'] == 'tongzhou']
pingchang_data = data_all.loc[data_all['stationId'] == 'pingchang']
mentougou_data = data_all.loc[data_all['stationId'] == 'mentougou']
daxing_data = data_all.loc[data_all['stationId'] == 'daxing']
fangshan_data = data_all.loc[data_all['stationId'] == 'fangshan']



### remove rows to keep all of the files have the same length
# a= shunyi_data[~shunyi_data.utc_time.isin(pinggu_data.utc_time)].dropna()

shunyi_data.drop(shunyi_data.tail(3).index,inplace=True)
miyun_data.drop(miyun_data.tail(3).index,inplace=True)
huairou_data.drop(huairou_data.tail(2).index,inplace=True)
tongzhou_data.drop(tongzhou_data.tail(2).index,inplace=True)
pingchang_data.drop(pingchang_data.tail(3).index,inplace=True)
mentougou_data.drop(mentougou_data.tail(1).index,inplace=True)
daxing_data.drop(daxing_data.tail(3).index,inplace=True)
fangshan_data.drop(fangshan_data.tail(3).index,inplace=True)

print (len(shunyi_data), len(miyun_data), len(huairou_data), len(pinggu_data), len(tongzhou_data),
       len(pingchang_data), len(mentougou_data), len(daxing_data),len(fangshan_data))
# shunyi_data.to_csv('data/preprocessed_data/bj_shunyi.csv', index=0)
# miyun_data.to_csv('data/preprocessed_data/bj_miyun.csv', index=0)
# huairou_data.to_csv('data/preprocessed_data/bj_huairou.csv', index=0)
# pinggu_data.to_csv('data/preprocessed_data/bj_pinggu.csv', index=0)
# tongzhou_data.to_csv('data/preprocessed_data/bj_tongzhou.csv', index=0)
# pingchang_data.to_csv('data/preprocessed_data/bj_changping.csv', index=0)
# mentougou_data.to_csv('data/preprocessed_data/bj_mentougou.csv', index=0)
# daxing_data.to_csv('data/preprocessed_data/bj_daxing.csv', index=0)
# fangshan_data.to_csv('data/preprocessed_data/bj_fangshan.csv', index=0)

"""
Part 3: merge air_quality data and feature data (data in 2018-05) 
"""

# dataframe_aq = pd.read_csv('data/raw_data/weather_data_kddCup/future_data_from_api/bj_airquality_2018_05.csv')
# dataframe_meo = pd.read_csv('data/raw_data/weather_data_kddCup/future_data_from_api/bj_meteorology_2018_05.csv')
#
# print (dataframe_aq.station_id.unique())
# print (dataframe_meo.station_id.unique())
# dataframe_aq.drop('id', axis=1, inplace=True)
# dataframe_meo.drop('id', axis=1, inplace=True)
#
# dataframe_aq['station_id'] = dataframe_aq['station_id'].str[:-3] # drop last three charactors of station_id (_aq)
# dataframe_meo['station_id'] = dataframe_meo['station_id'].str[:-4] # drop last four charactors (_meo), rename column to 'stationId'
#
# # rename column to keep the same name with history(train) data
# dataframe_aq.rename(columns={'time':'utc_time','PM25_Concentration':'PM2.5','PM10_Concentration':'PM10',
#                              'NO2_Concentration':'NO2','CO_Concentration':'CO','O3_Concentration':'O3','SO2_Concentration':'SO2'}, inplace=True)
# dataframe_meo.rename(columns={'time':'utc_time','PM25_Concentration':'PM2.5','PM10_Concentration':'PM10',
#                              'NO2_Concentration':'NO2','CO_Concentration':'CO','O3_Concentration':'O3','SO2_Concentration':'SO2'},  inplace=True)
#
#
# new_df = pd.merge(dataframe_meo, dataframe_aq, on=['station_id','utc_time'])
# new_df.rename(columns={'station_id':'stationId'},  inplace=True)
#
# # add 'longitude' and 'latitude' column
# data = pd.read_csv('data/preprocessed_data/bj_merged.csv')
# data2 = data[['longitude', 'latitude','stationId']]
# data2 = data2.drop_duplicates()
# print (data2)
# new_df = pd.merge(data2,new_df, on=['stationId'])
# # print (new_df.head(5))
#
# # re-arrange columns
# new_df = new_df[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed',
#                  'weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]
#
# new_df.to_csv('data/preprocessed_data/test/bj_201805.csv', index=0)

"""
Part 4: Save merged data by different area (data in 2018-05)
"""

data_all = pd.read_csv('data/preprocessed_data/test/bj_201805.csv')
# print (data_all.stationId.unique())

shunyi_data = data_all.loc[data_all['stationId'] == 'shunyi']
miyun_data = data_all.loc[data_all['stationId'] == 'miyun']
huairou_data = data_all.loc[data_all['stationId'] == 'huairou']
pinggu_data = data_all.loc[data_all['stationId'] == 'pinggu']
tongzhou_data = data_all.loc[data_all['stationId'] == 'tongzhou']
pingchang_data = data_all.loc[data_all['stationId'] == 'pingchang']
mentougou_data = data_all.loc[data_all['stationId'] == 'mentougou']
daxing_data = data_all.loc[data_all['stationId'] == 'daxing']
fangshan_data = data_all.loc[data_all['stationId'] == 'fangshan']


miyun_data.drop(miyun_data.tail(2).index,inplace=True)
huairou_data.drop(huairou_data.tail(2).index,inplace=True)
pinggu_data.drop(pinggu_data.tail(2).index,inplace=True)
tongzhou_data.drop(tongzhou_data.tail(3).index,inplace=True)
pingchang_data.drop(pingchang_data.tail(3).index,inplace=True)
mentougou_data.drop(mentougou_data.tail(2).index,inplace=True)
daxing_data.drop(daxing_data.tail(1).index,inplace=True)
fangshan_data.drop(fangshan_data.tail(1).index,inplace=True)

print (len(shunyi_data), len(miyun_data), len(huairou_data), len(pinggu_data), len(tongzhou_data),
       len(pingchang_data), len(mentougou_data), len(daxing_data),len(fangshan_data))

# shunyi_data.to_csv('data/preprocessed_data/test/bj_shunyi_201805.csv', index=0)
# miyun_data.to_csv('data/preprocessed_data/test/bj_miyun_201805.csv', index=0)
# huairou_data.to_csv('data/preprocessed_data/test/bj_huairou_201805.csv', index=0)
# pinggu_data.to_csv('data/preprocessed_data/test/bj_pinggu_201805.csv', index=0)
# tongzhou_data.to_csv('data/preprocessed_data/test/bj_tongzhou_201805.csv', index=0)
# pingchang_data.to_csv('data/preprocessed_data/test/bj_changping_201805.csv', index=0)
# mentougou_data.to_csv('data/preprocessed_data/test/bj_mentougou_201805.csv', index=0)
# daxing_data.to_csv('data/preprocessed_data/test/bj_daxing_201805.csv', index=0)
# fangshan_data.to_csv('data/preprocessed_data/test/bj_fangshan_201805.csv', index=0)


"""
Part 5: Grid data (data before 2018-01-31)
"""

# # daxing grid data - training
# history_data_grid = pd.read_csv('data/raw_data/weather_data_kddCup/Beijing_historical_meo_grid.csv')
# daxing_data = pd.read_csv('data/preprocessed_data/bj_daxing.csv')
#
# history_data_grid.rename(columns={'wind_speed/kph':'wind_speed'},  inplace=True)
# history_data_grid.rename(columns={'stationName':'stationId'},  inplace=True)
#
# history_data_grid['PM2.5'] = 0
# history_data_grid['PM10'] = 0
# history_data_grid['NO2'] = 0
# history_data_grid['CO'] = 0
# history_data_grid['O3'] = 0
# history_data_grid['SO2'] = 0
# # array = ['beijing_grid_280', 'beijing_grid_301', 'beijing_grid_259']
# # daxing_grid_data = history_data_grid.loc[history_data_grid['stationName'].isin(array)]
#
# daxing_grid_data280 = history_data_grid.loc[history_data_grid['stationId'] == 'beijing_grid_280']
#
# daxing_grid_data259 = history_data_grid.loc[history_data_grid['stationId'] == 'beijing_grid_259']
# daxing_grid_data301 = history_data_grid.loc[history_data_grid['stationId'] == 'beijing_grid_301']
#
# daxing_grid_data280 = daxing_grid_data280.iloc[712:]
# print (daxing_grid_data280.head(5))
# daxing_grid_data259 = daxing_grid_data259.iloc[712:]
# daxing_grid_data301 = daxing_grid_data301.iloc[712:]
#
#
# daxing_grid_data280.drop(daxing_grid_data280.tail(1877).index,inplace=True)
# daxing_grid_data259.drop(daxing_grid_data259.tail(1877).index,inplace=True)
# daxing_grid_data301.drop(daxing_grid_data301.tail(1877).index,inplace=True)
#
# daxing_grid_data280['weather'] = daxing_data['weather'].values
# daxing_grid_data259['weather'] = daxing_data['weather'].values
# daxing_grid_data301['weather'] = daxing_data['weather'].values
#
# daxing_grid_data280 = daxing_grid_data280[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]
# daxing_grid_data280 = daxing_grid_data280[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]
# daxing_grid_data280 = daxing_grid_data280[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]
#
#
# daxing_grid_data280.to_csv('data/preprocessed_data/daxing_grid_data280.csv', index=0)
# daxing_grid_data259.to_csv('data/preprocessed_data/daxing_grid_data259.csv', index=0)
# daxing_grid_data301.to_csv('data/preprocessed_data/daxing_grid_data301.csv', index=0)
#
# print (len(daxing_grid_data280), len(daxing_grid_data259),len(daxing_grid_data301))


# daxing grid data - testing
test_data_grid = pd.read_csv('data/raw_data/weather_data_kddCup/future_data_from_api/bj_meteorology_grid_2018-05.csv')
test_daxing_data = pd.read_csv('data/preprocessed_data/test/bj_daxing_201805.csv')

# array = ['beijing_grid_280', 'beijing_grid_301', 'beijing_grid_259']
test_data_grid.rename(columns={'time':'utc_time'},  inplace=True)
test_data_grid.rename(columns={'station_id':'stationId'},  inplace=True)

test_data_grid['PM2.5'] = 0
test_data_grid['PM10'] = 0
test_data_grid['NO2'] = 0
test_data_grid['CO'] = 0
test_data_grid['O3'] = 0
test_data_grid['SO2'] = 0

test_data_grid = test_data_grid[['stationId','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','PM2.5','PM10','NO2','CO','O3','SO2']]

daxing_grid_data280_201805 = test_data_grid.loc[test_data_grid['stationId'] == 'beijing_grid_280']
daxing_grid_data259_201805 = test_data_grid.loc[test_data_grid['stationId'] == 'beijing_grid_259']
daxing_grid_data301_201805 = test_data_grid.loc[test_data_grid['stationId'] == 'beijing_grid_301']


daxing_grid_data280_201805.drop(daxing_grid_data280_201805.tail(39).index,inplace=True)
daxing_grid_data259_201805.drop(daxing_grid_data259_201805.tail(39).index,inplace=True)
daxing_grid_data301_201805.drop(daxing_grid_data301_201805.tail(39).index,inplace=True)

daxing_grid_data280_201805['longitude'] = 116.3
daxing_grid_data280_201805['latitude'] = 39.7
daxing_grid_data259_201805['longitude'] = 116.4
daxing_grid_data259_201805['latitude'] = 39.7
daxing_grid_data301_201805['longitude'] = 116.2
daxing_grid_data301_201805['latitude'] = 39.7

daxing_grid_data280_201805 = daxing_grid_data280_201805[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]
daxing_grid_data259_201805 = daxing_grid_data280_201805[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]
daxing_grid_data301_201805 = daxing_grid_data280_201805[['longitude', 'latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather','stationId','PM2.5','PM10','NO2','CO','O3','SO2']]

print (daxing_grid_data280_201805.head(5))

print (len(daxing_grid_data280_201805), len(daxing_grid_data259_201805),len(daxing_grid_data301_201805))

daxing_grid_data280_201805.to_csv('data/preprocessed_data/test/daxing_grid_data280_201805.csv', index=0)
daxing_grid_data259_201805.to_csv('data/preprocessed_data/test/daxing_grid_data259_201805.csv', index=0)
daxing_grid_data301_201805.to_csv('data/preprocessed_data/test/daxing_grid_data301_201805.csv', index=0)

