import numpy as np
import pandas as pd
from rosely import WindRose
i=2007
while i<2018:
	print('HACKATHON-SHELL Wind Rose Diagrame for Year - '+str(i)+'\n')
	df=pd.read_csv(r'../Shell_Hackathon Dataset/Wind Data/wind_data_'+str(i)+'.csv',index_col='date', parse_dates=True)
	WR=WindRose()
	WR.df=df
	#print(df[:5])
	#print('\n\n\n')
	#print(df.describe())
	print('\n')
	names={'sped':'ws','drct':'wd'}
	WR.calc_stats(bins=10,variable_names=names)
	#print(WR.df[:2])
	WR.plot(title='HACKATHON-SHELL Wind Rose Diagrame for Year - '+str(i),output_type='show')
	WR.plot(title='HACKATHON-SHELL Wind Rose Diagrame for Year - '+str(i),output_type='save',out_file='Rose_YEAR-'+str(i)+'.html')
	if i==2009:
		i=2013
		continue
	elif i==2015:
		i=2017
		continue
	i+=1
#to save the windrose write as a  html file ---> WR.plot(output_type='save')
