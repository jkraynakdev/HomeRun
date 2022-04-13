import pandas as pd

def merge_csv():
	df = pd.concat(map(pd.read_csv, ['2021_fast.csv', '2021_break.csv','2019_fast.csv','2019_break.csv']), ignore_index=True)
	return df

def write_to_excel(df) -> None:
	df.to_excel("output_created.xlsx",
           	sheet_name='Sheet_name_1')  

def write_to_csv(df) -> None:
	df.to_csv('combined.csv')

if __name__ == '__main__':
	write_to_csv(merge_csv())