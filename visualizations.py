from data import Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter_plot(x: list, y: list) -> None:
	plt.scatter(x, y, marker='o')
	plt.show()

def pitch_type_heatmap(pitch_type: list, y: list, column_label: str, sample) -> None:
	df = pd.DataFrame({'Pitch Type' : pitch_type, column_label: y, 'Count' : data_object.assign_event_cat(sample)})
	# Bin Columns
	df[column_label] = pd.qcut(df[column_label], q=4, duplicates='drop')

	# Turn long format into a wide format
	df_heatmap = df.pivot_table(index="Pitch Type", columns=column_label, values='Count', aggfunc=np.mean)

	# plot it
	sns.heatmap(df_heatmap)
	plt.show()

if __name__ == '__main__':	
	df = pd.read_csv('combined.csv', encoding="utf-8-sig")
	data_object = Data(df)
	data_object.set_random_samples()
	sample = data_object.random_concat

	pitch_type = data_object.get_pitch_type(sample)
	VAA = data_object.get_VAA(sample)

	pitch_type_heatmap(pitch_type, VAA, 'VAA', sample)

	'''
	#scatter_plot(data_object.get_spin(home_run_sample), data_object.get_velo(home_run_sample))
	test_df = pd.DataFrame({'Type' : data_object.get_pitch_type(sample),
							'Velo': data_object.get_velo(sample),
							'Count': data_object.assign_event_cat(sample)})
	
	test_df['Velo'] = pd.qcut(test_df['Velo'], q=5, duplicates='drop')
	# Turn long format into a wide format
	df_heatmap = test_df.pivot_table(index="Type", columns="Velo", values='Count', aggfunc=np.mean)
	# plot it
	sns.heatmap(df_heatmap)
	plt.show()
	'''