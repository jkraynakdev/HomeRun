import pandas as pd
import math

class Data:
	''' Handles accessing using pitch data
	Attributes: 
		df: Pandas Dataframe type representing all pitches from 2021 season
		random_sample_homeruns: DataFrame of a random sample of 200 homerun pitches
		random_sample_non_homeruns: DataFrame of a random sample of 200 non-homerun pitches
		random_concat: DataFrame of the random_sample_homeruns + random_sample_non_homeruns
	'''	

	def __init__(self, df) -> None:
		self.df = df

		# Variable to store randomly sampled df
		self.random_sample_homeruns = None
		self.random_sample_non_homeruns = None
		self.random_concat = None

	def get_outcome_pitches(self):
		''' Returns a df where some sort of event (walk, single, etc.) happens '''
		return self.df.dropna(subset=['events'])

	def get_non_homerun_pitches(self):
		''' Returns a loc frame of non-home_run pitches '''
		return self.df.loc[self.df['events'] != 'home_run']

	def get_homerun_pitches(self):
		''' Returns loc frame of home_run pitches'''
		return self.df.loc[self.df['events'] == 'home_run']

	def random_sample_df(self, df_input):
		''' Returns a randomly sampled df '''
		#return df_input.sample(frac=0.5, replace=False, random_state=1)
		return df_input.sample(n=700, replace=False, random_state=1)

	def split_by_pitch_type(self, df, pitch_type: str):
		''' Returns a Pandas loc frame controlled by pitch type'''
		return df.loc[df['pitch_type'] == pitch_type]

	def set_random_samples(self) -> None:
		''' Setter function for class attributes '''
		self.random_sample_homeruns = self.random_sample_df(self.get_homerun_pitches())
		self.random_sample_non_homeruns = self.random_sample_df(self.get_non_homerun_pitches())
		self.random_concat = pd.concat([self.random_sample_homeruns, self.random_sample_non_homeruns])

	def assign_event_cat(self, in_df) -> list:
		''' Returns a list of wether a pitch was hit for a homerun (1) or not (0) '''
		events = []
		cat_vars = []

		events.append(in_df['events'].apply(lambda x: x).values.tolist())
		# Flatten
		events = events[0]

		for event in events:
			if event == 'home_run':
				cat_vars.append(1)
			else:
				cat_vars.append(0)

		return cat_vars

	def get_feature(self, in_df, feature: str) -> list:
		''' TODO '''
		# VAA has slightly different logic
		if feature == 'VAA':
			return self.get_VAA(in_df)

		else:
			feature_list = []
			feature_list.append(in_df[feature].apply(lambda x: x).values.tolist())

			# Flatten and return
			return feature_list[0]

	def get_pitch_type(self, in_df) -> list:
		pitch_types = []
		pitch_types.append(in_df['pitch_type'].apply(lambda x: x).values.tolist())

		# Flatten and return
		return pitch_types[0]

	def calc_approach(self, v_y: float, a_y: float, 
						v_z: float, a_z: float) -> float:
		''' Calculates VAA'''
		vy_f = -math.sqrt(v_y**2 - 2*(2*a_y*(50-17/12)))
		t = (vy_f-v_y)/(a_y)
		vz_f = v_z+(a_z*t)
		vaa = -math.atan(vz_f/vy_f)*(180/math.pi)

		return vaa

	def get_VAA(self, in_df) -> list:
		v_y = []
		a_y = []
		y_f = []
		v_z = []
		a_z = []

		approach_angles = []

		v_y.append(in_df['vy0'].apply(lambda x: x).values.tolist())
		a_y.append(in_df['ay'].apply(lambda x: x).values.tolist())
		v_z.append(in_df['vz0'].apply(lambda x: x).values.tolist())
		a_z.append(in_df['az'].apply(lambda x: x).values.tolist())

		# Flatten lists
		v_y, a_y, v_z, a_z = v_y[0], a_y[0], v_z[0], a_z[0]

		for i in range(len(v_y)):
			approach_angles.append(self.calc_approach(v_y[i], a_y[i], v_z[i], a_z[i]))

		return approach_angles


if __name__ == '__main__':
	df = pd.read_csv('combined.csv', encoding="utf-8-sig")

	data = Data(df)
	data.set_random_samples()

	print(data.df['events'].value_counts())
