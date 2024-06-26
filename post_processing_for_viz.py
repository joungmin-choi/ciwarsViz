'''
Require installation of the below python3 packages :
pip3 install openpyxl, pandas, numpy, scikit-learn
'''

import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
import warnings
import os
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

###############
#### INPUT ####
###############
meta_data_filename = "metadata_input" # Metadata user upload
output_info_filename = "sample_list.csv" # CSV file having Timepoint,KrakenOutput,DiamondOutput

save_dir = "./input_for_viz/"
os.makedirs(save_dir, exist_ok = True)


###################################
######  Metadata processing  ######
###################################
try :
	meta_data = pd.read_excel(meta_data_filename + ".xlsx")
except :
	meta_data = pd.read_csv(meta_data_filename + ".csv")

try :
	print(len(meta_data))
except :
	raise Exception("Cannot read meta data file!")

metadata_colums = meta_data.columns.tolist()
selected_metadata_columns = []
for col in metadata_colums :
	if 'time' in col.lower() :
		timepoint_col = col
		selected_metadata_columns.append(col)
	elif 'sample' in col.lower() :
		samplename_col = col
		selected_metadata_columns.append(col)
	else :
		try :
			meta_data[col].astype('float')
			selected_metadata_columns.append(col)
		except :
			continue

meta_data = meta_data[selected_metadata_columns]


try :
	print(timepoint_col)
except :
	raise Exception("Cannot find column indicating timepoint in meta data file!")


try :
	print(samplename_col)
except :
	raise Exception("Cannot find column indicating sample name in meta data file!")


doc_list = []
for i in range(len(meta_data)) :
	year = meta_data[timepoint_col][i].split('-')[0]
	month = meta_data[timepoint_col][i].split('-')[1]
	day = meta_data[timepoint_col][i].split('-')[2]
	if len(year) == 4 :
		year = year[2:]
	if len(day) == 1 :
		day = '0' + day
	if len(month) == 1 :
		month = '0' + month
	timepoint = year + "-" + month + "-" + day
	doc_list.append(timepoint)

meta_data[timepoint_col] = doc_list
meta_data.rename(columns = {samplename_col : "Sample", timepoint_col : "DOC"}, inplace = True)
meta_data.to_csv(os.path.join(save_dir,"meta_data.csv"), mode = "w", index = False)

################################
### Find anomaly in metadata ###
################################
del meta_data['Sample']

meta_data.rename(columns = {'DOC' : 'timepoint'}, inplace = True)
meta_data.set_index('timepoint', inplace = True, drop = True)

q1 = meta_data.quantile(0.25)
q3 = meta_data.quantile(0.75)
iqr = q3-q1

meta_feature_list = meta_data.columns.tolist()

sample_list = meta_data.index.tolist()
sample_iqr_outlier = []
feature_iqr_outlier_df = pd.DataFrame()
for sample in sample_list :
	feature_outlier = []
	for i in range(len(meta_feature_list)) :
		if (meta_data[meta_feature_list[i]][sample] <= q1[i] - 1.5*iqr[i]) or (meta_data[meta_feature_list[i]][sample] >= q3[i] + 1.5*iqr[i]) :
			feature_outlier.append('anomaly')
		else :
			feature_outlier.append('normal')
	feature_iqr_outlier_df[sample] = feature_outlier

feature_iqr_outlier_df = feature_iqr_outlier_df.T
feature_iqr_outlier_df.columns = meta_data.columns
feature_iqr_outlier_df.reset_index(inplace = True, drop = False)
feature_iqr_outlier_df = feature_iqr_outlier_df.rename(columns = {'index' : 'timepoint'})

meta_data.reset_index(inplace = True, drop = False)
final_df = pd.DataFrame()
for feature in meta_feature_list :
	tmp_value_df = meta_data[['timepoint', feature]]
	tmp_anomaly_df = feature_iqr_outlier_df[['timepoint', feature]]
	tmp_value_df.columns = ['timepoint', 'value']
	tmp_anomaly_df.columns = ['timepoint', 'is_anomaly']
	tmp_merged = pd.merge(tmp_value_df, tmp_anomaly_df)
	tmp_merged['feature'] = feature
	final_df = pd.concat([final_df, tmp_merged], axis = 0)

final_df.to_csv(os.path.join(save_dir, "viz_metadata_anomaly_data.csv"), mode = "w", index = False)


###################################
#### Kraken2 output processing ####
###################################

try : 
	otu_sample_info = pd.read_csv(output_info_filename)
except :
	raise Exception("Cannot read **sample_list.csv** having kraken and diamond output path!")


for col in otu_sample_info.columns.tolist() :
	if 'time' in col.lower() :
		otu_time_col = col 
	if 'kraken' in col.lower() :
		kraken_col = col 
	if 'diamond' in col.lower() :
		diamond_col = col

otu_sample_info.rename(columns = {otu_time_col : "Timepoint", kraken_col : "KrakenOutput", diamond_col : "DiamondOutput"}, inplace = True)


taxa_list = ['P', 'C', 'O', 'F', 'G', 'S']
taxa_fullname_list = ['phylum', 'class', 'order', 'family', 'genus', 'species']

data_dict = {}
for taxa in taxa_list :
	data_dict[taxa] = pd.DataFrame()

for i in range(len(otu_sample_info)) : #len(otu_sample_info)
	filename = otu_sample_info['KrakenOutput'][i]
	try :
		data = pd.read_csv(filename, header = None, sep = '\t') 
	except Exception:
		print(Exception)
		continue

	year = otu_sample_info['Timepoint'][i].split('-')[0]
	month = otu_sample_info['Timepoint'][i].split('-')[1]
	day = otu_sample_info['Timepoint'][i].split('-')[2]
	if len(year) == 4 :
		year = year[2:]
	if len(day) == 1 :
		day = '0' + day
	if len(month) == 1 :
		month = '0' + month
	timepoint = year + "-" + month + "-" + day
	for taxa in taxa_list :
		data_selected_taxa = data[data[3] == taxa]
		data_selected_taxa = data_selected_taxa.drop([1,2,3,4], axis = 1)
		total_sum = data_selected_taxa[0].sum()
		data_selected_taxa.reset_index(inplace = True, drop = True)
		data_selected_taxa[0] = data_selected_taxa[0]/total_sum
		data_selected_taxa = data_selected_taxa.rename(columns = {0 : timepoint, 5 : 'OTU'})
		if i == 0 :
			data_dict[taxa] = data_selected_taxa.copy()
		else :
			data_dict[taxa] = pd.merge(data_dict[taxa], data_selected_taxa, left_on = 'OTU', right_on = 'OTU')

for i in range(len(taxa_list)) : #taxa_list
	taxa = taxa_list[i]
	data_dict[taxa].set_index('OTU', inplace = True, drop = True)
	if taxa == 'S' :
		corr_timepoint = data_dict[taxa].corr()
		colnames = corr_timepoint.columns.tolist()
		tmp_row = []
		tmp_col = []
		tmp_val = []
		for col in colnames :
			for row in colnames :
				tmp_row.append(row)
				tmp_col.append(col)
				tmp_val.append(corr_timepoint[col][row])
		corr_timepoint = pd.DataFrame({'row' : tmp_row, 'col' : tmp_col, 'val' : tmp_val})
		corr_timepoint.to_csv(os.path.join(save_dir, "corr_timepoints.csv"), mode = "w", index = False)
	data_dict[taxa] = data_dict[taxa].T
	top30_otu_list = pd.DataFrame(data_dict[taxa].sum()).sort_values(0, ascending = False)[:30].index.tolist()

	## code added for stacked barplot
	data_dict_stacked = data_dict.copy()
	top19_otu_list = pd.DataFrame(data_dict_stacked[taxa].sum()).sort_values(0, ascending = False)[:19].index.tolist()	
	data_dict_stacked[taxa]['others'] = data_dict_stacked[taxa].drop(columns=top19_otu_list).sum(axis=1)
	top20_otu_list = top19_otu_list + ['others']
	data_dict_stacked[taxa] = data_dict_stacked[taxa][top20_otu_list]
	top20_otu_list = [otu.strip() for otu in top20_otu_list]
	data_dict_stacked[taxa].columns = top20_otu_list

	top20_class = data_dict_stacked[taxa].T[:20].T
	top20_class['Sum'] = top20_class.T.sum()
	top20_class.index.name = 'timepoint'
	top20_class.to_csv(os.path.join(save_dir, "data_eff_" + taxa + "_stacked.csv"), mode = "w", index = True)
	if taxa == 'S' :
		top20_class.to_csv(os.path.join(save_dir, "data_eff.csv"), mode = "w", index = True)
		del top20_class['Sum']
		top20_class.T.to_csv(os.path.join(save_dir, "transpose_eff.csv"), mode = "w", index = True)

	data_dict[taxa] = data_dict[taxa][top30_otu_list]
	top30_otu_list = [otu.strip() for otu in top30_otu_list]
	data_dict[taxa].columns = top30_otu_list
	corr_otu = data_dict[taxa].corr()
	tmp_row = []
	tmp_col = []
	tmp_val = []
	for col in top30_otu_list :
		for row in top30_otu_list :
			tmp_row.append(row)
			tmp_col.append(col)
			tmp_val.append(corr_otu[col][row])
	corr_otu = pd.DataFrame({'row' : tmp_row, 'col' : tmp_col, 'val' : tmp_val})
	corr_otu.to_csv(os.path.join(save_dir, 'corr_' + taxa_fullname_list[i] + ".csv"), mode = "w", index = False)

###################################
#### Diamond output processing ####
###################################

for i in range(len(otu_sample_info)) : #len(otu_sample_info)
	filename = otu_sample_info['DiamondOutput'][i]
	data = pd.read_csv(filename, sep = '\t') 
	year = otu_sample_info['Timepoint'][i].split('-')[0]
	month = otu_sample_info['Timepoint'][i].split('-')[1]
	day = otu_sample_info['Timepoint'][i].split('-')[2]
	if len(year) == 4 :
		year = year[2:]
	if len(day) == 1 :
		day = '0' + day
	if len(month) == 1 :
		month = '0' + month
	timepoint = year + "-" + month + "-" + day
	data = data.drop_duplicates(['gene', 'drug', 'protein_accession', 'gene_family'], keep = 'first')
	data.reset_index(inplace = True, drop = True)
	#data_fpkm = data[['gene', 'drug', 'protein_accession', 'gene_family', 'FPKM_Normalization']]
	#data_tpm = data[['gene', 'drug', 'protein_accession', 'gene_family', 'TPM_Normalization']]
	data = data[['gene', 'drug', 'protein_accession', 'gene_family', 'rpob_Normalization']]
	data = data.rename(columns = {'rpob_Normalization' : timepoint})
	#data_fpkm = data_fpkm.rename(columns = {'FPKM_Normalization' : timepoint})
	#data_tpm = data_tpm.rename(columns = {'TPM_Normalization' : timepoint})
	if i == 0 :
		arg_abun_data = data.copy()
		#arg_abun_data_fpkm = data_fpkm.copy()
		#arg_abun_data_tpm = data_tpm.copy()
	else :
		arg_abun_data = pd.merge(arg_abun_data, data, on = ["gene", 'drug', 'protein_accession', 'gene_family'])
		#arg_abun_data_fpkm = pd.merge(arg_abun_data_fpkm, data_fpkm, on = ["gene", 'drug', 'protein_accession', 'gene_family'])
		#arg_abun_data_tpm = pd.merge(arg_abun_data_tpm, data_tpm, on = ["gene", 'drug', 'protein_accession', 'gene_family'])

melted_arg_abun_data = pd.melt(arg_abun_data, id_vars=["gene", "drug", "protein_accession", "gene_family"], var_name="timepoint", value_name="abundance")
melted_arg_abun_data = melted_arg_abun_data.sort_values(by=["gene", "drug", "protein_accession", "gene_family", "timepoint"]).reset_index(drop=True)
melted_arg_abun_data = melted_arg_abun_data.drop(['protein_accession', 'gene_family'], axis=1)
melted_arg_abun_data = melted_arg_abun_data.drop_duplicates()
melted_arg_abun_data = melted_arg_abun_data.groupby(["gene", "drug", "timepoint"])['abundance'].sum().reset_index()
melted_arg_abun_data.to_csv(os.path.join(save_dir, "data_piechart.csv"), index = False)



tmp_arg_abundance = melted_arg_abun_data.copy()
del tmp_arg_abundance['drug']
arg_abun_data_gene_family_sum = pd.pivot_table(tmp_arg_abundance, values='abundance', index=['timepoint'],columns=['gene'])

'''
melted_arg_abun_data_fpkm = pd.melt(arg_abun_data_fpkm, id_vars=["gene", "drug", "protein_accession", "gene_family"], var_name="timepoint", value_name="abundance")
melted_arg_abun_data_fpkm = melted_arg_abun_data_fpkm.sort_values(by=["gene", "drug", "protein_accession", "gene_family", "timepoint"]).reset_index(drop=True)
melted_arg_abun_data_fpkm = melted_arg_abun_data_fpkm.drop(['protein_accession', 'gene_family'], axis=1)
melted_arg_abun_data_fpkm = melted_arg_abun_data_fpkm.drop_duplicates()
melted_arg_abun_data_fpkm = melted_arg_abun_data_fpkm.groupby(["gene", "drug", "timepoint"])['abundance'].sum().reset_index()
melted_arg_abun_data_fpkm.to_csv(os.path.join(save_dir, "data_piechart_fpkm.csv"), index = False)


melted_arg_abun_data_tpm = pd.melt(arg_abun_data_tpm, id_vars=["gene", "drug", "protein_accession", "gene_family"], var_name="timepoint", value_name="abundance")
melted_arg_abun_data_tpm = melted_arg_abun_data_tpm.sort_values(by=["gene", "drug", "protein_accession", "gene_family", "timepoint"]).reset_index(drop=True)
melted_arg_abun_data_tpm = melted_arg_abun_data_tpm.drop(['protein_accession', 'gene_family'], axis=1)
melted_arg_abun_data_tpm = melted_arg_abun_data_tpm.drop_duplicates()
melted_arg_abun_data_tpm = melted_arg_abun_data_tpm.groupby(["gene", "drug", "timepoint"])['abundance'].sum().reset_index()
melted_arg_abun_data_tpm.to_csv(os.path.join(save_dir, "data_piechart_tpm.csv"), index = False)
'''

feature_list = arg_abun_data_gene_family_sum.columns.tolist()

rm_trend_df = pd.DataFrame()
rm_trend_df['timepoint'] = arg_abun_data_gene_family_sum.index.tolist()

for feature in feature_list :
	tmp_feature_value_list = []
	for i in range(len(arg_abun_data_gene_family_sum)) :
		if i == 0 :
			tmp_feature_value_list.append(arg_abun_data_gene_family_sum[feature][i])
		else :
			tmp_feature_value_list.append(arg_abun_data_gene_family_sum[feature][i] - arg_abun_data_gene_family_sum[feature][i-1])
	rm_trend_df[feature] = tmp_feature_value_list

rm_trend_df.set_index('timepoint', inplace = True, drop  = True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(rm_trend_df)
scaled = pd.DataFrame(scaled)
scaled.columns = rm_trend_df.columns
scaled.index = rm_trend_df.index
scaled = scaled.sort_index()

data = scaled
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3-q1

data_arg_list = data.columns.tolist()
anomalous_arg_list = []
normal_arg_list = []

sample_list = data.index.tolist()
sample_iqr_outlier = []
feature_iqr_outlier_df = pd.DataFrame()
for sample in sample_list :
	feature_outlier = []
	for i in range(len(data_arg_list)) :
		if (data[data_arg_list[i]][sample] <= q1[i] - 1.5*iqr[i]) or (data[data_arg_list[i]][sample] >= q3[i] + 1.5*iqr[i]) :
			feature_outlier.append('anomaly')
			if data_arg_list[i] not in anomalous_arg_list :
				anomalous_arg_list.append(data_arg_list[i])
		else :
			feature_outlier.append('normal')
	feature_iqr_outlier_df[sample] = feature_outlier

normal_arg_list = [arg for arg in data_arg_list if arg not in anomalous_arg_list]

feature_iqr_outlier_df = feature_iqr_outlier_df.T
feature_iqr_outlier_df.columns = data.columns
feature_iqr_outlier_df.reset_index(inplace = True, drop = False)
feature_iqr_outlier_df = feature_iqr_outlier_df.rename(columns = {'index' : 'timepoint'})

pd.DataFrame(normal_arg_list).T.to_csv(os.path.join(save_dir, "arg_normal_list.csv"), mode = "w", index = False, header = False)
pd.DataFrame(anomalous_arg_list).T.to_csv(os.path.join(save_dir, "arg_anomaly_list.csv"), mode = "w", index = False, header = False)
arg_abun_data_gene_family_sum.reset_index(inplace = True, drop = False)

final_df = pd.DataFrame()
for arg in normal_arg_list :
	tmp_value_df = arg_abun_data_gene_family_sum[['timepoint', arg]]
	tmp_anomaly_df = feature_iqr_outlier_df[['timepoint', arg]]
	tmp_value_df.columns = ['timepoint', 'abundance']
	tmp_anomaly_df.columns = ['timepoint', 'is_anomaly']
	tmp_merged = pd.merge(tmp_value_df, tmp_anomaly_df)
	tmp_merged['ARG'] = arg
	final_df = pd.concat([final_df, tmp_merged], axis = 0)

final_df.to_csv(os.path.join(save_dir, "viz_ARG_normal_data.csv"), mode = "w", index = False)

final_df = pd.DataFrame()
for arg in anomalous_arg_list :
	tmp_value_df = arg_abun_data_gene_family_sum[['timepoint', arg]]
	tmp_anomaly_df = feature_iqr_outlier_df[['timepoint', arg]]
	tmp_value_df.columns = ['timepoint', 'abundance']
	tmp_anomaly_df.columns = ['timepoint', 'is_anomaly']
	tmp_merged = pd.merge(tmp_value_df, tmp_anomaly_df)
	tmp_merged['ARG'] = arg
	final_df = pd.concat([final_df, tmp_merged], axis = 0)

final_df.to_csv(os.path.join(save_dir, "viz_ARG_anomaly_data.csv"), mode = "w", index = False)

#######################################
### Count the # of ARGs for anomaly ###
#######################################
summary_info_df = final_df.groupby(['timepoint', 'is_anomaly']).count().reset_index()
final_summary_df = pd.DataFrame()
for sample in sample_list :
	tmp_df = summary_info_df[summary_info_df['timepoint'] == sample]
	tmp_dict = {}
	idx_list = tmp_df.index
	tmp_dict['timepoint'] = [tmp_df['timepoint'][idx_list[0]]]
	for idx in idx_list :
		tmp_dict[tmp_df['is_anomaly'][idx]] = [tmp_df['ARG'][idx]]
	tmp_dict = pd.DataFrame(tmp_dict)
	final_summary_df = pd.concat([final_summary_df, tmp_dict], axis = 0)

final_summary_df.fillna(0, inplace = True)

for col in ['anomaly', 'normal'] :
	try :
		final_summary_df[col] = final_summary_df[col].astype('float')
	except :
		print("No " + col)

#final_summary_df['anomaly'] = final_summary_df['anomaly'].astype('float')
#final_summary_df['normal'] = final_summary_df['normal'].astype('float')

final_summary_df.to_csv(os.path.join(save_dir, "viz_ARG_anomaly_summary.csv"), mode = "w", index = False)

grouped = arg_abun_data.groupby('drug')
arg_abun_data_drug_sum = grouped.sum()
arg_abun_data_drug_sum = pd.DataFrame(arg_abun_data_drug_sum)
arg_abun_data_drug_sum.drop(['gene', 'gene_family', 'protein_accession'], axis = 1, inplace = True)
arg_abun_data_drug_sum = arg_abun_data_drug_sum.T
del arg_abun_data_drug_sum['unclassified']
drug_class = arg_abun_data_drug_sum.columns.tolist()
arg_abun_data_drug_sum.reset_index(inplace = True, drop = False)
arg_abun_data_drug_sum.rename(columns = {'index' : 'timepoint'}, inplace = True)

'''
grouped_fpkm = arg_abun_data_fpkm.groupby('drug')
arg_abun_data_drug_sum_fpkm = grouped_fpkm.sum()
arg_abun_data_drug_sum_fpkm = pd.DataFrame(arg_abun_data_drug_sum_fpkm)
arg_abun_data_drug_sum_fpkm.drop(['gene', 'gene_family', 'protein_accession'], axis = 1, inplace = True)
arg_abun_data_drug_sum_fpkm = arg_abun_data_drug_sum_fpkm.T
del arg_abun_data_drug_sum_fpkm['unclassified']
arg_abun_data_drug_sum_fpkm.reset_index(inplace = True, drop = False)
arg_abun_data_drug_sum_fpkm.rename(columns = {'index' : 'timepoint'}, inplace = True)


grouped_tpm = arg_abun_data_tpm.groupby('drug')
arg_abun_data_drug_sum_tpm = grouped_tpm.sum()
arg_abun_data_drug_sum_tpm = pd.DataFrame(arg_abun_data_drug_sum_tpm)
arg_abun_data_drug_sum_tpm.drop(['gene', 'gene_family', 'protein_accession'], axis = 1, inplace = True)
arg_abun_data_drug_sum_tpm = arg_abun_data_drug_sum_tpm.T
del arg_abun_data_drug_sum_tpm['unclassified']
arg_abun_data_drug_sum_tpm.reset_index(inplace = True, drop = False)
arg_abun_data_drug_sum_tpm.rename(columns = {'index' : 'timepoint'}, inplace = True)
'''

'''
pca = PCA(n_components=2)
x = arg_abun_data_drug_sum.loc[:, drug_class].values
y = arg_abun_data_drug_sum.loc[:,['timepoint']].values
x = StandardScaler().fit_transform(x)
if len(arg_abun_data_drug_sum) > 1 :
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal_component_1', 'principal_component_2'])
'''
arg_abun_data_drug_sum_copy = arg_abun_data_drug_sum.copy()
arg_abun_data_drug_sum_copy.rename(columns = {'beta-lactam' : 'betalactam'}, inplace = True)
	# arg_abun_data_drug_sum_copy = pd.concat([arg_abun_data_drug_sum_copy, principalDf], axis = 1)
arg_abun_data_drug_sum_copy.to_csv(os.path.join(save_dir, 'pca_output_arg_abundance_rpob.csv'), index=False, float_format = "%.8f")


'''
arg_abun_data_drug_sum_fpkm.rename(columns = {'beta-lactam' : 'betalactam'}, inplace = True)
arg_abun_data_drug_sum_fpkm.to_csv(os.path.join(save_dir, 'pca_output_arg_abundance_fpkm.csv'), index=False, float_format = "%.8f")

arg_abun_data_drug_sum_tpm.rename(columns = {'beta-lactam' : 'betalactam'}, inplace = True)
arg_abun_data_drug_sum_tpm.to_csv(os.path.join(save_dir, 'pca_output_arg_abundance_tpm.csv'), index=False, float_format = "%.8f")
'''

# Standardize your data (excluding the "timepoint" column)
x = arg_abun_data_drug_sum.drop(columns=["timepoint"]).values
x = StandardScaler().fit_transform(x)

# Perform nMDS (Multidimensional Scaling)
mds = MDS(n_components=2, random_state=42)
principalComponents = mds.fit_transform(x)

# Create a DataFrame for the principal components
principalDf = pd.DataFrame(data=principalComponents, columns=['nMDS_component_1', 'nMDS_component_2'])

# Create a range of K values to test
if len(arg_abun_data_drug_sum) > 11 :
	k_values = range(2, 11)  # You can adjust the range
elif len(arg_abun_data_drug_sum) == 1 :
	k_values = []
else :
	k_values = range(2, len(arg_abun_data_drug_sum))

# Initialize lists to store silhouette scores
silhouette_scores = []

# Iterate through different K values and calculate silhouette score
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(principalDf)  # Use the principal components for clustering
    silhouette_avg = silhouette_score(principalDf, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the K with the highest silhouette score
try :
	optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
	# Perform K-means clustering with the optimal K (using principal components)
	optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
	optimal_cluster_labels = optimal_kmeans.fit_predict(principalDf)

	# Add cluster labels to the principal components DataFrame
	principalDf['cluster_number'] = optimal_cluster_labels
except :
	principalDf['cluster_number'] = [0]

# Add the "timepoint" column to the DataFrame
principalDf['timepoint'] = arg_abun_data_drug_sum['timepoint'].values

# Save the DataFrame to a CSV file
arg_cluster_file_name = "nMDS_cluster_results.csv"

results_file = os.path.join(save_dir, arg_cluster_file_name)
principalDf.to_csv(results_file, index=False)

file_names = ['data_eff_C_stacked.csv','data_eff_F_stacked.csv','data_eff_G_stacked.csv','data_eff_O_stacked.csv', 'data_eff_P_stacked.csv','data_eff_S_stacked.csv']

for file_name in file_names:
    # Load the data from the CSV file
    timepoints_species_data = pd.read_csv(os.path.join(save_dir, file_name))

    # Standardize your data (excluding the "timepoint" column)
    columns_to_drop = ["timepoint", "others", "Sum"]
    x_species = timepoints_species_data.drop(columns=columns_to_drop).values
    x_species = StandardScaler().fit_transform(x_species)

    # Perform nMDS (Multidimensional Scaling)
    mds_species = MDS(n_components=2, random_state=42)
    principalComponents_species = mds_species.fit_transform(x_species)

    # Create a DataFrame for the principal components
    principalDf_species = pd.DataFrame(data=principalComponents_species, columns=['nMDS_component_1', 'nMDS_component_2'])

    # Create a range of K values to test
    if len(principalDf_species) > 11 :
    	k_values_species = range(2, 11)  # You can adjust the range
    elif len(principalDf_species) == 1 :
    	k_values_species = []
    else :
    	k_values_species = range(2, len(principalDf_species))

    # Initialize lists to store silhouette scores
    silhouette_scores_species = []

    # Iterate through different K values and calculate silhouette score
    for k in k_values_species:
        kmeans_species = KMeans(n_clusters=k, random_state=42)
        cluster_labels_species = kmeans_species.fit_predict(principalDf_species)  # Use the principal components for clustering
        silhouette_avg_species = silhouette_score(principalDf_species, cluster_labels_species)
        silhouette_scores_species.append(silhouette_avg_species)

    try :
    	# Find the K with the highest silhouette score
    	optimal_k_species = k_values_species[silhouette_scores_species.index(max(silhouette_scores_species))]

    	# Perform K-means clustering with the optimal K (using principal components)
    	optimal_kmeans_species = KMeans(n_clusters=optimal_k_species, random_state=42)
    	optimal_cluster_labels_species = optimal_kmeans_species.fit_predict(principalDf_species)

    	# Add cluster labels to the principal components DataFrame
    	principalDf_species['cluster_number'] = optimal_cluster_labels_species
    except :
    	principalDf_species['cluster_number'] = [0]

    # Add the "timepoint" column to the DataFrame
    principalDf_species['timepoint'] = timepoints_species_data['timepoint'].values

    # Save the results to a CSV file in the results directory
    results_file = os.path.join(save_dir, file_name.replace('.csv', '_clustering_results.csv'))
    principalDf_species.to_csv(results_file, index=False)