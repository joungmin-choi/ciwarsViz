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

doc_list = []
for i in range(len(meta_data)) :
	year = meta_data['Time'][i].split('-')[0]
	month = meta_data['Time'][i].split('-')[1]
	day = meta_data['Time'][i].split('-')[2]
	if len(day) == 1 :
		day = '0' + day
	timepoint = year + "-" + month + "-" + day
	doc_list.append(timepoint)

meta_data['Time'] = doc_list
meta_data.rename(columns = {'Sample Name' : "Sample", "Time" : "DOC"}, inplace = True)
meta_data.to_csv(os.path.join(save_dir,"meta_data.csv"), mode = "w", index = False)


###################################
#### Kraken2 output processing ####
###################################

otu_sample_info = pd.read_csv(output_info_filename)

taxa_list = ['P', 'C', 'O', 'F', 'G', 'S']
taxa_fullname_list = ['phylum', 'class', 'order', 'family', 'genus', 'species']

data_dict = {}
for taxa in taxa_list :
	data_dict[taxa] = pd.DataFrame()

for i in range(len(otu_sample_info)) : #len(otu_sample_info)
	filename = otu_sample_info['KrakenOutput'][i]
	data = pd.read_csv(filename, header = None, sep = '\t') 
	year = otu_sample_info['Timepoint'][i].split('-')[0]
	month = otu_sample_info['Timepoint'][i].split('-')[1]
	day = otu_sample_info['Timepoint'][i].split('-')[2]
	if len(day) == 1 :
		day = '0' + day
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

top20_class = data_dict['C'].T[:20].T
top20_class['Sum'] = top20_class.T.sum()
top20_class.index.name = 'timepoint'
top20_class.to_csv(os.path.join(save_dir, "data_eff.csv"), mode = "w", index = True)
del top20_class['Sum']
top20_class = top20_class.T
top20_class.index.name = 'classes'
top20_class.to_csv(os.path.join(save_dir, "transpose_eff.csv"), mode = "w", index = True)


###################################
#### Diamond output processing ####
###################################

for i in range(len(otu_sample_info)) : #len(otu_sample_info)
	filename = otu_sample_info['DiamondOutput'][i]
	data = pd.read_csv(filename, sep = '\t') 
	year = otu_sample_info['Timepoint'][i].split('-')[0]
	month = otu_sample_info['Timepoint'][i].split('-')[1]
	day = otu_sample_info['Timepoint'][i].split('-')[2]
	if len(day) == 1 :
		day = '0' + day
	timepoint = year + "-" + month + "-" + day
	data = data.drop_duplicates(['gene', 'drug', 'protein_accession', 'gene_family'], keep = 'first')
	data.reset_index(inplace = True, drop = True)
	data = data[['gene', 'drug', 'protein_accession', 'gene_family', '16S_Normalization']]
	data = data.rename(columns = {'16S_Normalization' : timepoint})
	if i == 0 :
		arg_abun_data = data.copy()
	else :
		arg_abun_data = pd.merge(arg_abun_data, data, on = ["gene", 'drug', 'protein_accession', 'gene_family'])

grouped = arg_abun_data.groupby('gene_family')
arg_abun_data_gene_family_sum = grouped.sum()
arg_abun_data_gene_family_sum = pd.DataFrame(arg_abun_data_gene_family_sum)
arg_abun_data_gene_family_sum.drop(['drug', 'protein_accession'], axis = 1, inplace = True)
arg_abun_data_gene_family_sum.set_index('gene', inplace = True, drop = True)
arg_abun_data_gene_family_sum = arg_abun_data_gene_family_sum.T
arg_abun_data_gene_family_sum = arg_abun_data_gene_family_sum.sort_index()
arg_abun_data_gene_family_sum.index.name = 'timepoint'

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

sample_list = data.index.tolist()
sample_iqr_outlier = []
feature_iqr_outlier_df = pd.DataFrame()
for sample in sample_list :
	feature_outlier = []
	for i in range(len(data.columns)) :
		if (data[data.columns[i]][sample] <= q1[i] - 1.5*iqr[i]) or (data[data.columns[i]][sample] >= q3[i] + 1.5*iqr[i]) :
			feature_outlier.append('anomaly')
		else :
			feature_outlier.append('normal')
	feature_iqr_outlier_df[sample] = feature_outlier

feature_iqr_outlier_df = feature_iqr_outlier_df.T
feature_iqr_outlier_df.columns = data.columns
feature_iqr_outlier_df.reset_index(inplace = True, drop = False)
feature_iqr_outlier_df = feature_iqr_outlier_df.rename(columns = {'index' : 'timepoint'})

arg_list = feature_iqr_outlier_df.columns.tolist()
arg_list.remove("timepoint")
pd.DataFrame(arg_list).T.to_csv(os.path.join(save_dir, "arg_list.csv"), mode = "w", index = False, header = False)

arg_abun_data_gene_family_sum.reset_index(inplace = True, drop = False)

final_df = pd.DataFrame()
for arg in arg_list :
	tmp_value_df = arg_abun_data_gene_family_sum[['timepoint', arg]]
	tmp_anomaly_df = feature_iqr_outlier_df[['timepoint', arg]]
	tmp_value_df.columns = ['timepoint', 'abundance']
	tmp_anomaly_df.columns = ['timepoint', 'is_anomaly']
	tmp_merged = pd.merge(tmp_value_df, tmp_anomaly_df)
	tmp_merged['ARG'] = arg
	final_df = pd.concat([final_df, tmp_merged], axis = 0)

final_df.to_csv(os.path.join(save_dir, "viz_EFF_ARG_anomaly_data.csv"), mode = "w", index = False)


grouped = arg_abun_data.groupby('drug')
arg_abun_data_drug_sum = grouped.sum()
arg_abun_data_drug_sum = pd.DataFrame(arg_abun_data_drug_sum)
arg_abun_data_drug_sum.drop(['gene', 'gene_family', 'protein_accession'], axis = 1, inplace = True)
arg_abun_data_drug_sum = arg_abun_data_drug_sum.T
del arg_abun_data_drug_sum['unclassified']
drug_class = arg_abun_data_drug_sum.columns.tolist()
arg_abun_data_drug_sum.reset_index(inplace = True, drop = False)
arg_abun_data_drug_sum.rename(columns = {'index' : 'timepoint'}, inplace = True)


pca = PCA(n_components=2)
x = arg_abun_data_drug_sum.loc[:, drug_class].values
y = arg_abun_data_drug_sum.loc[:,['timepoint']].values
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal_component_1', 'principal_component_2'])


arg_abun_data_drug_sum.rename(columns = {'beta-lactam' : 'betalactam'}, inplace = True)
arg_abun_data_drug_sum = pd.concat([arg_abun_data_drug_sum, principalDf], axis = 1)
arg_abun_data_drug_sum.to_csv(os.path.join(save_dir, 'pca_output_arg_abundance_16S.csv'), index=False, float_format = "%.8f")

