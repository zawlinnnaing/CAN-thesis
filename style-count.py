import pandas

csv_file_path = 'data/wikiart/all_data_info_utf.csv'
df = pandas.read_csv(csv_file_path, encoding='utf-8')
labels = df['style']
# label_dict = dict(enumerate(label_dict))
# label_dict = dict((v, k) for k, v in label_dict.items())
# print(label_dict)
labels_counts = labels.value_counts(dropna=True)
print(labels_counts[:29])
