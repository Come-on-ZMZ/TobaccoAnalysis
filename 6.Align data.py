import pandas as pd

def process_data(feature_path, label_path, output_path, missing_id):
    # 特征数据
    data = pd.read_excel(feature_path)

    # 获取样本ID和时间
    unique_ids = data['ID'].unique()
    unique_times = sorted(data['Time'].unique())

    # 时间重命名为T1-T3
    time_labels = [f'T{i+1}' for i in range(len(unique_times))]

    # 特征重命名为F1-F7
    feature_columns = data.columns[2:]  # 选特征列
    feature_labels = [f'F{i+1}' for i in range(len(feature_columns))]

    # 创建DataFrame，包含每个ID以及每个时间的特征
    new_data = pd.DataFrame(unique_ids, columns=['ID'])

    # 为每个时间和特征添加新列
    for i, time in enumerate(unique_times):
        for j, col in enumerate(feature_columns):
            new_data[f'{time_labels[i]}_{feature_labels[j]}'] = None

    # 填充新的DataFrame
    for index, row in data.iterrows():
        time_label = time_labels[unique_times.index(row['Time'])]
        for j, col in enumerate(feature_columns):
            new_data.loc[new_data['ID'] == row['ID'], f'{time_label}_{feature_labels[j]}'] = row[col]

    # 缺失数据所在行
    missing_data = new_data[new_data['ID'].str.contains(missing_id)]
    new_data = new_data[~new_data['ID'].str.contains(missing_id)]

    # 将缺失数据放到末尾
    new_data = pd.concat([new_data, missing_data], ignore_index=True)

    # 加载标签
    label_data = pd.read_excel(label_path)

    # 数据合并
    new_data = pd.merge(label_data, new_data, on='ID', how='left')

    # 剔除缺失值
    new_data = new_data.dropna()

    # 保存新数据
    new_data.to_excel(output_path, index=False)
    return new_data

# 2022年数据
process_data('Data/Feature_2022.xlsx',
               'Data/Label_2022.xlsx',
              'Data/Data_22.xlsx',
               'S3')

# 2023年数据
process_data('Data/Feature_2023.xlsx',
             'Data/Label_2023.xlsx',
             'Data/Data_23.xlsx',
             'S4')
