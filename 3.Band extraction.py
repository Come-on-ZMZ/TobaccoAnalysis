#3.Extraction of reflectance data based on band position
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

# 加载数据
smoothed_data = pd.read_excel('Data/NC_SAV.xlsx')
position_data = pd.read_excel('Data/NC_POS.xlsx')

def extract_values(smoothed_data, position_data):
    # 确保ID列的数据类型一致
    smoothed_data['ID'] = smoothed_data['ID'].astype(str)
    position_data['ID'] = position_data['ID'].astype(str)

    for i in range(1, 6):
        band_column = f'Band{i}'
        reflectance_column = f'Ref{i}'
        # 提取对应的波段
        position_data[reflectance_column] = position_data.apply(
            lambda row: smoothed_data.loc[smoothed_data['ID'] == row['ID'],
            int(row[band_column])].values[0] if int(row[band_column]) in smoothed_data.columns else None, axis=1
        )

    # 合并光谱数据
    band_data_columns = ['ID', 'Time', 'LNC(mg/g)', 'SPAD'] + list(smoothed_data.columns[4:])
    band_data = smoothed_data[band_data_columns]
    processed_data = pd.merge(position_data, band_data, on=['ID', 'Time', 'LNC(mg/g)', 'SPAD'], how='left')
    print(processed_data)

    return processed_data

# 提取导出数据
corrected_data = extract_values(smoothed_data, position_data)
corrected_data.to_excel('Data/NC_Band.xlsx', index=False)
