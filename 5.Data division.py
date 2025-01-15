#5.The data are arranged in increasing order of S, R, N
import pandas as pd

# 读取数据
data = pd.read_excel('Data/NC_SFI.xlsx')

# 移除T，只保留SRN
data['ID'] = data['ID'].str.extract(r'T\d(S\dR\dN\d+)')

# 提取年份信息
data['Year'] = pd.to_datetime(data['Time']).dt.year

# 按年份分组
grouped_by_year = data.groupby('Year')

# 处理每个年份的数据
for year, group in grouped_by_year:
    # 获取ID中地块(S)、重复(R)和小区(N)信息
    group[['S', 'R', 'N']] = group['ID'].str.extract(r'(S\d)(R\d)(N\d+)')

    # 按地块(S)、重复(R)、小区(N)和时间排序
    sorted_group = group.sort_values(by=['S', 'R', 'N', 'Time'])

    # 删除辅助列
    sorted_group.drop(['Year', 'S', 'R', 'N'], axis=1, inplace=True)

    # 导出数据集
    sorted_group.to_excel(f'Data/Feature_{year}.xlsx', index=False)
