# 2.根据光谱一阶导数据，确定特殊点位所在波段位置
import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def process(data):
    def find_band(data, start, end, criteria):
        cols = [col for col in data.columns if str(col).isnumeric() and start <= int(col) <= end]
        relevant_data = data[cols]

        if criteria == 'max':
            return relevant_data.idxmax(axis=1)
        elif criteria == 'min':
            return relevant_data.idxmin(axis=1)
        elif criteria == 'zero':
            return relevant_data.sub(0).abs().idxmin(axis=1)

    data['Band1'] = find_band(data, 410, 450, 'max')
    data['Band2'] = find_band(data, 530, 580, 'zero')
    data['Band3'] = find_band(data, 650, 700, 'zero')
    data['Band4'] = find_band(data, 680, 740, 'max')
    data['Band5'] = find_band(data, 740, 780, 'zero')

    return data

# 加载一阶导数据
data = pd.read_excel('Data/NC_FOD.xlsx')
processed = process(data)

# 提取波段
result_columns = ['ID', 'Time', 'LNC(mg/g)', 'SPAD', 'Band1', 'Band2', 'Band3', 'Band4', 'Band5']
result = processed[result_columns]

# 打印结果
print(result)

# 导出数据
result.to_excel('Data/NC_POS.xlsx', index=False)