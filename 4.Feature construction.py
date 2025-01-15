#4.Constructing the SFI features
import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rcParams

# 设置全局字体为Times New Roman, 12号
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
title_font = {'size': 12, 'color': 'k', 'family': 'Times New Roman'}  # 标题字体

# 定义拟合函数
def lorentz(x, x0, a, gamma):
    return a / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))

def gaussian(x, Rs, R0, x0, sigma):
    return Rs - (Rs - R0) * np.exp(-((x0 - x) ** 2) / (2 * sigma ** 2))

def linear(x, m, c):
    return m * x + c

# 加载数据
data = pd.read_excel('Data/NC_Band.xlsx')

# 确定波段范围
wavelengths = data.columns[14:].astype(float)  # 第15列及之后为波段数据
print(wavelengths)

# DataFrame的列结构,添加波段列： + wavelengths.tolist()
results_df = pd.DataFrame(columns=["ID", "Time", "LNC(mg/g)", "SPAD", "Iz", "Ig", "Il1", "Il2", "Il3"])

# 选择样本
plot_samples_indices = [225, 231, 325, 331, 512, 356, 421]

# 创建绘图
plt.figure(figsize=(8, 6), dpi=100)

# 处理每个样本
for idx in data.index:
    sample = data.iloc[idx]

    # Linear1拟合与积分
    linear_bounds1 = (sample['Band1'] + 20, sample['Band2'] - 60)
    linear_mask1 = (wavelengths >= linear_bounds1[0]) & (wavelengths <= linear_bounds1[1])
    popt_linear1, _ = curve_fit(linear, wavelengths[linear_mask1], sample.iloc[14:].values[linear_mask1],
                                p0=[0, sample['Ref1']])
    Il1 = trapezoid(linear(wavelengths[linear_mask1], *popt_linear1), wavelengths[linear_mask1])

    # Lorentz拟合与积分
    lorentz_bounds = (sample['Band2']-50, sample['Band2']+40)
    lorentz_mask = (wavelengths >= lorentz_bounds[0]) & (wavelengths <= lorentz_bounds[1])
    popt_lorentz, _ = curve_fit(lorentz, wavelengths[lorentz_mask], sample.iloc[14:].values[lorentz_mask], p0=[sample['Band2'], sample['Ref2'], 30])
    Iz = trapezoid(lorentz(wavelengths[lorentz_mask], *popt_lorentz), wavelengths[lorentz_mask])

    # Linear2拟合与积分
    linear_bounds2 = (sample['Band2'] + 40, sample['Band3'] - 20)
    linear_mask2 = (wavelengths >= linear_bounds2[0]) & (wavelengths <= linear_bounds2[1])
    popt_linear2, _ = curve_fit(linear, wavelengths[linear_mask2], sample.iloc[14:].values[linear_mask2],
                                p0=[0, (sample['Ref3'] + sample['Ref2']) / 2])
    Il2 = trapezoid(linear(wavelengths[linear_mask2], *popt_linear2), wavelengths[linear_mask2])

    # Gaussian拟合与积分
    gaussian_bounds = (sample['Band3'] + 20, sample['Band5'] - 10)
    gaussian_mask = (wavelengths >= gaussian_bounds[0]) & (wavelengths <= gaussian_bounds[1])
    popt_gaussian, _ = curve_fit(gaussian, wavelengths[gaussian_mask], sample.iloc[14:].values[gaussian_mask], p0=[sample['Ref5'], sample['Ref3'], sample['Band3'], (sample['Band4'] - sample['Band3'])/2])
    Ig = trapezoid(gaussian(wavelengths[gaussian_mask], *popt_gaussian), wavelengths[gaussian_mask])

    # Linear3拟合与积分
    linear_bounds3 = (sample['Band5'], 900)
    linear_mask3 = (wavelengths >= linear_bounds3[0]) & (wavelengths <= linear_bounds3[1])
    popt_linear3, _ = curve_fit(linear, wavelengths[linear_mask3], sample.iloc[14:].values[linear_mask3], p0=[0, sample['Ref5']])
    Il3 = trapezoid(linear(wavelengths[linear_mask3], *popt_linear3), wavelengths[linear_mask3])

    # 将结果添加到results_df中 添加光谱数据：+ spectral_data.tolist()
    spectral_data = sample.iloc[14:].values
    new_row = pd.DataFrame([[sample["ID"], sample["Time"], sample["LNC(mg/g)"], sample["SPAD"], Iz, Ig, Il1, Il2, Il3]   ],
                           columns=["ID", "Time", "LNC(mg/g)", "SPAD", "Iz", "Ig", "Il1", "Il2", "Il3"] )  # “]” 后 “+ wavelengths.tolist()”
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # 绘制选定样本的拟合曲线
    if idx in plot_samples_indices:
        plt.plot(wavelengths, sample.iloc[14:].values, linewidth=1.0, label=f'{sample["ID"]}')
        plt.scatter(wavelengths[lorentz_mask], lorentz(wavelengths[lorentz_mask], *popt_lorentz), marker='+', s=20)
        plt.scatter(wavelengths[gaussian_mask], gaussian(wavelengths[gaussian_mask], *popt_gaussian), marker='*', s=20)
        plt.plot(wavelengths[linear_mask1], linear(wavelengths[linear_mask1], *popt_linear1), linestyle='--', linewidth=1.2)
        plt.plot(wavelengths[linear_mask2], linear(wavelengths[linear_mask2], *popt_linear2), linestyle='--', linewidth=1.2)
        plt.plot(wavelengths[linear_mask3], linear(wavelengths[linear_mask3], *popt_linear3), linestyle='--', linewidth=1.2)

# 设置图形标题和轴标签
plt.title('Example of Spectral Curve Fitting (Selected Samples)', title_font, pad=8)
plt.xlim(400, 1000)
plt.ylim(0, 0.8)
plt.xticks(np.arange(400, 1001, 50))
plt.yticks(np.arange(0, 0.81, 0.1))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend(loc='upper left', borderaxespad=0.6, prop={'size': 11})
plt.grid()
plt.tight_layout()
plt.show()

# 导出积分结果
results_df.to_excel('Data/NC_SFI.xlsx', index=False)
