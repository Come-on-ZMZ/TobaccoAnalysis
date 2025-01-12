# 1.数据预处理：平滑和求一阶导
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

# 设置全局字体为Times New Roman, 12号
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
title_font = {'size': 12, 'color': 'k', 'family': 'Times New Roman'} # 标题字体

def data_process(file_path, output_sav, output_der, selected_samples):
    data = pd.read_excel(file_path)
    wavelength_columns = data.columns[4:]

    # 将数据转换为numpy数组
    spectrum_data = data[wavelength_columns].to_numpy()

    # 光谱Savitzky-Golay平滑处理，设置窗口(奇数)，多项式阶数(小于窗口)
    smoothed_data = savgol_filter(spectrum_data, window_length=15, polyorder=2, axis=1)
    smoothed_df_1 = pd.DataFrame(smoothed_data, columns=wavelength_columns)
    smoothed_df = pd.concat([data.iloc[:, :4], smoothed_df_1], axis=1)

    # 计算光谱平滑后的一阶导数
    derivatives = np.gradient(smoothed_data, axis=1)
    derivatives_df_1 = pd.DataFrame(derivatives, columns=wavelength_columns)
    derivatives_df = pd.concat([data.iloc[:, :4], derivatives_df_1], axis=1)

    # 导出数据
    smoothed_df.to_excel(output_sav, index=False)
    derivatives_df.to_excel(output_der, index=False)

    # 绘制原始光谱曲线、平滑后的光谱曲线以及一阶导数曲线
    fig, ax = plt.figure(figsize=(8, 6), dpi=100), plt.gca()
    color1 = np.array([82, 173, 77]) / 255  # 调配颜色
    color2 = np.array([115, 186, 224]) / 255
    color3 = np.array([240, 180, 141]) / 255
    color4 = np.array([244, 241, 222]) / 255
    color5 = np.array([239, 65, 67]) / 255

    # 区域划分
    ax.text(460, 0.32, 'Linear area 1', ha='center', va='center', fontsize=11)
    ax.text(550, 0.32, 'Lorentz area', ha='center', va='center', fontsize=11)
    ax.text(640, 0.32, 'Linear area 2', ha='center', va='center', fontsize=11)
    ax.text(726, 0.52, 'Gaussian area', ha='center', va='center', fontsize=11)
    ax.text(838, 0.52, 'Linear area 3', ha='center', va='center', fontsize=11)
    ax.text(950, 0.52, 'Discarded area', ha='center', va='center', fontsize=11)

    # 区域颜色
    plt.axhline(0, linewidth=1.2, linestyle='dotted')
    plt.vlines([420], -0.1, 0.7, linestyle='dotted', linewidth=1.2)
    plt.vlines([500], -0.1, 0.7, linestyle='dotted', linewidth=1.2)
    plt.vlines([600], -0.1, 0.7, linestyle='dotted', linewidth=1.2)
    plt.vlines([680], -0.1, 0.7, linestyle='dotted', linewidth=1.2)
    plt.vlines([776], -0.1, 0.7, linestyle='dotted', linewidth=1.2)
    plt.vlines([900], -0.1, 0.7, linestyle='dotted', linewidth=1.2)

    plt.axvspan(400, 1000, color=color4, alpha=0.5)
    plt.axvspan(420, 500, color=color2, alpha=0.5)
    plt.axvspan(500, 600, color=color1, alpha=0.5)
    plt.axvspan(600, 680, color=color2, alpha=0.5)
    plt.axvspan(680, 776, color=color3, alpha=0.5)
    plt.axvspan(776, 900, color=color2, alpha=0.5)

    # 锚点坐标
    annotations = [(425, 0.015, 'P1', (435, -0.045)),
                   (554, 0.000, 'P2', (530, -0.06)),
                   (678, 0.000, 'P3', (688, -0.06)),
                   (717, 0.345, 'P4', (685, 0.395)),
                   (778, 0.000, 'P5', (795, 0.050))]

    for x, y, label, text_position in annotations:  # 锚点文本
        plt.annotate(label, xy=(x, y), xytext=text_position, color=color5, textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=color5))

    # 光谱曲线
    for sample_index in selected_samples:
        original_spectrum = spectrum_data[sample_index, :]
        smoothed_spectrum = smoothed_data[sample_index, :]
        derivatives_spectrum = derivatives[sample_index, :]
        plt.plot(wavelength_columns, original_spectrum, '-.', linewidth=1.2)
        plt.plot(wavelength_columns, smoothed_spectrum, '-', linewidth=1.0)
        plt.plot(wavelength_columns, derivatives_spectrum * 10, '--', linewidth=1.0)

    plt.title('Example of spectral curve division', title_font, pad=8)
    plt.xlim(400, 1000)
    plt.ylim(-0.1, 0.6)
    plt.xticks(np.arange(400, 1001, 50))
    plt.yticks(np.arange(-0.1, 0.61, 0.1))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Values of spectral reflectance and FOD')
    ax.plot([], [], 'k-.', label='Original spectral curve')   # 图例
    ax.plot([], [], 'k-', label='Smoothed spectral curve')
    ax.plot([], [], 'k--', label='Spectral FOD curve')
    plt.legend(loc='upper left', borderaxespad=0.8, prop={'size': 11})
    plt.show()

    # 保存图像为PDF
    pdf = PdfPages('Figure/Division.pdf')
    pdf.savefig(fig)
    pdf.close()

# 文件路径、选定样本
input_data = 'Data/NC_Data.xlsx'
output_sav = 'Data/NC_SAV.xlsx'
output_der = 'Data/NC_FOD.xlsx'
selected_samples = [236, 350, 402] 

# 导出文件
data_process(input_data, output_sav, output_der, selected_samples)