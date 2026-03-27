import matplotlib.pyplot as plt
import plotly.io as pio
import webbrowser

# 配置参数
class SentimentConfig:
    PREDICT_DAYS = 30  # 预测未来天数
    NEGATIVE_THRESHOLD = 0.3  # 负面情感阈值（占比）
    POSITIVE_THRESHOLD = 0.6  # 正面情感阈值（占比）
    LIKES_WEIGHT = 0.1  # 点赞权重系数
    SHARE_WEIGHT = 0.5  # 分享权重系数
    ANOMALY_WINDOW = 7  # 异常检测窗口
    CONFIDENCE_LEVEL = 0.95  # 置信区间
    
    # 设置中文字体和浏览器
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'STHeitiSC-Light']
    plt.rcParams['axes.unicode_minus'] = False
    pio.renderers.default = "browser"
    webbrowser.register('safari', None, webbrowser.GenericBrowser('/Applications/Safari.app/Contents/MacOS/Safari'))
