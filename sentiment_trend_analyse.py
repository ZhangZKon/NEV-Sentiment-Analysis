import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import pathlib
import webbrowser
from tkinter import filedialog
import tkinter as tk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from snownlp import SnowNLP
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import ast
import json
import logging
import warnings
from config import SentimentConfig
from analyzer.new_sentiment_analyzer import SentimentAnalyzer
from analyzer.entity_recognizer import EntityRecognizer
from analyzer.attack_analyzer import BrandCentricAnalyzer

warnings.filterwarnings('ignore')


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class EnhancedSentimentAnalyzer:
    """
    增强版情感分析器，集成品牌和关键词双重分析：
    - 极性分类（正面、负面、中性）
    - 品牌分析
    - 关键词分析
    - 品牌+关键词组合分析
    - 交互式可视化
    - 趋势预测
    - 异常检测
    - 预警系统
    """
    def __init__(self):
        self.df = None
        self.pivot_df = None
        self.brand_keyword_pivot_df = None
        self.exploded_df = None
        self.prediction_results = {}
        self.anomalies = {}
        self.alerts = []
        self.content_cols = ['评论内容', 'comment', 'content', '正文', '文本']
        self.alpha = SentimentConfig.LIKES_WEIGHT  # 点赞权重系数
        self.beta = SentimentConfig.SHARE_WEIGHT  # 转发权重系数
        self.external_sentiment_analyzer = SentimentAnalyzer()
        self.external_entity_recognizer = EntityRecognizer()
        self.external_attack_analyzer = BrandCentricAnalyzer()
        

    def load_data_via_dialog(self) -> Optional[pd.DataFrame]:
        """通过对话框加载数据文件"""
        root = tk.Tk()
        root.withdraw()
        try:
            file_path = filedialog.askopenfilename(
                title="请选择CSV文件",
                filetypes=[("CSV文件", "*.csv")]
            )
            root.destroy()  # 无论成功或取消，都销毁窗口

            if not file_path:
                print("未选择文件")
                return None

            try:
                # 直接读取到实例变量
                temp_df = pd.read_csv(file_path, encoding='utf-8')
                self.df = self.preprocess_data(temp_df)
                logging.info(f"成功加载数据文件: {file_path}")
                return True
            except Exception as e:
                logging.error(f"加载数据文件失败: {e}")
                return None
        # 双保险：任何异常也销毁窗口
        except Exception as e:
            root.destroy()  # 异常安全
            logging.error(f"对话框异常: {e}")
            return None

    def select_brand_gui(self, brand_list: list) -> Optional[str]:
        """通过下拉菜单选择品牌"""
        selected = {"value": None}

        def on_select():
            selected["value"] = var.get()
            logging.info(f"用户选择的品牌: {selected['value']}")
            window.destroy()
            root.quit()

        def on_close():
            logging.warning("用户关闭了品牌选择窗口")
            root.quit()

        root = tk.Tk()
        root.withdraw()
        window = tk.Toplevel(root)
        window.title("请选择品牌")
        window.geometry("300x120")
        window.protocol("WM_DELETE_WINDOW", on_close)

        tk.Label(window, text="请选择要分析的品牌：").pack(pady=10)
        var = tk.StringVar(window)
        var.set(brand_list[0])
        option_menu = tk.OptionMenu(window, var, *brand_list)
        option_menu.pack()

        tk.Button(window, text="确认", command=on_select).pack(pady=10)
        root.mainloop()

        return selected["value"]

    def select_keyword_gui(self, keyword_list: list) -> Optional[str]:
        """通过下拉菜单选择关键词"""
        selected = {"value": None}

        def on_select():
            selected["value"] = var.get()
            logging.info(f"用户选择的关键词: {selected['value']}")
            window.destroy()
            root.quit()

        def on_close():
            logging.warning("用户关闭了关键词选择窗口")
            root.quit()

        root = tk.Tk()
        root.withdraw()
        window = tk.Toplevel(root)
        window.title("请选择关键词")
        window.geometry("300x120")
        window.protocol("WM_DELETE_WINDOW", on_close)

        tk.Label(window, text="请选择要分析的关键词：").pack(pady=10)
        var = tk.StringVar(window)
        var.set(keyword_list[0])
        option_menu = tk.OptionMenu(window, var, *keyword_list)
        option_menu.pack()

        tk.Button(window, text="确认", command=on_select).pack(pady=10)
        root.mainloop()

        return selected["value"]
    
    def get_content_column(self, df):
        """获取内容列名（自动匹配候选列）"""
        return next((col for col in self.content_cols if col in df.columns), None)
    
    def preprocess_data(self, df: pd.DataFrame, use_external_analyzer: bool = False) -> pd.DataFrame:
        """数据预处理(添加权重计算)"""
        # 处理时间列
        time_cols = ['发布时间', 'comment_time', 'date', 'time','create_date_time']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df["日期"] = df[time_col].dt.date
        else:
            today = datetime.today()
            df['日期'] = [today - timedelta(days=len(df)-i-1) for i in range(len(df))]

        # 识别点赞和转发列（扩展列名识别范围）
        like_cols = ['liked_count', '点赞数', 'likes', '点赞', 'like_count', '点赞量', '点赞次数']
        share_cols = ['shared_count', '转发数', 'shares', '转发', 'share_count', '转发量', '转发次数']

        like_col = next((col for col in like_cols if col in df.columns), None)
        share_col = next((col for col in share_cols if col in df.columns), None)

        # 记录识别到的列名，便于调试
        logging.info(f"识别到点赞列: {like_col}, 转发列: {share_col}")

        # 标准化点赞和转发数到[0,1]
        if like_col:
            df[like_col] = pd.to_numeric(df[like_col], errors='coerce').fillna(0)
            max_like = df[like_col].max()
            if max_like > 0:
                df['like_norm'] = df[like_col] / max_like
            else:
                df['like_norm'] = 0
        else:
            df['like_norm'] = 0

        if share_col:
            df[share_col] = pd.to_numeric(df[share_col], errors='coerce').fillna(0)
            max_share = df[share_col].max()
            if max_share > 0:
                df['share_norm'] = df[share_col] / max_share
            else:
                df['share_norm'] = 0
        else:
            df['share_norm'] = 0

        # 计算权重：weight = 1 + α*likes + β*shares
        df['weight'] = 1 + self.alpha * df['like_norm'] + self.beta * df['share_norm']
        
        # 处理极性分类
        if "极性分类" in df.columns:
            df["极性"] = df["极性分类"].str.extract(r"(正面|负面|中性)")
        else:
            content_col = self.get_content_column(df)
            if not content_col:
                raise ValueError("找不到内容列")

            if not content_col:
                logging.warning(f"未找到内容列，必须包含以下任意一列：{self.content_cols}")
                return df

            if use_external_analyzer:
                print("尝试使用外部模型进行情感分析...")
                try:
                    df, _ = self.external_sentiment_analyzer.analyze_batch(df, mode=1)
                    if "极性分类" in df.columns:
                        df["极性"] = df["极性分类"].str.extract(r"(正面|负面|中性)")
                    else:
                        raise ValueError("极性分类列不存在")

                    if df["极性"].isna().mean() > 0.9:
                        raise ValueError("提取极性后为空过多")
                except Exception as e:
                    logging.warning(f"外部模型失败，回退至 SnowNLP：{e}")
                    df['sentiment_score'] = [
                        SnowNLP(str(text)).sentiments
                        for text in tqdm(df[content_col].astype(str), desc="情感分析")
                    ]
                    df['极性'] = df['sentiment_score'].apply(
                        lambda x: '正面' if x > 0.6 else ('负面' if x < 0.4 else '中性')
                    )
            else:
                print("使用 SnowNLP 进行情感分析...")
                df['sentiment_score'] = [
                    SnowNLP(str(text)).sentiments
                    for text in tqdm(df[content_col].astype(str), desc="情感分析")
                ]
                df['极性'] = df['sentiment_score'].apply(
                    lambda x: '正面' if x > 0.6 else ('负面' if x < 0.4 else '中性')
                )

        # brand_cols = ["品牌", "source_keyword", "brand", "品牌关键词", "来源关键词", "品牌名称"]
        # brand_col = None
        # for col in brand_cols:
        #    if col in df.columns:
        #        brand_col = col
        #        break

        # 记录使用的品牌列，便于调试
        # logging.info(f"使用品牌列: {brand_col if brand_col else '默认(全部)'}")

        # if brand_col:
        #    if brand_col != "品牌":
        #        df.rename(columns={brand_col: "品牌"}, inplace=True)
        #else:
        #    df["品牌"] = "全部"

        # 清理数据
        df = df.dropna(subset=["日期", "极性"])
        df["日期"] = pd.to_datetime(df["日期"])
        logging.info(f"数据预处理完成，共{len(df)}条记录")
        return df
    
    def extract_brand_from_content(self,df: pd.DataFrame) -> List:
        """从文本中提取实体和品牌名"""
        #处理内容列，统计实体
        content_col = self.get_content_column(df)
        if not content_col:
            raise ValueError("找不到内容列")
        
        try:
            # 使用找到的 content_col，而不是硬编码
            all_text = " ".join(df[content_col].dropna().astype(str))
            # 识别整个文本的实体
            result = self.external_entity_recognizer.extract_entities(all_text)
            entities = result.get('实体列表', [])
            print(f"成功识别到 {len(entities)} 个实体")

        except KeyError:
            logging.error(f"内容列 '{content_col}' 在DataFrame中不存在")
        except Exception as e:
            logging.error(f"实体识别失败: {str(e)}") 
        
        # 品牌处理（支持多个可能的列名）
        if entities:
            stats = self.external_entity_recognizer.get_entity_statistics(entities)
            brand_col = stats["品牌排行TOP5"]
            brands = [item["品牌"] for item in brand_col]
        else:
            print("未识别到任何品牌")
            brands = ["全部"]
        
        return brands
                       
    
    def extract_keywords_from_detail(self, detail_text: str) -> List[str]:
        """从分析详情中提取关键词"""
        keywords = []
        try:
            if pd.isna(detail_text) or detail_text == '':
                return keywords

            # 尝试解析JSON或字典格式
            if isinstance(detail_text, str):
                try:
                    detail_dict = ast.literal_eval(detail_text)
                except:
                    try:
                        detail_dict = json.loads(detail_text)
                    except:
                        return keywords
            else:
                detail_dict = detail_text

            # 提取汽车术语
            if isinstance(detail_dict, dict):
                # 尝试多种可能的键名
                term_keys = ['汽车术语识别', 'car_terms', 'keywords', 'terms']
                for key in term_keys:
                    if key in detail_dict:
                        terms_info = detail_dict[key]
                        if isinstance(terms_info, dict):
                            if '触发术语' in terms_info:
                                trigger_terms = terms_info['触发术语']
                                if isinstance(trigger_terms, dict):
                                    keywords.extend(trigger_terms.keys())
                                elif isinstance(trigger_terms, list):
                                    keywords.extend(trigger_terms)
                            elif 'terms' in terms_info:
                                terms = terms_info['terms']
                                if isinstance(terms, dict):
                                    keywords.extend(terms.keys())
                                elif isinstance(terms, list):
                                    keywords.extend(terms)
                        elif isinstance(terms_info, list):
                            keywords.extend(terms_info)
                        break

                # 如果没有找到术语，尝试其他可能的字段
                if not keywords:
                    for key, value in detail_dict.items():
                        if 'keyword' in key.lower() or 'term' in key.lower():
                            if isinstance(value, list):
                                keywords.extend(value)
                            elif isinstance(value, dict):
                                keywords.extend(value.keys())
                            elif isinstance(value, str):
                                keywords.append(value)
        except Exception as e:
            logging.warning(f"提取关键词时出错: {e}")

        return keywords

    def explode_brand_keyword(self, df: pd.DataFrame) -> pd.DataFrame:
        """展开品牌+关键词组合"""
        records = []

        # 检查是否有分析详情列
        detail_cols = ['分析详情', 'analysis_detail', 'detail', 'keywords']
        detail_col = None
        for col in detail_cols:
            if col in df.columns:
                detail_col = col
                break

        print("正在展开品牌+关键词组合...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
            brand = row['品牌'] if '品牌' in row else '全部'
            date = row['日期']
            emotion = row['极性']
            weight = row.get('weight', 1)  # 获取权重值，默认为1

            # 提取关键词
            keywords = []
            if detail_col and detail_col in row:
                keywords = self.extract_keywords_from_detail(row[detail_col])

            # 如果没有关键词，标记为"未识别"
            if not keywords:
                keywords = ["未识别"]

            # 为每个关键词创建一条记录
            for keyword in keywords:
                records.append({
                    '品牌': brand,
                    '关键词': keyword,
                    '日期': date,
                    '极性': emotion,
                    'weight': weight,  # 添加权重
                    '原始索引': idx
                })

        exploded_df = pd.DataFrame(records)
        logging.info(f"品牌+关键词展开完成，共{len(exploded_df)}条记录")
        return exploded_df

    def prepare_brand_keyword_pivot(self, exploded_df: pd.DataFrame) -> pd.DataFrame:
        """准备品牌+关键词透视表数据（加权版）"""
        
        pivot_df = (
            exploded_df.groupby(['品牌', '关键词', '日期', '极性'])
            .agg({'weight': 'sum'})  # 先聚合得到'weight'列
            .rename(columns={'weight': '加权数量'})  # 显式重命名为'加权数量'
            .reset_index()  # 重置索引
            .pivot_table(
                index=['品牌', '关键词', '日期'],
                columns='极性',
                values='加权数量',  # 现在可以正确找到该列
                fill_value=0
            )
            .reset_index()
        )

        # 确保所有极性列都存在
        for col in ["中性", "正面", "负面"]:
            if col not in pivot_df.columns:
                pivot_df[col] = 0

        pivot_df["加权总数"] = pivot_df[["中性", "正面", "负面"]].sum(axis=1)

        # 计算占比(修改除0错误)
        for col in ["正面", "中性", "负面"]:
            pivot_df[col + "占比"] = np.where(
                pivot_df["加权总数"]>0,
                pivot_df[col] / pivot_df["加权总数"],
                np.nan
            )

        return pivot_df

    def prepare_pivot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备透视表数据（仅品牌）"""
        pivot_df = (
            df.groupby(["品牌", "日期", "极性"])
            .size()
            .reset_index(name="数量")
            .pivot_table(
                index=["品牌", "日期"],
                columns="极性",
                values="数量",
                fill_value=0
            )
            .reset_index()
        )

        # 确保所有极性列都存在
        for col in ["中性", "正面", "负面"]:
            if col not in pivot_df.columns:
                pivot_df[col] = 0

        pivot_df["加权总数"] = pivot_df[["中性", "正面", "负面"]].sum(axis=1)

        # 计算占比(已修改)
        for col in ["中性", "正面", "负面"]:
            pivot_df[col + "占比"] = np.where(
                pivot_df["加权总数"]>0,
                pivot_df[col] / pivot_df["加权总数"],
                np.nan
            )

        return pivot_df

    def get_top_keywords_for_brand(self, brand: str, top_n: int = 10) -> List[str]:
        """获取品牌的热门关键词"""
        if self.exploded_df is None:
            return []

        brand_data = self.exploded_df[self.exploded_df['品牌'] == brand]
        keyword_counts = brand_data['关键词'].value_counts()

        # 过滤掉"未识别"
        keyword_counts = keyword_counts[keyword_counts.index != "未识别"]

        return keyword_counts.head(top_n).index.tolist()

    def plot_brand_keyword_trend(self, brand: str, keyword: str = None):
        """绘制品牌+关键词组合的情感趋势图"""
        if self.brand_keyword_pivot_df is None:
            logging.error("请先运行数据处理")
            return

        if keyword:
            # 品牌+关键词组合分析
            df_filtered = self.brand_keyword_pivot_df[
                (self.brand_keyword_pivot_df["品牌"] == brand) &
                (self.brand_keyword_pivot_df["关键词"] == keyword)
            ].sort_values("日期")
            title = f"{brand} - {keyword}：每日情感极性占比趋势"
            output_suffix = f"_{keyword}"
        else:
            # 仅品牌分析
            df_filtered = self.brand_keyword_pivot_df[
                self.brand_keyword_pivot_df["品牌"] == brand
            ].groupby("日期").agg({
                "正面": "sum",
                "中性": "sum",
                "负面": "sum",
                "加权总数": "sum"
            }).reset_index()

            # 重新计算占比(修改除0错误)
            for col in ["正面", "中性", "负面"]:
                df_filtered[col + "占比"] = np.where(
                    df_filtered["加权总数"]>0,
                    df_filtered[col] / df_filtered["加权总数"],
                    np.nan
                )

            title = f"{brand}：每日情感极性占比趋势（所有关键词）"
            output_suffix = "_all_keywords"

        if df_filtered.empty:
            print(f"未找到品牌 {brand} 和关键词 {keyword} 的数据")
            return

        fig = go.Figure()

        # 添加趋势线
        fig.add_trace(go.Scatter(
            x=df_filtered["日期"],
            y=df_filtered["正面占比"],
            mode='lines+markers',
            name='正面占比',
            line=dict(color='#28a745', width=2),
            hovertemplate='日期: %{x}<br>正面占比: %{y:.1%}'
        ))

        fig.add_trace(go.Scatter(
            x=df_filtered["日期"],
            y=df_filtered["中性占比"],
            mode='lines+markers',
            name='中性占比',
            line=dict(color='#ffc107', width=2),
            hovertemplate='日期: %{x}<br>中性占比: %{y:.1%}'
        ))

        fig.add_trace(go.Scatter(
            x=df_filtered["日期"],
            y=df_filtered["负面占比"],
            mode='lines+markers',
            name='负面占比',
            line=dict(color='#dc3545', width=2),
            hovertemplate='日期: %{x}<br>负面占比: %{y:.1%}'
        ))

        # 添加预警线
        fig.add_hline(
            y=SentimentConfig.NEGATIVE_THRESHOLD,
            line_dash="dash",
            line_color="#dc3545",
            annotation_text="负面预警线"
        )

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="情感极性占比",
            yaxis=dict(tickformat=".0%"),
            hovermode="x unified",
            height=600,
            showlegend=True
        )

        fig.show(renderer="browser")

        # 保存图表
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output", brand)
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(f"{output_dir}/趋势图{output_suffix}.html")
        fig.write_image(f"{output_dir}/趋势图{output_suffix}.png")
        logging.info(f"已保存品牌 {brand} 的趋势图到 {output_dir}")

    def plot_keyword_comparison(self, brand: str, keywords: List[str] = None):
        """绘制关键词对比图"""
        if self.brand_keyword_pivot_df is None:
            logging.error("请先运行数据处理")
            return

        if not keywords:
            keywords = self.get_top_keywords_for_brand(brand, 5)

        if not keywords:
            print(f"品牌 {brand} 没有足够的关键词数据")
            return

        fig = make_subplots(
            rows=len(keywords), cols=1,
            subplot_titles=[f"{kw} 负面情感趋势" for kw in keywords],
            vertical_spacing=0.1
        )

        for i, keyword in enumerate(keywords):
            df_kw = self.brand_keyword_pivot_df[
                (self.brand_keyword_pivot_df["品牌"] == brand) &
                (self.brand_keyword_pivot_df["关键词"] == keyword)
            ].sort_values("日期")

            if df_kw.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=df_kw["日期"],
                    y=df_kw["负面占比"],
                    mode='lines+markers',
                    name=f'{keyword} 负面占比',
                    line=dict(color='#dc3545', width=2),
                    showlegend=i == 0
                ),
                row=i+1, col=1
            )

        fig.update_layout(
            title=f"{brand} 关键词负面情感对比",
            height=200 * len(keywords),
            showlegend=True
        )

        fig.update_xaxes(title_text="日期", row=len(keywords), col=1)
        fig.update_yaxes(title_text="负面占比", tickformat=".0%")
        fig.show(renderer="browser")

        # 保存图表
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output", brand)
        os.makedirs(output_dir, exist_ok=True)
        print("保存路径：", os.path.abspath(output_dir))
        fig.write_html(f"{output_dir}/关键词对比图.html")
        fig.write_image(f"{output_dir}/关键词对比图.png")
        logging.info(f"已保存品牌 {brand} 的关键词对比图到 {output_dir}")

    def plot_emotion_pie_by_brand_keyword(self, brand: str, keyword: str = None, mode: str = "30d"):
        """绘制品牌+关键词组合的情感分布饼图"""
        if self.exploded_df is None:
            logging.error("请先运行数据处理")
            return

        end_date = self.exploded_df["日期"].max()
        if mode == "30d":
            start_date = end_date - pd.Timedelta(days=29)
            title_suffix = "最近30天"
        elif mode == "7d":
            start_date = end_date - pd.Timedelta(days=6)
            title_suffix = "最近7天"
        else:
            raise ValueError("模式必须为 '30d' 或 '7d'")

        if keyword:
            df_filtered = self.exploded_df[
                (self.exploded_df["品牌"] == brand) &
                (self.exploded_df["关键词"] == keyword) &
                (self.exploded_df["日期"] >= start_date) &
                (self.exploded_df["日期"] <= end_date)
            ]
            title = f"{brand} - {keyword} 在{title_suffix}的情感极性分布"
            output_suffix = f"_{keyword}_{mode}"
        else:
            df_filtered = self.exploded_df[
                (self.exploded_df["品牌"] == brand) &
                (self.exploded_df["日期"] >= start_date) &
                (self.exploded_df["日期"] <= end_date)
            ]
            title = f"{brand} 在{title_suffix}的情感极性分布（所有关键词）"
            output_suffix = f"_all_{mode}"

        if df_filtered.empty:
            print(f"在指定时间范围内未找到相关数据")
            return

        counts = df_filtered["极性"].value_counts()

        # 设置颜色
        colors = {'正面': '#28a745', '中性': '#ffc107', '负面': '#dc3545'}
        color_sequence = [colors.get(emotion, '#6c757d') for emotion in counts.index]

        fig = px.pie(
            counts,
            names=counts.index,
            values=counts.values,
            title=title,
            color_discrete_sequence=color_sequence
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}: %{value}条<br>占比: %{percent}'
        )

        fig.show(renderer="browser")

        # 保存图表
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output", brand)
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(f"{output_dir}/情感饼图{output_suffix}.html")
        fig.write_image(f"{output_dir}/情感饼图{output_suffix}.png")
        logging.info(f"已保存品牌 {brand} 的情感饼图到 {output_dir}")

    def get_time_features(self, dates: pd.Series) -> np.ndarray:
        """根据日期提取周期性特征（日周期/周周期）"""
        day_of_week = dates.dt.dayofweek  # 0-6
        sin_dow = np.sin(2 * np.pi * day_of_week / 7)
        cos_dow = np.cos(2 * np.pi * day_of_week / 7)

        # 可扩展加月周期
        day = dates.dt.day
        sin_dom = np.sin(2 * np.pi * day / 30.5)
        cos_dom = np.cos(2 * np.pi * day / 30.5)

        return np.stack([sin_dow, cos_dow, sin_dom, cos_dom], axis=1)

    def prepare_lstm_data(self, y, dates, look_back=10):
        """准备LSTM训练数据"""
        if y.ndim > 1:
            y = y.flatten()

        # 主变量归一化
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))  # shape: (N, 1)

        # 时间周期特征：sin/cos of weekday + day of month
        time_features = self.get_time_features(dates)  # shape: (N, 4)
        scaler_time = MinMaxScaler()
        time_scaled = scaler_time.fit_transform(time_features)

        X, Y = [], []
        for i in range(look_back, len(y_scaled)):
            # shape = (look_back, 1)
            y_seq = y_scaled[i - look_back:i].reshape(-1, 1)
            # shape = (look_back, 4)
            t_seq = time_scaled[i - look_back:i]
            # 合并成 (look_back, 5)
            seq = np.concatenate([y_seq, t_seq], axis=1)
            X.append(seq)
            Y.append(y_scaled[i, 0])  # scalar target

        return np.array(X), np.array(Y), scaler_y

    def train_lstm(self, X, Y, epochs=100, batch_size=32, learning_rate=0.001):
        """训练LSTM模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(Y).unsqueeze(-1)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=2, dropout=0.2)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

        return model, device

    def predict_with_lstm(self, model, device, y, dates, scaler, days, look_back=10):
        """使用LSTM进行预测"""
        model.eval()

        if y.ndim > 1:
            y = y.flatten()

        y_scaled = scaler.transform(y.reshape(-1, 1))
        current_y = y_scaled[-look_back:].reshape(-1, 1)

        last_date = dates.iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days)
        all_dates = pd.concat([dates, pd.Series(future_dates)]).reset_index(drop=True)

        time_features = self.get_time_features(all_dates)
        scaler_time = MinMaxScaler()
        time_scaled = scaler_time.fit_transform(time_features)

        predictions = []
        with torch.no_grad():
            for i in range(days):
                current_time = time_scaled[len(y_scaled) - look_back + i: len(y_scaled) + i]
                current_seq = np.concatenate([current_y, current_time], axis=1)

                x = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
                pred = model(x)
                pred_value = pred.cpu().numpy()[0, 0]

                predictions.append(pred_value)
                current_y = np.vstack([current_y[1:], [[pred_value]]])

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

    def calculate_confidence_interval(self, model, device, X, Y, scaler, alpha=0.05):
        """计算置信区间"""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = model(X_tensor).cpu().numpy()

        residuals = Y.reshape(-1, 1) - predictions
        residual_std = np.std(residuals)

        from scipy.stats import t
        n = len(Y)
        t_value = t.ppf(1 - alpha/2, n-1)

        return residual_std * t_value

    def predict_sentiment_trend_brand_keyword(self, brand: str, keyword: str = None, days: int = 30, model_type: str = 'lstm'):
        """预测品牌+关键词组合的情感趋势"""
        if self.brand_keyword_pivot_df is None:
            logging.error("请先运行数据处理")
            return None

        if keyword:
            df_filtered = self.brand_keyword_pivot_df[
                (self.brand_keyword_pivot_df["品牌"] == brand) &
                (self.brand_keyword_pivot_df["关键词"] == keyword)
            ].sort_values("日期")
            target_key = f"{brand}_{keyword}"
        else:
            # 聚合所有关键词
            df_filtered = self.brand_keyword_pivot_df[
                self.brand_keyword_pivot_df["品牌"] == brand
            ].groupby("日期").agg({
                "正面": "sum",
                "中性": "sum",
                "负面": "sum",
                "加权总数": "sum"
            }).reset_index()

            for col in ["正面", "中性", "负面"]:
                df_filtered[col + "占比"] = df_filtered[col] / df_filtered["加权总数"]

            target_key = f"{brand}_all"

        if len(df_filtered) < 15:
            logging.warning(f"数据点太少，无法进行可靠预测")
            return None

        results = {}
        for sentiment in ['正面占比', '中性占比', '负面占比']:
            y = df_filtered[sentiment].values
            if len(y) < 15:
                continue

            try:
                if model_type == 'lstm':
                    look_back = min(10, len(y) // 3)
                    if len(y) < look_back + 5:
                        logging.warning(f"数据不足以进行LSTM预测: {sentiment}")
                        continue

                    dates = df_filtered["日期"]
                    X, Y, scaler = self.prepare_lstm_data(y, dates, look_back)

                    if len(X) < 5:
                        logging.warning(f"LSTM训练数据不足: {sentiment}")
                        continue

                    print(f"正在训练 {sentiment} 的LSTM模型...")
                    model, device = self.train_lstm(X, Y, epochs=100, batch_size=min(16, len(X)))
                    pred = self.predict_with_lstm(model, device, y, dates, scaler, days, look_back)

                    confidence_margin = self.calculate_confidence_interval(model, device, X, Y, scaler)
                    ci_lower = pred - confidence_margin
                    ci_upper = pred + confidence_margin
                    ci = np.column_stack([ci_lower, ci_upper])

                    # 其他模型类型的代码保持不变...
                    pred = np.clip(pred, 0, 1)
                    if isinstance(ci, np.ndarray):
                        ci = np.clip(ci, 0, 1)

                    future_dates = pd.date_range(
                        df_filtered["日期"].iloc[-1] + pd.Timedelta(days=1),
                        periods=days
                    )

                    results[sentiment] = {
                        'predicted': pred,
                        'confidence_interval': ci,
                        'dates': future_dates
                    }

                    logging.info(f"成功预测 {sentiment}")

            except Exception as e:
                logging.error(f"预测 {sentiment} 时出错: {e}")
                continue

        self.prediction_results[target_key] = results
        return results

    def plot_prediction_results_brand_keyword(self, brand: str, keyword: str = None):
        """绘制品牌+关键词组合的预测结果"""
        target_key = f"{brand}_{keyword}" if keyword else f"{brand}_all"
        if target_key not in self.prediction_results:
            logging.error(f"未找到预测结果")
            return

        results = self.prediction_results[target_key]

        if keyword:
            df_filtered = self.brand_keyword_pivot_df[
                (self.brand_keyword_pivot_df["品牌"] == brand) &
                (self.brand_keyword_pivot_df["关键词"] == keyword)
            ].sort_values("日期")
            title = f"{brand} - {keyword} 情感趋势预测分析"
        else:
            df_filtered = self.brand_keyword_pivot_df[
                self.brand_keyword_pivot_df["品牌"] == brand
            ].groupby("日期").agg({
                "正面": "sum",
                "中性": "sum",
                "负面": "sum",
                "加权总数": "sum"
            }).reset_index()

            for col in ["正面", "中性", "负面"]:
                df_filtered[col + "占比"] = df_filtered[col] / df_filtered["加权总数"]

            title = f"{brand} 情感趋势预测分析（所有关键词）"

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('正面情感趋势预测', '中性情感趋势预测', '负面情感趋势预测'),
            vertical_spacing=0.1
        )

        colors = {'正面占比': '#28a745', '中性占比': '#ffc107', '负面占比': '#dc3545'}
        color_rgba = {'正面占比': 'rgba(40, 167, 69, 0.2)', '中性占比': 'rgba(255, 193, 7, 0.2)', '负面占比': 'rgba(220, 53, 69, 0.2)'}
        row_map = {'正面占比': 1, '中性占比': 2, '负面占比': 3}

        for sentiment, result in results.items():
            row = row_map[sentiment]
            color = colors[sentiment]

            # 历史数据
            fig.add_trace(
                go.Scatter(
                    x=df_filtered["日期"],
                    y=df_filtered[sentiment],
                    mode='lines+markers',
                    name=f'历史{sentiment}',
                    line=dict(color=color, width=2),
                    showlegend=True if row == 1 else False
                ),
                row=row, col=1
            )

            # 预测数据
            fig.add_trace(
                go.Scatter(
                    x=result['dates'],
                    y=result['predicted'],
                    mode='lines+markers',
                    name=f'预测{sentiment}',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=True if row == 1 else False
                ),
                row=row, col=1
            )

            # 置信区间
            if isinstance(result['confidence_interval'], np.ndarray):
                fig.add_trace(
                    go.Scatter(
                        x=result['dates'],
                        y=result['confidence_interval'][:, 1],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=row, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=result['dates'],
                        y=result['confidence_interval'][:, 0],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=color_rgba[sentiment],
                        name=f'置信区间' if row == 1 else None,
                        showlegend=True if row == 1 else False
                    ),
                    row=row, col=1
                )

        fig.update_layout(
            title=title,
            height=800,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="占比", tickformat=".0%")

        fig.show(renderer="browser")

        # 保存图表
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output", brand)
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{keyword}" if keyword else "_all"
        fig.write_html(f"{output_dir}/情感预测图{suffix}.html")
        fig.write_image(f"{output_dir}/情感预测图{suffix}.png")
        logging.info(f"已保存预测图表到 {output_dir}")

    def generate_brand_keyword_report(self, brand: str, keyword: str = None):
        """生成品牌+关键词分析报告"""
        report = []

        report.append("=" * 60)
        report.append("品牌+关键词情感分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"分析品牌: {brand}")
        if keyword:
            report.append(f"分析关键词: {keyword}")
        else:
            report.append("分析范围: 所有关键词")
        report.append("")

        # 数据概览
        if self.exploded_df is not None:
            if keyword:
                data_filtered = self.exploded_df[
                    (self.exploded_df['品牌'] == brand) &
                    (self.exploded_df['关键词'] == keyword)
                ]
            else:
                data_filtered = self.exploded_df[self.exploded_df['品牌'] == brand]

            report.append("数据概览:")
            report.append(f" 总记录数: {len(data_filtered)}")
            report.append(f" 时间范围: {data_filtered['日期'].min()} 至 {data_filtered['日期'].max()}")

            emotion_counts = data_filtered['极性'].value_counts()
            for emotion, count in emotion_counts.items():
                pct = count / len(data_filtered) * 100
                report.append(f"  {emotion}: {count}条 ({pct:.1f}%)")
            report.append("")

        # 热门关键词（当分析所有关键词时）
        if not keyword:
            top_keywords = self.get_top_keywords_for_brand(brand, 10)
            if top_keywords:
                report.append("热门关键词:")
                for i, kw in enumerate(top_keywords, 1):
                    kw_data = self.exploded_df[
                        (self.exploded_df['品牌'] == brand) &
                        (self.exploded_df['关键词'] == kw)
                    ]
                    report.append(f"  {i}. {kw} ({len(kw_data)}条)")
                report.append("")

        # 预测结果
        target_key = f"{brand}_{keyword}" if keyword else f"{brand}_all"
        if target_key in self.prediction_results:
            report.append("未来30天预测:")
            results = self.prediction_results[target_key]
            for sentiment, result in results.items():
                avg_pred = np.mean(result['predicted'])
                report.append(f"  {sentiment}平均值: {avg_pred:.1%}")
            report.append("")

        # 预警信息
        brand_alerts = [alert for alert in self.alerts if alert.get('brand') == brand]
        if brand_alerts:
            report.append("预警信息:")
            for alert in brand_alerts:
                report.append(f"  [{alert['level']}] {alert['message']}")
            report.append("")

        report_text = "\n".join(report)

        # 保存报告
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = pathlib.Path(base_dir) / "output" / brand
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{keyword}" if keyword else "_all"
        report_path = f"{output_dir}/brand_keyword_report{suffix}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        logging.info(f"分析报告已保存至: {report_path}")
        return report_text

    def analyze_brand_keyword_combination(self, brand: str, keyword: str = None):
        """分析品牌整体情况"""
        print(f"\n=== 分析品牌：{brand}" + (f" + 关键词：{keyword}" if keyword else " (所有关键词)") + " ===")
        
        try:
        # 1. 趋势分析
            print("1. 生成情感趋势图...")
            self.plot_brand_keyword_trend(brand, keyword)
        
        except Exception as e:
            print(f"❗ 分析中断！错误信息: {str(e)}")
            
        # 2. 分布分析
        print("2. 生成情感分布饼图...")
        self.plot_emotion_pie_by_brand_keyword(brand, keyword, "7d")
        self.plot_emotion_pie_by_brand_keyword(brand, keyword, "30d")

        # 3. 趋势预测
        print("3. 进行趋势预测...")
        self.predict_sentiment_trend_brand_keyword(brand, keyword, days=30, model_type='lstm')
        self.plot_prediction_results_brand_keyword(brand, keyword)

        # 4. 生成报告
        print("4. 生成分析报告...")
        self.generate_brand_keyword_report(brand, keyword)

        print(f"=== 品牌{brand}" + (f"+关键词{keyword}" if keyword else "(所有关键词)") + "分析完成 ===")

    def run_dual_analysis(self, brand: str = None):
        """运行品牌+关键词双重分析"""
        print("=== 品牌+关键词双重情感分析系统 ===")

        # 1. 加载数据(修改后)
        print("1. 加载数据...")
        if not self.load_data_via_dialog():
            print("数据加载失败，终止分析")
            return

        print(f"数据加载完成，共 {len(self.df)} 条记录")

        # 2. 展开品牌+关键词组合
        print("2. 处理品牌+关键词组合...")
        self.exploded_df = self.explode_brand_keyword(self.df)
        self.brand_keyword_pivot_df = self.prepare_brand_keyword_pivot(self.exploded_df)
        print(f"品牌+关键词组合处理完成")

        # 3. 选择品牌（使用您提供的逻辑）
        brands = self.extract_brand_from_content(self.df)
        # brands = self.df['品牌'].dropna().unique().tolist()
        if brand and brand in brands:
            target_brands = [brand]
        else:
            print("3. 选择品牌...")
            selected = self.select_brand_gui(["全部"] + brands)
            if not selected:
                print("未选择品牌，流程中断")
                return
            target_brands = brands if selected == "全部" else [selected]

        # 4. 对每个目标品牌进行分析
        for selected_brand in target_brands:
            print(f"\n--- 开始分析品牌: {selected_brand} ---")


            # 选择分析模式
            print("4. 选择分析模式...")
            analysis_modes = ["品牌整体分析", "品牌+特定关键词分析", "品牌关键词对比分析"]

            root = tk.Tk()
            root.withdraw()
            window = tk.Toplevel(root)
            window.title(f"选择 {selected_brand} 的分析模式")
            window.geometry("400x200")

            selected_mode = {"value": None}

            def on_mode_select(mode):
                selected_mode["value"] = mode
                window.destroy()
                root.quit()

            tk.Label(window, text=f"请选择 {selected_brand} 的分析模式：", font=("Arial", 12)).pack(pady=10)

            for mode in analysis_modes:
                tk.Button(
                    window,
                    text=mode,
                    command=lambda m=mode: on_mode_select(m),
                    width=30
                ).pack(pady=5)

            root.mainloop()

            if not selected_mode["value"]:
                print(f"未选择品牌 {selected_brand} 的分析模式，跳过")
                continue

            # 5. 根据选择的模式执行分析
            if selected_mode["value"] == "品牌整体分析":
                self.analyze_brand_keyword_combination(selected_brand, None)

            elif selected_mode["value"] == "品牌+特定关键词分析":
                # 选择关键词
                keywords = self.get_top_keywords_for_brand(selected_brand, 20)
                if not keywords:
                    print(f"品牌 {selected_brand} 没有可分析的关键词")
                    continue

                print("5. 选择关键词...")
                selected_keyword = self.select_keyword_gui(keywords)
                if not selected_keyword:
                    print("未选择关键词，跳过该品牌")
                    continue

                self.analyze_brand_keyword_combination(selected_brand, selected_keyword)

            elif selected_mode["value"] == "品牌关键词对比分析":
                # 关键词对比分析
                print("5. 生成关键词对比分析...")
                self.plot_keyword_comparison(selected_brand)

                # 为每个热门关键词生成报告
                top_keywords = self.get_top_keywords_for_brand(selected_brand, 5)
                for keyword in top_keywords:
                    self.analyze_brand_keyword_combination(selected_brand, keyword)

            print(f"--- 品牌 {selected_brand} 分析完成 ---")

        print("\n=== 双重分析完成 ===")



if __name__ == "__main__":
    analyzer = EnhancedSentimentAnalyzer()
    analyzer.run_dual_analysis()

