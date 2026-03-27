# -*- coding: utf-8 -*-
import os
import re
import math
import time
import datetime
import pandas as pd
import jieba
import torch
from tkinter import Tk, filedialog, simpledialog
from collections import Counter, defaultdict
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#后续要更改为从词典打开
AUTO_TERM_DICT = {
    '新能源': ['新能源', '电动', '混动', '纯电'],
    '汽车': ['汽车', '车辆', '车子', '座驾'],
    '续航': ['续航', '里程', '电量'],
    '充电': ['充电', '充电桩', '快充'],
    '性能': ['性能', '动力', '加速'],
    '外观': ['外观', '颜值', '设计'],
    '内饰': ['内饰', '配置', '舒适'],
    '价格': ['价格', '价位', '性价比'],
    '服务': ['服务', '售后', '维修'],
    'New Energy': ['new energy', 'electric', 'hybrid', 'pure electric'],
    'Automobile': ['car', 'vehicle', 'ride', 'automobile'],
    'Range': ['range', 'mileage', 'battery level'],
    'Charging': ['charging', 'charging station', 'fast charging'],
    'Performance': ['performance', 'power', 'acceleration'],
    'Exterior': ['exterior', 'appearance', 'design'],
    'Interior': ['interior', 'features', 'comfort'],
    'Price': ['price', 'price range', 'cost performance'],
    'Service': ['service', 'after-sales', 'maintenance']
}


EMOTION_MAP = {
    "admiration": "钦佩",
    "amusement": "娱乐",
    "anger": "愤怒",
    "annoyance": "恼怒",
    "approval": "赞同",
    "caring": "关心",
    "confusion": "困惑",
    "curiosity": "好奇",
    "desire": "渴望",
    "disappointment": "失望",
    "disapproval": "不赞同",
    "disgust": "厌恶",
    "embarrassment": "尴尬",
    "excitement": "兴奋",
    "fear": "恐惧",
    "gratitude": "感激",
    "grief": "悲痛",
    "joy": "喜悦",
    "love": "爱",
    "nervousness": "紧张",
    "optimism": "乐观",
    "pride": "自豪",
    "realization": "领悟",
    "relief": "如释重负",
    "remorse": "懊悔",
    "sadness": "悲伤",
    "surprise": "惊讶",
    "neutral": "中立"
}


class SentimentAnalyzer:
    def __init__(self):
        self._load_models()

    def _load_models(self):
        self.emo_pipeline = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained("SchuylerH/bert-multilingual-go-emtions"),
            tokenizer=AutoTokenizer.from_pretrained("SchuylerH/bert-multilingual-go-emtions"),
            top_k=5,
            device=0
        )

        self.zh_pipeline = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-full-chinese"),
            tokenizer=AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-full-chinese"),
            top_k=None,
            device=0
        )

        self.en_pipeline = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment"),
            tokenizer=AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment"),
            top_k=None,
            device=0
        )

        zh_sentence_model = SentenceTransformer("shibing624/text2vec-base-chinese")
        zh_sentence_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.zh_kw = KeyBERT(zh_sentence_model)

        en_sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        en_sentence_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.en_kw = KeyBERT(en_sentence_model)

    def _detect_language(self, text: str) -> str:
        return detect(text)

    def _split_text(self, text, max_len=500):
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]

    def _compute_confidence(self, emotion_scores, polarity_score, w1=0.6, w2=0.4):
        scores = [e["score"] for e in emotion_scores]
        entropy = -sum(p * math.log(p + 1e-12) for p in scores)
        max_entropy = math.log(len(scores))
        entropy_conf = 1 - entropy / max_entropy if max_entropy > 0 else 0
        final_score = round(w1 * polarity_score + w2 * entropy_conf, 4)
        
        return {
            "置信度评分": final_score,
        }

    def _classify_polarity_zh(self, text: str) -> dict:
        raw = self.zh_pipeline(text)
        candidates = raw[0]
        polarity_scores = {"正面": 0.0, "中性": 0.0, "负面": 0.0}
        for item in candidates:
            label = item["label"].lower().replace("star", "").strip()
            try:
                star = int(label)
            except:
                continue
            if star <= 2:
                polarity_scores["负面"] += item["score"]
            elif star == 3:
                polarity_scores["中性"] += item["score"]
            else:
                polarity_scores["正面"] += item["score"]
        main_polarity = max(polarity_scores, key=polarity_scores.get)
        return {
            "语言": "zh",
            "极性分类": main_polarity,
            "极性置信度": round(polarity_scores[main_polarity], 4),
            "极性细分": {k: round(v, 4) for k, v in polarity_scores.items()}
        }

    def _classify_polarity_en(self, text: str) -> dict:
        raw = self.en_pipeline(text)[0]
        polarity_scores = {"正面": 0.0, "中性": 0.0, "负面": 0.0}
        label_map = {
            "LABEL_0": "负面", "LABEL_1": "中性", "LABEL_2": "正面",
            "negative": "负面", "neutral": "中性", "positive": "正面"
        }
        for item in raw:
            label = label_map.get(item["label"], "中性")
            polarity_scores[label] += item["score"]
        main_polarity = max(polarity_scores, key=polarity_scores.get)
        return {
            "语言": "en",
            "极性分类": main_polarity,
            "极性置信度": round(polarity_scores[main_polarity], 4),
            "极性细分": {k: round(v, 4) for k, v in polarity_scores.items()}
        }

    def _extract_auto_terms_zh(self, text: str) -> dict:
        tokens = jieba.lcut(text)
        matched = defaultdict(list)
        for theme, keywords in AUTO_TERM_DICT.items():
            for kw in keywords:
                if kw in tokens:
                    matched[theme].append(kw)
        return {
            "语言": "zh",
            "触发主题数量": len(matched),
            "触发术语": {k: list(set(v)) for k, v in matched.items()}
        }

    def _extract_auto_terms_en(self, text: str) -> dict:
        words = text.lower().split()
        matched = defaultdict(list)
        for theme, keywords in AUTO_TERM_DICT.items():
            for kw in keywords:
                if kw.lower() in words:
                    matched[theme].append(kw)
        return {
            "语言": "en",
            "触发主题数量": len(matched),
            "触发术语": {k: list(set(v)) for k, v in matched.items()}
        }

    def _extract_emotions(self, text: str):
        result = self.emo_pipeline(text)[0]
        emotion_list = [
            {
                "label": EMOTION_MAP.get(item["label"], item["label"]),
                "score": round(item["score"], 4)
            }
            for item in result
        ]
        return emotion_list

    def _merge_results(self, results: list) -> dict:
        valid_results = [res for res in results if isinstance(res, dict) and res.get("极性分类")]

        if not valid_results:
            return {}

        full_text = "".join([res.get("文本", "") for res in valid_results])

        emotion_counter = defaultdict(float)
        for res in valid_results:
            for item in res.get("情感细分", []):
                emotion_counter[item["label"]] += item["score"]
        emotion_merged = [
            {"label": label, "score": round(score / len(valid_results), 4)}
            for label, score in emotion_counter.items()
        ]
        emotion_merged.sort(key=lambda x: x["score"], reverse=True)

        polarity_total = {"正面": 0.0, "中性": 0.0, "负面": 0.0}
        for res in valid_results:
            for k, v in res.get("极性分类", {}).get("极性细分", {}).items():
                polarity_total[k] += v
        avg_polarity = {k: round(v / len(valid_results), 4) for k, v in polarity_total.items()}
        main_polarity = max(avg_polarity, key=avg_polarity.get)
        main_score = avg_polarity[main_polarity]
        lang = next((res.get("极性分类", {}).get("语言", "zh") for res in valid_results), "zh")

        keywords_merged = list({kw["词语"]: kw for res in valid_results for kw in res.get("关键词", [])}.values())
        keywords_merged.sort(key=lambda x: x["得分"], reverse=True)

        auto_terms_total = defaultdict(set)
        for res in valid_results:
            for theme, kws in res.get("汽车术语识别", {}).get("触发术语", {}).items():
                auto_terms_total[theme].update(kws)
        auto_terms_merged = {k: list(v) for k, v in auto_terms_total.items()}

        conf_scores = [
            res.get("总置信度", {}).get("置信度评分")
            for res in valid_results
            if isinstance(res.get("总置信度", {}).get("置信度评分"), float)
        ]
        avg_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else "无可用数据"

        return {
            "文本": full_text,
            "情感细分": emotion_merged,
            "极性分类": {
                "语言": lang,
                "极性分类": main_polarity,
                "极性置信度": main_score,
                "极性细分": avg_polarity,
            },
            "关键词": keywords_merged,
            "汽车术语识别": {
                "语言": lang,
                "触发主题数量": len(auto_terms_merged),
                "触发术语": auto_terms_merged,
            },
            "总置信度": {"置信度评分": avg_conf},
        }
    
    #后续输入要改成list[str]提高速度
    def analyze_text(self, text: str) -> dict:
        if not isinstance(text, str):
            text = str(text) if not pd.isna(text) else ""
        text = text.strip()
        if not text:
            return {
                "文本": "",
                "语言": "unknown",
                "极性分类": {
                    "语言": "unknown",
                    "极性分类": "空",
                    "极性置信度": 0.0,
                    "极性细分": {}
                },
                "情感细分": [],
                "关键词": [],
                "汽车术语识别": {"语言": "unknown", "触发主题数量": 0, "触发术语": {}},
                "总置信度": {"置信度评分": 0.0}
            }

        lang = self._detect_language(text)
        if len(text) > 500:
            segments = self._split_text(text)
            results = [self.analyze_text(seg) for seg in segments]
            return self._merge_results(results)

        if lang.startswith("zh"):
            polarity = self._classify_polarity_zh(text)
            keywords = self.zh_kw.extract_keywords(text, top_n=2)
            auto_terms = self._extract_auto_terms_zh(text)
        elif lang == "en":
            polarity = self._classify_polarity_en(text)
            keywords = self.en_kw.extract_keywords(text, top_n=2)
            auto_terms = self._extract_auto_terms_en(text)
        else:
            return {
                "文本": text,
                "语言": lang,
                "极性分类": {
                    "语言": lang,
                    "极性分类": "中性",
                    "极性置信度": 0.0,
                    "极性细分": {}
                },
                "情感细分": [],
                "关键词": [],
                "汽车术语识别": {"语言": lang, "触发主题数量": 0, "触发术语": {}},
                "总置信度": {"置信度评分": 0.0}
            }

        emotion = self._extract_emotions(text)
        conf = self._compute_confidence(emotion, polarity["极性置信度"])

        return {
            "文本": text,
            "语言": lang,
            "极性分类": {
                "语言": lang,
                "极性分类": polarity["极性分类"],
                "极性置信度": polarity["极性置信度"],
                "极性细分": polarity["极性细分"]
            },
            "情感细分": emotion,
            "关键词": [{"词语": k, "得分": round(s, 4)} for k, s in keywords],
            "汽车术语识别": auto_terms,
            "总置信度": conf
        }


    def analyze_batch(self, df: pd.DataFrame, mode=1):
        results, polarities = [], []

        for idx, row in df.iterrows():
            try:
                if mode == 1:
                    text = row.get("content") or row.get("正文", "")
                elif mode == 2:
                    title = str(row.get("帖子标题", ""))
                    post = str(row.get("帖子内容", ""))
                    comments = str(row.get("评论内容", ""))

                    context = [title, post]
                    context_text = "。".join([t for t in context if t.strip()])
                    comment_weight = 0.8
                    context_weight = max(1, round((1 - comment_weight) * 5))
                    comment_weight_factor = max(1, round(comment_weight * 5))
                    comment_part = ("。" + comments.strip()) * comment_weight_factor
                    context_part = ("。" + context_text) * context_weight
                    text = f"原始帖子内容为：{context_part} 以下是用户对该帖子内容的评论：{comment_part}"
                else:
                    raise ValueError("Unsupported mode")

                res = self.analyze_text(text)
                if not isinstance(res, dict):
                    raise ValueError("analyze_text 返回非字典类型")

                res.pop("文本", None)
                results.append(res)

                p = res.get("极性分类", {})
                label = p.get("极性分类", "未知")
                score = p.get("极性置信度", 0.0)
                polarities.append(f"{label}（{score}）")

            except Exception as e:
                print(f"第 {idx} 行出错: {e}")
                results.append({})
                polarities.append("错误")

        df = df.iloc[:len(results)].copy()
        df["极性分类"] = polarities
        df["分析详情"] = [str(r) for r in results]
        return df, polarities

    def save_results(self, df):
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录（当前目录的父目录）
        project_root = os.path.dirname(current_dir)
        # 构建 data 文件夹路径（在项目根目录下）
        output_dir = os.path.join(project_root, "data")
        # 确保 data 文件夹存在
        os.makedirs(output_dir, exist_ok=True)
        # 创建带时间戳的文件名
        filename = f"output_sentiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # 构建完整文件路径
        path = os.path.join(output_dir, filename)
        # 保存文件
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"结果保存到：{path}")

    def print_distribution(self, polarities):
        label_counter = Counter([p.split("（")[0] if "（" in p else p for p in polarities])
        for label in ["正面", "中性", "负面", "空", "错误", "未知"]:
            print(f"{label}: {label_counter.get(label, 0)}")

    def run_from_ui(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            print("未选择文件，程序退出。")
            return

        mode = simpledialog.askinteger("模式选择", "1 = 正文\n2 = 帖子 + 评论")
        df = pd.read_csv(file_path)

        start = time.time()
        df, polarities = self.analyze_batch(df, mode=mode)
        self.save_results(df)
        self.print_distribution(polarities)
        print(f"\n总耗时：{time.time() - start:.2f} 秒")


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    #result = analyzer.analyze_text("小米汽车发布后科技圈和汽车圈都炸了。这款新能源汽车采用了先进的智能驾驶技术，配备了高性能的电池系统。雷军在发布会上详细介绍了这款车的创新功能和市场定位。雷军表示，小米汽车将采用“100%自研”模式，并采用“硬核”技术，以确保其产品在市场上具有竞争力。6月20日，广汽集团召开了一次小规模的媒体沟通会，董事长、总经理冯兴亚亲自出席并回应了广汽埃安员工持股等热点话题。冯兴亚首先详细介绍了广汽埃安混合所有制改革与员工持股计划的具体情况，明确指出广汽埃安与所谓的“车圈恒大”和“爆雷”无关。冯兴亚表示，2022年新能源行业高速发展，广汽埃安进行了混改，并推动了员工持股计划。根据协议，管理层和普通员工的股票都有5年的锁定期，直到2027年才届满。在锁定期内，如果员工离职，就必须退股，并按照上一年的埃安净资产值计算退还本金。这一设置旨在留住优秀员工，共同推动企业发展。")
    #from pprint import pprint
    #pprint(result)
    
    analyzer.run_from_ui()