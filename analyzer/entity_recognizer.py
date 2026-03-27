import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import jieba
import jieba.posseg as pseg
from transformers import pipeline
import torch
import logging

class EntityRecognizer:
    """实体识别和分类器 - 增强版"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.entity_counter = Counter()  # 实体频次统计
        self.context_patterns = {} # 上下文模式
        self._load_models()
        self._init_automotive_dict()
        self._build_rule_patterns()
    
    def _load_models(self):
        """加载NER模型"""
        try:
            model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
            self.ner_pipeline = pipeline(
                "ner", 
                model=model_name, 
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"NER模型加载失败: {e}")
            self.ner_pipeline = None
    
    def _init_automotive_dict(self):
        """初始化汽车领域词典"""
        # 扩展品牌词典
        self.automotive_brands = [
            "理想", "小鹏", "蔚来", "比亚迪","乘龙","小米",
            "特斯拉", "极氪", 
        ]
        
        # 车型系列
        self.automotive_models = [
            "Model S", "Model 3", "Model X", "Model Y", "Cybertruck",
            "理想ONE", "理想L9", "理想L8", "理想L7", "理想L6",
            "小鹏P7", "小鹏P5", "小鹏G9", "小鹏G6", "小鹏G3",
            "蔚来ES8", "蔚来ES6", "蔚来EC6", "蔚来ET7", "蔚来ET5",
            "汉EV", "唐EV", "宋EV", "秦EV", "海豚", "海豹", "元PLUS"
        ]
        
        # 人群标签
        self.user_groups = [
            "车主", "潜在买家", "粉丝", "黑子", "键盘侠", "专业测评", "媒体",
            "理想车主", "特斯拉车主", "蔚来车主", "小鹏车主", "比亚迪车主",
            "奔驰车主", "宝马车主", "奥迪车主", "大众车主", "丰田车主",
            "新能源车主", "燃油车主", "豪车车主", "平民车主"
        ]
        
        # 情感词汇
        self.sentiment_words = {
            "正面": ["优秀", "强大", "领先", "创新", "智能", "高端", "豪华", "性价比", "值得", "推荐"],
            "负面": ["垃圾", "差劲", "落后", "智障", "低端", "坑爹", "后悔", "不推荐", "避坑"],
            "攻击性": ["黑子", "水军", "托", "洗地", "吹嘘", "炒作", "割韭菜", "智商税"]
        }
        
        self.alias2canon = {
            "BYD": "比亚迪",
            "Tesla": "特斯拉",
            "TESLA": "特斯拉",
            "Model3": "Model 3",
            "Model-3": "Model 3",
            "理想one": "理想ONE",  # 常见大小写/变体
        }
        
        # 安全地将词汇添加到jieba词典
        try:
            for brand in self.automotive_brands:
                jieba.add_word(brand, freq=1000, tag='BRAND')
            for model in self.automotive_models:
                jieba.add_word(model, freq=800, tag='MODEL')
            for group in self.user_groups:
                jieba.add_word(group, freq=600, tag='GROUP')
        except Exception as e:
            print(f"添加jieba词典时出错: {e}")
    
    def extract_entities(self, text: str, lang: str = 'zh') -> dict:
        """提取并分类实体 - 增强版"""
        # 多种方法组合提取
        entities = []
        
        # 1. 基于规则的实体识别
        rule_entities = self._extract_by_rules(text)
        entities.extend(rule_entities)
        
        # 2. 基于jieba分词的实体识别  
        jieba_entities = self._extract_by_jieba(text)
        entities.extend(jieba_entities)
        
        # 3. 基于transformers的NER（如果可用）
        if self.ner_pipeline:
            ner_entities = self._extract_by_ner(text)
            entities.extend(ner_entities)
        
        # 4. 去重和合并
        merged_entities = self._merge_entities(entities)
        
        # 5. 统计频次和排名
        ranked_entities = self._rank_entities(merged_entities, text)
        
        return {
            "实体列表": ranked_entities,
            "统计信息": {
                "总实体数": len(ranked_entities),
                "品牌数": len([e for e in ranked_entities if e["类别"] == "BRAND"]),
                "人群数": len([e for e in ranked_entities if e["类别"] == "GROUP"]),
                "高频实体": ranked_entities[:5] if ranked_entities else []
            }
        }
    
    def _build_rule_patterns(self):
        """
        预编译三类正则。
        - 英文/数字词用伪边界：(?<![A-Za-z0-9]) 与 (?![A-Za-z0-9])
        - 允许轻微变体（举例：Model 3 的空格/连字符）
        """
        def _latin_guard(word: str) -> str:
            # 若含有拉丁字母或数字，为其加“伪边界”
            if re.search(r"[A-Za-z0-9]", word):
                return rf"(?<![A-Za-z0-9]){word}(?![A-Za-z0-9])"
            return word

        def _normalize_variant(word: str) -> str:
            # 示例：让 "Model 3" 匹配 "Model-3"/"Model3"
            word = re.escape(word)
            word = word.replace(r"\ ", r"\s*-?\s*")  # 空格 -> 可选的空格/连字符
            return word

        self.brand_patterns = []
        for b in self.automotive_brands:
            pat = _latin_guard(re.escape(b))
            self.brand_patterns.append(re.compile(pat, flags=re.IGNORECASE))

        self.model_patterns = []
        for m in self.automotive_models:
            pat = _latin_guard(_normalize_variant(m))
            self.model_patterns.append(re.compile(pat, flags=re.IGNORECASE))

        self.group_patterns = []
        for g in self.user_groups:
            pat = _latin_guard(re.escape(g))
            self.group_patterns.append(re.compile(pat, flags=re.IGNORECASE))

    def _canonicalize(self, text_snippet: str) -> str:
        """别名归一：把命中的片段映射成标准名称"""
        key = text_snippet.strip()
        return self.alias2canon.get(key, key)

    def _extract_by_rules(self, text: str) -> List[dict]:
        """改进版：正则 + finditer，抓全量命中，并进行别名归一"""
        entities = []

        # 品牌
        for pat in self.brand_patterns:
            for m in pat.finditer(text):
                span_txt = text[m.start():m.end()]
                entities.append({
                    "文本": self._canonicalize(span_txt),
                    "类别": "BRAND",
                    "位置": [m.start(), m.end()],
                    "置信度": 0.95,
                    "来源": "规则匹配"
                })

        # 车型
        for pat in self.model_patterns:
            for m in pat.finditer(text):
                span_txt = text[m.start():m.end()]
                entities.append({
                    "文本": self._canonicalize(span_txt),
                    "类别": "PRODUCT_SERIES",
                    "位置": [m.start(), m.end()],
                    "置信度": 0.90,
                    "来源": "规则匹配"
                })

        # 人群
        for pat in self.group_patterns:
            for m in pat.finditer(text):
                span_txt = text[m.start():m.end()]
                entities.append({
                    "文本": self._canonicalize(span_txt),
                    "类别": "GROUP",
                    "位置": [m.start(), m.end()],
                    "置信度": 0.85,
                    "来源": "规则匹配"
                })

        return entities
    def _extract_by_jieba(self, text: str) -> List[dict]:
        """基于jieba分词的实体识别"""
        entities = []
        
        # 使用posseg进行词性标注
        words = pseg.cut(text)
        
        for word, flag in words:
            if len(word) < 2:  # 过滤单字
                continue
                
            entity_type = None
            confidence = 0.8
            
            # 根据词性和自定义标签分类
            if flag == 'BRAND' or word in self.automotive_brands:
                entity_type = "BRAND"
                confidence = 0.9
            elif flag == 'MODEL' or word in self.automotive_models:
                entity_type = "PRODUCT_SERIES"
                confidence = 0.85
            elif flag == 'GROUP' or word in self.user_groups:
                entity_type = "GROUP"
                confidence = 0.8
            elif flag in ['nr', 'nrf']:  # 人名
                entity_type = "PERSON"
                confidence = 0.7
            elif flag in ['nt', 'nz']:   # 机构名
                entity_type = "ORGANIZATION"
                confidence = 0.75
            
            if entity_type:
                start_idx = text.find(word)
                entities.append({
                    "文本": word,
                    "类别": entity_type,
                    "位置": [start_idx, start_idx + len(word)] if start_idx >= 0 else [0, len(word)],
                    "置信度": confidence,
                    "来源": "jieba分词"
                })
        
        return entities
    
    def _extract_by_ner(self, text: str) -> List[dict]:
        """基于transformers NER的实体识别"""
        entities = []
        
        try:
            ner_results = self.ner_pipeline(text)
            
            for entity in ner_results:
                entity_text = entity['word'].replace('##', '')  # 清理subword标记
                entity_label = entity['entity_group']
                start_pos = entity['start']
                end_pos = entity['end']
                confidence = entity['score']
                
                # 汽车领域实体重分类
                refined_label = self._classify_automotive_entity(entity_text, entity_label)
                
                entities.append({
                    "文本": entity_text,
                    "类别": refined_label,
                    "位置": [start_pos, end_pos],
                    "置信度": round(confidence, 4),
                    "来源": "NER模型"
                })
        
        except Exception as e:
            print(f"NER提取失败: {e}")
        
        return entities
    
    def _classify_automotive_entity(self, entity_text: str, base_label: str) -> str:
        """汽车领域实体重分类"""
        entity_lower = entity_text.lower()
        
        # 品牌识别
        if any(brand.lower() in entity_lower or entity_lower in brand.lower() 
               for brand in self.automotive_brands):
            return "BRAND"
        
        # 车型识别
        if any(model.lower() in entity_lower or entity_lower in model.lower() 
               for model in self.automotive_models):
            return "PRODUCT_SERIES"
        
        # 人群识别
        if any(group.lower() in entity_lower or entity_lower in group.lower()
               for group in self.user_groups):
            return "GROUP"
        
        # 保持原有标签但转换为统一格式
        label_mapping = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION', 
            'LOC': 'LOCATION',
            'MISC': 'MISCELLANEOUS'
        }
        
        return label_mapping.get(base_label, base_label)
    
    def _merge_entities(self, entities: List[dict]) -> List[dict]:
        """合并重复实体"""
        # 按文本内容分组
        entity_groups = defaultdict(list)
        
        for entity in entities:
            key = (entity["文本"].lower(), entity["类别"])
            entity_groups[key].append(entity)
        
        merged = []
        for (text, category), group in entity_groups.items():
            # 选择置信度最高的
            best_entity = max(group, key=lambda x: x["置信度"])
            
            # 合并来源信息
            sources = list(set([e["来源"] for e in group]))
            best_entity["来源"] = "+".join(sources)
            
            # 记录出现次数
            best_entity["出现次数"] = len(group)
            
            merged.append(best_entity)
        
        return merged
    
    def _rank_entities(self, entities: List[dict], text: str) -> List[dict]:
        """对实体进行排名（品牌出现次数优先版）"""
        
        for entity in entities:
            score = 0
            category = entity.get("类别", "")

            if category == "BRAND":
                # 品牌：出现次数优先
                occurrence_score = min(entity.get("出现次数", 1) / 10, 1.0)
                score += occurrence_score * 0.6  # 提高出现次数权重

                # 置信度和类别权重减小
                score += entity.get("置信度", 0) * 0.1
                category_weights = {"BRAND": 1.5}
                score += category_weights.get(category, 1.5) * 0.1

                # 位置加分
                pos = text.find(entity.get("文本", ""))
                if pos >= 0:
                    score += (1 - pos / max(len(text), 1)) * 0.1

                # 长度权重不再计入品牌
            else:
                # 非品牌实体保持原逻辑
                score += entity.get("置信度", 0) * 0.25

                occurrence_score = min(entity.get("出现次数", 1) / 10, 1.0)
                score += occurrence_score * 0.25

                length_score = min(len(entity.get("文本", "")) / 10, 1.0)
                score += length_score * 0.15

                category_weights = {
                    "PRODUCT_SERIES": 1.0,
                    "GROUP": 0.8,
                    "PERSON": 0.6,
                    "ORGANIZATION": 0.5
                }
                score += category_weights.get(category, 0.5) * 0.25

            entity["综合得分"] = round(score, 4)

        # 按综合得分排序
        ranked_entities = sorted(entities, key=lambda x: x["综合得分"], reverse=True)

        # 添加排名信息
        for i, entity in enumerate(ranked_entities, 1):
            entity["排名"] = i

        return ranked_entities


    
    def get_entity_statistics(self, entities: List[dict]) -> dict:
        """获取实体统计信息"""
        if not entities:
            return {
                "错误": "没有识别到任何实体",
                "类别分布": {},
                "高频实体TOP10": [],
                "品牌排行TOP5": [],
                "人群排行TOP5": []
            }
        
        # 按类别统计
        category_stats = Counter([e["类别"] for e in entities])
        
        # 高频实体（前10）
        high_freq_entities = entities[:10]
        
        # 品牌排行
        brand_ranking = [e for e in entities if e["类别"] == "BRAND"][:5]
        
        # 人群排行  
        group_ranking = [e for e in entities if e["类别"] == "GROUP"][:5]
        
        return {
            "类别分布": dict(category_stats),
            "高频实体TOP10": [{"实体": e["文本"], "得分": e["综合得分"], "排名": e["排名"]} 
                           for e in high_freq_entities],
            "品牌排行TOP5": [{"品牌": e["文本"], "得分": e["综合得分"]} 
                          for e in brand_ranking],
            "人群排行TOP5": [{"人群": e["文本"], "得分": e["综合得分"]} 
                          for e in group_ranking]
        }


    def get_brands_rank(self, entities: List[dict], set_brands_number: int) -> List[dict]:
        """获取检测到的品牌实体数量最多的品牌列表（增加阈值检测机制）"""
        if not entities:
            logging.warning("没有提供任何实体数据")
            return []
        
        # 过滤出品牌实体
        brand_entities = [e for e in entities if e.get("类别") == "BRAND"]
        
        if not brand_entities:
            logging.warning("未检测到任何品牌实体")
            return []
        
        # 按品牌分组计数
        brand_counts = {}
        for entity in brand_entities:
            brand_name = entity.get("文本", "")
            if brand_name:
                brand_counts[brand_name] = brand_counts.get(brand_name, 0) + 1
        
        # 筛选数量大于阈值的品牌
        brands_over_number = [
            {"品牌": brand, "数量": count}
            for brand, count in brand_counts.items()
            if count > set_brands_number
        ]
        
        # 按数量降序排序
        brands_over_number.sort(key=lambda x: x["数量"], reverse=True)
        
        # 检测是否有品牌符合阈值
        if not brands_over_number:
            warning_msg = f"没有品牌实体数量超过设定的阈值({set_brands_number})"
            print(f"警告: {warning_msg}")
            logging.warning(warning_msg)
            
            # 可选：返回所有品牌（即使未达到阈值）
            # brands_over_number = [
            #     {"品牌": brand, "数量": count}
            #     for brand, count in brand_counts.items()
            # ]
            # brands_over_number.sort(key=lambda x: x["数量"], reverse=True)
        
        return brands_over_number
        
# 使用示例和简化测试
if __name__ == "__main__":
    # 简化版测试，不依赖jieba和transformers
    print("=== 简化版实体识别测试 ===")
    
    # 创建实体识别器
    recognizer = EntityRecognizer()
    
    # 测试文本
    test_text = """
    理想车主真的素质低吗？作为一个理想L9的车主，我觉得这种说法很不公平。
    特斯拉车主也经常被黑，但是特斯拉Model Y的销量还是很好。
    蔚来的服务确实不错，蔚来车主都很满意。比亚迪汉EV也是很好的选择。
    有些键盘侠总是喜欢攻击新能源车主，说什么智商税，其实他们根本不懂。
    奔驰宝马奥迪这些豪车车主也有很多争议。
    """
    
    print("测试文本:")
    print(test_text)
    print("\n" + "="*50)
    
    # 进行实体识别
    try:
        result = recognizer.extract_entities(test_text)
        
        print("=== 实体识别结果 ===")
        entities = result.get('实体列表', [])
        print(f"成功识别到 {len(entities)} 个实体")
        
        if entities:
            print(f"\n前10个高频实体:")
            for i, entity in enumerate(entities[:10], 1):
                print(f"{i:2d}. {entity['文本']:8s} ({entity['类别']:15s}) - 得分: {entity['综合得分']:.4f}")
            
            # 显示统计信息
            stats = recognizer.get_entity_statistics(entities)
            
            print(f"\n=== 统计信息 ===")
            print(f"类别分布: {stats['类别分布']}")
            
            if stats["品牌排行TOP5"]:
                print(f"\n品牌排行TOP5:")
                for i, brand in enumerate(stats["品牌排行TOP5"], 1):
                    print(f"  {i}. {brand['品牌']} (得分: {brand['得分']:.4f})")
            
            if stats["人群排行TOP5"]:
                print(f"\n人群排行TOP5:")
                for i, group in enumerate(stats["人群排行TOP5"], 1):
                    print(f"  {i}. {group['人群']} (得分: {group['得分']:.4f})")
        
        else:
            print("未识别到任何实体，可能需要检查:")
            print("1. 文本内容是否包含汽车相关实体")
            print("2. 词典是否正确加载")
            print("3. 识别算法是否正常工作")
            
    except Exception as e:
        print(f"实体识别过程出错: {e}")
        print("错误详情:", type(e).__name__)
        import traceback
        traceback.print_exc()
    
    print(f"\n测试完成！")