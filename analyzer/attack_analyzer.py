import re
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import math
from .entity_recognizer import EntityRecognizer
import json
from tqdm import tqdm
import hashlib
import pickle


class BrandCentricAnalyzer:
    """以指定品牌为中心的情感关系分析器，包含攻击关系分析功能"""
    
    def __init__(self, entity_recognizer=EntityRecognizer()):
        self.entity_recognizer = entity_recognizer
        self.sentiment_patterns = self._init_sentiment_patterns()
        self.relationship_patterns = self._init_relationship_patterns()
        self.attack_patterns = self._init_attack_patterns()               # 新增攻击模式
        self.automotive_brands = self._get_automotive_brands()
        self._text_base_cache = {}                                        # 缓存文本基础分析（品牌发现+攻击分析）
        self._brand_specific_cache = {}                                   # 缓存品牌特定分析
        self._df_cache = {}                                               # 缓存完整DataFrame结果
        
        self.max_cache_size = 2000       # 最大缓存数量

        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 新增：缓存统计
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'cleanups': 0,
            'last_cleanup_time': None
        }


    def _get_automotive_brands(self) -> List[str]:
        """获取汽车品牌列表（从实体识别器或手动定义）"""
        if self.entity_recognizer and hasattr(self.entity_recognizer, 'automotive_brands'):
            return self.entity_recognizer.automotive_brands
        
        # 备用品牌列表
        return [
            "理想", "小鹏", "蔚来", "比亚迪", "特斯拉", "乘龙", "长安", "奇瑞", 
            "哈弗", "红旗", "奔驰", "宝马", "奥迪", "大众", "丰田", "本田",
            "极氪", "岚图", "智己", "问界", "极狐", "深蓝", "哪吒", "零跑"
        ]
    
    def _init_sentiment_patterns(self) -> Dict[str, List[str]]:
        """初始化情感模式"""
        return {
            "强烈正面": ["完美", "杰出", "卓越", "顶级", "最好", "领先", "王者", "碾压", "吊打", "完胜"],
            "正面": ["不错", "很好", "优秀", "强大", "先进", "智能", "值得", "推荐", "满意", "支持"],
            "负面": ["差", "不好", "落后", "失望", "后悔", "不推荐", "避坑", "问题", "缺点"],
            "强烈负面": ["垃圾", "废物", "智障", "恶心", "坑爹", "骗子", "割韭菜", "智商税", "辣鸡"]
        }
    
    def _init_relationship_patterns(self) -> Dict[str, List[str]]:
        """初始化关系模式"""
        return {
            "正面对比": [r"(.+?)比(.+?)好", r"(.+?)胜过(.+?)", r"(.+?)超越(.+?)", r"选择(.+?)而不是(.+?)"],
            "负面对比": [r"(.+?)不如(.+?)", r"(.+?)比(.+?)差", r"(.+?)输给(.+?)", r"(.+?)被(.+?)完爆"],
            "直接攻击": [r"(.+?)就是垃圾", r"别买(.+?)", r"(.+?)坑爹", r"(.+?)智商税"],
            "间接攻击": [r"(.+?)车主素质", r"(.+?)粉丝", r"开(.+?)的都是"],
            "中性提及": [r"(.+?)和(.+?)", r"(.+?)、(.+?)", r"(.+?)还是(.+?)"],
            "用户群体": [r"(.+?)车主", r"(.+?)用户", r"(.+?)粉丝"]
        }
    
    def _init_attack_patterns(self) -> Dict[str, List[str]]:
        """初始化攻击模式（更细分的攻击关系）"""
        return {
            "直接贬低": [
                r"(.+?)(质量差|有问题|缺陷|故障|垃圾|烂|差劲|不行|坑|翻车)",
                r"(.+?)(摸黑|黑|踩|diss|喷)",
                r"针对(.+?)的摸黑",
            ],
            "对比攻击": [
                r"(.+?)不如(.+?)好",
                r"比(.+?)强多了",
                r"(.+?)完爆(.+?)",
                r"提到(.+?).*觉得(.+?)不值",
                r"(.+?).*代差.*(.+?)",
                r"只要提到(.+?).*(.+?)就",
            ],
            "价值质疑": [
                r"(.+?)(不值|割韭菜|智商税|坑钱)",
                r"(.+?)(价格虚高|太贵|性价比低)"
            ],
            "预测攻击": [
                r"(.+?)(会被.*反噬|要凉|要完|会倒闭)",
                r"(.+?).*逆风.*情况"
            ]
        }

# 核心分析方法
# 情感分析
    def _analyze_brand_sentiment_with_csv(self, row: pd.Series, brand: str, text: str) -> dict:
        """改进的品牌情感分析：使用平衡版本"""
        return self._analyze_brand_sentiment_balanced(row, brand, text)
    
    def _analyze_brand_sentiment_balanced(self, row: pd.Series, brand: str, text: str) -> dict:
        """平衡的品牌情感分析：对负面情感更加谨慎"""
        
        # 1. 提取分析详情
        detailed_analysis = self._extract_detailed_analysis(row)
        
        # 2. 对高置信度结果进行平衡处理
        if self._should_use_direct_polarity(detailed_analysis):
            return self._create_balanced_polarity_result(detailed_analysis, brand, text)
        
        # 3. 其余逻辑保持不变，但使用保守的标签转换
        mentioned_brands = self._discover_brands_in_text(text)
        
        if len(mentioned_brands) <= 1:
            attribution_result = self._simplified_single_brand_attribution(text, brand, detailed_analysis)
        else:
            attribution_result = self._brand_attribution_analysis(text, brand, detailed_analysis)
        
        # 4. 情感强度计算（对负面更保守）
        final_strength = self._calculate_conservative_sentiment_strength(
            detailed_analysis, attribution_result, text, brand
        )
        
        # 5. 保守的标签判断
        sentiment_label = self._strength_to_label_conservative(
            final_strength, detailed_analysis['confidence']
        )
        
        return {
            'sentiment_label': sentiment_label,
            'strength': round(final_strength, 3),
            'confidence': round(detailed_analysis['confidence'], 3),
            'evidence': self._extract_brand_evidence(text, brand),
            'method_used': 'balanced_analysis'
        }
    
    def _create_balanced_polarity_result(self, detailed_analysis: dict, brand: str, text: str) -> dict:
        """平衡的极性分类结果创建"""
        global_sentiment = detailed_analysis['global_sentiment']
        confidence = detailed_analysis['confidence']
        
        # 对负面情感进行二次验证
        if global_sentiment == '负面':
            negative_verification = self._verify_negative_sentiment(text, brand, detailed_analysis)
            if not negative_verification['is_valid']:
                # 如果负面情感验证失败，降低强度或转为中性
                if confidence < 0.6:
                    global_sentiment = '中立'
                    strength = 0.0
                else:
                    # 保持负面但降低强度
                    strength = -confidence * negative_verification['adjustment_factor']
            else:
                # 验证通过，但使用更保守的强度计算
                strength = -confidence * self._get_negative_amplifier(confidence, negative_verification)
        elif global_sentiment == '正面':
            # 正面情感保持相对激进（因为假阳性风险较低）
            strength = confidence * 1.3
        else:
            strength = 0.0
        
        # 确保不超出范围
        strength = max(-2.0, min(2.0, strength))
        
        return {
            'sentiment_label': global_sentiment,
            'strength': round(strength, 3),
            'confidence': round(confidence, 3),
            'evidence': self._extract_brand_evidence(text, brand),
            'method_used': 'balanced_polarity'
        }
    
# 关系分析
    def _analyze_brand_pair_relationship(self, row: pd.Series, brand1: str, 
                                    brand2: str, text: str) -> dict:
        """分析两个品牌之间的关系"""
        # 1. 检测关系模式
        relationship_type, relationship_strength = self._detect_relationship_pattern(text, brand1, brand2)
        
        # 2. 获取CSV情感信息
        csv_sentiment = self._extract_csv_sentiment_info(row)
        
        # 3. 位置和上下文分析 → 保存结果并用于判断
        position_analysis = self._analyze_brand_positions(text, brand1, brand2)
        
        # 4. 综合判断关系（新增位置分析因素）
        if relationship_type == 'unknown':
            # 三重判断：情感 + 位置 + 默认规则
            if position_analysis.get('dominant') == brand1:
                relationship = f'{brand1} > {brand2}'
            elif position_analysis.get('dominant') == brand2:
                relationship = f'{brand1} < {brand2}'
            elif csv_sentiment['polarity'] == '正面':
                relationship = f'{brand1} > {brand2}'
            elif csv_sentiment['polarity'] == '负面':
                relationship = f'{brand1} < {brand2}'
            else:
                relationship = f'{brand1} ~ {brand2}'
            
            # 强度计算考虑位置权重
            position_weight = position_analysis.get('weight', 1.0)
            if '>' in relationship:
                final_strength = csv_sentiment['polarity_score'] * position_weight
            elif '<' in relationship:
                final_strength = -csv_sentiment['polarity_score'] * position_weight
            else:
                final_strength = 0.0
        else:
            relationship = relationship_type
            final_strength = relationship_strength
        
        # 关系类型分类（保持不变）
        if '>' in relationship:
            relation_category = f'{brand1}优于{brand2}'
        elif '<' in relationship:
            relation_category = f'{brand1}不如{brand2}'
        elif 'attack' in relationship_type:
            relation_category = '攻击关系'
        elif 'compare' in relationship_type:
            relation_category = '对比关系'
        else:
            relation_category = '中性提及'
        
        # 提取关系证据
        evidence = self._extract_relationship_evidence(text, brand1, brand2)
        
        # 返回结果包含位置分析 → 确保所有分析结果都被保存
        return {
            'relationship': relationship,
            'strength': round(final_strength, 3),
            'type': relation_category,
            'evidence': evidence,
            'position_analysis': position_analysis  # 新增关键字段
        }

    def _detect_relationship_pattern(self, text: str, brand1: str, brand2: str) -> Tuple[str, float]:
        """检测品牌关系模式"""
        
        # 首先检查攻击关系
        attack_result = self._detect_attack_relationship(text, brand1, brand2)
        if attack_result != ('unknown', 0.0):
            return attack_result
        
        # 然后检查其他关系模式
        for pattern_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        first, second = match[0].strip(), match[1].strip()
                        
                        if pattern_type == '正面对比':
                            if first == brand1 and second == brand2:
                                return f'{brand1} > {brand2}', 1.2
                            elif first == brand2 and second == brand1:
                                return f'{brand1} < {brand2}', -1.2
                        
                        elif pattern_type == '负面对比':
                            if first == brand1 and second == brand2:
                                return f'{brand1} < {brand2}', -1.0
                            elif first == brand2 and second == brand1:
                                return f'{brand1} > {brand2}', 1.0
                        
                        elif pattern_type == '直接攻击':
                            if brand1 in str(match):
                                return f'attack_{brand1}', -1.5
                            elif brand2 in str(match):
                                return f'attack_{brand2}', -1.5
        
        return 'unknown', 0.0

    def _detect_attack_relationship(self, text: str, brand1: str, brand2: str) -> Tuple[str, float]:
        """检测攻击关系（更细分的攻击类型）"""
        attack_results = []
        
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                
                for match in matches:
                    if isinstance(match, tuple):
                        for brand_candidate in match:
                            brand_candidate = brand_candidate.strip()
                            if brand_candidate == brand1:
                                attack_results.append((f'attack_{brand1}_{attack_type}', -1.8))
                            elif brand_candidate == brand2:
                                attack_results.append((f'attack_{brand2}_{attack_type}', -1.8))
                    else:
                        brand_candidate = match.strip()
                        if brand_candidate == brand1:
                            attack_results.append((f'attack_{brand1}_{attack_type}', -1.8))
                        elif brand_candidate == brand2:
                            attack_results.append((f'attack_{brand2}_{attack_type}', -1.8))
        
        # 如果有多个攻击结果，返回强度最大的那个
        if attack_results:
            return max(attack_results, key=lambda x: abs(x[1]))
        
        return ('unknown', 0.0)

# 归因分析​
    def _brand_attribution_analysis(self, text: str, brand: str, detailed_analysis: dict) -> dict:
        """品牌归因分析：判断全局情感是否指向目标品牌"""
        
        attribution_score = 0.0
        attribution_confidence = 0.5
        attribution_evidence = []
        
        # 1. 位置权重分析
        position_weight = self._calculate_brand_position_weight(text, brand)
        
        # 2. 语法关系分析
        grammar_weight = self._analyze_brand_grammar_relation(text, brand, detailed_analysis)
        
        # 3. 关键词归属分析
        keyword_weight = self._analyze_keyword_attribution(text, brand, detailed_analysis['keywords'])
        
        # 4. 多品牌对比分析
        comparison_weight = self._analyze_multi_brand_comparison(text, brand)
        
        # 5. 情感词距离分析
        sentiment_distance_weight = self._analyze_sentiment_word_distance(text, brand)
        
        # 综合计算归因得分
        attribution_score = (
            position_weight * 0.3 +
            grammar_weight * 0.25 +
            keyword_weight * 0.2 +
            comparison_weight * 0.15 +
            sentiment_distance_weight * 0.1
        )
        
        # 计算归因置信度
        attribution_confidence = min(1.0, abs(attribution_score) + 0.3)
        
        return {
            'attribution_score': attribution_score,
            'attribution_confidence': attribution_confidence,
            'evidence': attribution_evidence,
            'components': {
                'position': position_weight,
                'grammar': grammar_weight,
                'keywords': keyword_weight,
                'comparison': comparison_weight,
                'sentiment_distance': sentiment_distance_weight
            }
        }

    def _simplified_single_brand_attribution(self, text: str, brand: str, detailed_analysis: dict) -> dict:
        """简化的单品牌归因分析"""
        
        # 单品牌场景下，简化归因逻辑
        attribution_score = 0.8  # 默认高归因
        attribution_confidence = 0.8
        
        # 只做基础的位置和距离检查
        brand_pos = text.find(brand)
        if brand_pos == -1:
            attribution_score = 0.3
            attribution_confidence = 0.3
        else:
            # 检查否定模式
            negation_patterns = [rf'不是{brand}', rf'没有{brand}', rf'除了{brand}', rf'不要{brand}']
            for pattern in negation_patterns:
                if re.search(pattern, text):
                    attribution_score = 0.2
                    break
            
            # 检查强调模式
            emphasis_patterns = [rf'就是{brand}', rf'只有{brand}', rf'特别是{brand}', rf'尤其是{brand}']
            for pattern in emphasis_patterns:
                if re.search(pattern, text):
                    attribution_score = 0.95
                    break
        
        return {
            'attribution_score': attribution_score,
            'attribution_confidence': attribution_confidence,
            'method': 'simplified_single_brand'
        }

#上下文分析
    def _analyze_brand_context(self, text: str, brand: str) -> dict:
        """分析品牌上下文情感"""
        brand_pos = text.find(brand)
        if brand_pos == -1:
            return {'context_sentiment': 0.0}
        
        # 获取品牌前后的上下文（前后各20个字符）
        start = max(0, brand_pos - 20)
        end = min(len(text), brand_pos + len(brand) + 20)
        context = text[start:end]
        
        sentiment_score = 0.0
        
        # 在上下文中寻找情感词汇
        for sentiment_type, words in self.sentiment_patterns.items():
            for word in words:
                if word in context:
                    if sentiment_type == '强烈正面':
                        sentiment_score += 2.0
                    elif sentiment_type == '正面':
                        sentiment_score += 1.0
                    elif sentiment_type == '负面':
                        sentiment_score -= 1.0
                    elif sentiment_type == '强烈负面':
                        sentiment_score -= 2.0
        
        return {'context_sentiment': max(-2.0, min(2.0, sentiment_score))}

    def _get_context_adjustment(self, text: str, brand: str) -> float:
        """获取上下文微调得分"""
        
        # 检测否定词
        negation_patterns = [rf'不是{brand}', rf'没有{brand}', rf'除了{brand}']
        negation_adjustment = 0.0
        
        for pattern in negation_patterns:
            if re.search(pattern, text):
                negation_adjustment -= 0.3
        
        # 检测强调词
        emphasis_patterns = [rf'就是{brand}', rf'只有{brand}', rf'特别是{brand}']
        emphasis_adjustment = 0.0
        
        for pattern in emphasis_patterns:
            if re.search(pattern, text):
                emphasis_adjustment += 0.2
        
        return negation_adjustment + emphasis_adjustment

# 辅助分析方法
    def _discover_related_brands(self, df: pd.DataFrame, target_brand: str) -> List[str]:
        """动态发现与目标品牌相关的其他品牌"""
        related_brands = set()
        
        # 获取包含目标品牌的文本
        content_col = self._get_content_column(df)
        target_texts = df[df[content_col].str.contains(target_brand, na=False)]
        
        for _, row in target_texts.iterrows():
            text = str(row[content_col])
            
            # 方法1：使用实体识别器
            if self.entity_recognizer:
                entities = self.entity_recognizer.extract_entities(text)
                brand_entities = [e for e in entities.get('实体列表', []) 
                                if e.get('类别') == 'BRAND']
                for entity in brand_entities:
                    related_brands.add(entity['文本'])
            
            # 方法2：基于预定义品牌列表匹配
            for brand in self.automotive_brands:
                if brand in text and brand != target_brand:
                    related_brands.add(brand)
            
            # 方法3：基于关系模式提取
            extracted_brands = self._extract_brands_from_patterns(text, target_brand)
            related_brands.update(extracted_brands)
        
        return list(related_brands)
    
    def _extract_brands_from_patterns(self, text: str, target_brand: str) -> List[str]:
        """基于关系模式提取品牌"""
        extracted_brands = []
        
        for pattern_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                
                for match in matches:
                    if isinstance(match, tuple):
                        for brand_candidate in match:
                            brand_candidate = brand_candidate.strip()
                            if (brand_candidate in self.automotive_brands and 
                                brand_candidate != target_brand and
                                len(brand_candidate) > 1):
                                extracted_brands.append(brand_candidate)
                    else:
                        brand_candidate = match.strip()
                        if (brand_candidate in self.automotive_brands and 
                            brand_candidate != target_brand and
                            len(brand_candidate) > 1):
                            extracted_brands.append(brand_candidate)
        
        return extracted_brands
    
    def _get_content_column(self, df: pd.DataFrame) -> str:
        """自动识别内容列"""
        possible_columns = ['内容', 'text', 'content', '文本', 'comment']
        for col in possible_columns:
            if col in df.columns:
                return col
        return df.columns[0]  # 默认使用第一列
    
    def _add_target_brand_analysis(self, df: pd.DataFrame, target_brand: str) -> pd.DataFrame:
        """为目标品牌添加基础分析"""
        content_col = self._get_content_column(df)
        
        # 目标品牌的基础情感分析
        df[f'{target_brand}_提及'] = df[content_col].str.contains(target_brand, na=False)
        df[f'{target_brand}_情感倾向'] = '无关联'
        df[f'{target_brand}_情感强度'] = 0.0
        df[f'{target_brand}_关键证据'] = ''
        
        for idx, row in df.iterrows():
            if row[f'{target_brand}_提及']:
                analysis = self._analyze_brand_sentiment_with_csv(
                    row, target_brand, str(row[content_col])
                )
                
                df.loc[idx, f'{target_brand}_情感倾向'] = analysis['sentiment_label']
                df.loc[idx, f'{target_brand}_情感强度'] = analysis['strength']
                df.loc[idx, f'{target_brand}_关键证据'] = analysis['evidence']
        
        return df
    
    def _add_brand_relationship_columns(self, df: pd.DataFrame, 
                                      target_brand: str, related_brand: str) -> pd.DataFrame:
        """为相关品牌添加关系分析列"""
        content_col = self._get_content_column(df)
        
        # 关系列命名
        relationship_col = f'{target_brand}_vs_{related_brand}_关系'
        strength_col = f'{target_brand}_vs_{related_brand}_强度'
        type_col = f'{target_brand}_vs_{related_brand}_类型'
        evidence_col = f'{target_brand}_vs_{related_brand}_证据'
        
        # 初始化列
        df[relationship_col] = '无关系'
        df[strength_col] = 0.0
        df[type_col] = '无关系'
        df[evidence_col] = ''
        
        for idx, row in df.iterrows():
            text = str(row[content_col])
            
            # 只分析同时包含两个品牌的文本
            if target_brand in text and related_brand in text:
                relationship_analysis = self._analyze_brand_pair_relationship(
                    row, target_brand, related_brand, text
                )
                
                df.loc[idx, relationship_col] = relationship_analysis['relationship']
                df.loc[idx, strength_col] = relationship_analysis['strength']
                df.loc[idx, type_col] = relationship_analysis['type']
                df.loc[idx, evidence_col] = relationship_analysis['evidence']
        
        return df
    
    def _should_use_direct_polarity(self, detailed_analysis: dict) -> bool:
        """判断是否应该直接使用极性分类结果"""
        confidence = detailed_analysis.get('confidence', 0.0)
        global_sentiment = detailed_analysis.get('global_sentiment', '中立')
        
        # 高置信度且非中性的结果直接采用
        return (confidence >= 0.5 and global_sentiment in ['正面', '负面'])

    def _create_direct_polarity_result(self, detailed_analysis: dict, brand: str, text: str) -> dict:
        """直接基于极性分类创建结果"""
        global_sentiment = detailed_analysis['global_sentiment']
        confidence = detailed_analysis['confidence']
        
        # 直接映射情感强度
        if global_sentiment == '正面':
            strength = confidence * 1.5  # 放大正面情感
        elif global_sentiment == '负面':
            strength = -confidence * 1.5  # 放大负面情感
        else:
            strength = 0.0
        
        # 确保不超出范围
        strength = max(-2.0, min(2.0, strength))
        
        return {
            'sentiment_label': global_sentiment,
            'strength': round(strength, 3),
            'confidence': round(confidence, 3),
            'evidence': self._extract_brand_evidence(text, brand),
            'method_used': 'direct_polarity'
        }


    def _calculate_improved_sentiment_strength(self, detailed_analysis: dict, 
                                            attribution_result: dict, text: str, brand: str) -> float:
        """改进的情感强度计算"""
        
        global_sentiment_score = detailed_analysis['global_score']
        attribution_score = attribution_result['attribution_score']
        confidence = detailed_analysis['confidence']
        
        # 新的权重分配（更加倾向于全局情感）
        base_strength = global_sentiment_score * attribution_score
        
        # 置信度加权：高置信度时放大情感强度
        confidence_amplifier = 1.0 + (confidence - 0.5) * 0.8  # [0.6, 1.4]
        
        # 应用置信度放大
        amplified_strength = base_strength * confidence_amplifier
        
        # 上下文微调（减少权重）
        context_adjustment = self._get_context_adjustment(text, brand) * 0.1
        
        # 最终强度计算
        final_strength = amplified_strength + context_adjustment
        
        # 非线性变换：减少中间值
        final_strength = self._apply_nonlinear_transformation(final_strength)
        
        # 标准化到[-2, 2]范围
        return max(-2.0, min(2.0, final_strength))

    def _apply_nonlinear_transformation(self, strength: float) -> float:
        """应用非线性变换减少中间值"""
        
        if abs(strength) < 0.1:
            return strength * 0.5  # 压缩接近0的值
        elif abs(strength) < 0.3:
            # 对[0.1, 0.3]区间进行拉伸
            sign = 1 if strength >= 0 else -1
            abs_strength = abs(strength)
            # 将[0.1, 0.3]映射到[0.2, 0.8]
            transformed = 0.2 + (abs_strength - 0.1) * 3.0
            return sign * min(transformed, 2.0)
        else:
            # 保持较强的情感不变
            return strength

    def _strength_to_label_improved(self, strength: float, confidence: float) -> str:
        """改进的强度到标签转换：动态阈值"""
        
        # 根据置信度动态调整阈值
        if confidence >= 0.7:
            threshold = 0.08  # 高置信度，很低阈值
        elif confidence >= 0.55:
            threshold = 0.12  # 中高置信度，低阈值
        elif confidence >= 0.45:
            threshold = 0.18  # 中等置信度，中等阈值
        else:
            threshold = 0.25  # 低置信度，高阈值
        
        if strength >= threshold:
            return "正面"
        elif strength <= -threshold:
            return "负面"
        else:
            return "中立"

    def _get_analysis_method(self, detailed_analysis: dict, brand_count: int) -> str:
        """记录使用的分析方法"""
        
        if detailed_analysis.get('confidence', 0) >= 0.5 and detailed_analysis.get('global_sentiment') in ['正面', '负面']:
            return "direct_polarity"
        elif brand_count <= 1:
            return "simplified_attribution"
        else:
            return "full_attribution"
    def _verify_negative_sentiment(self, text: str, brand: str, detailed_analysis: dict) -> dict:
        """验证负面情感的合理性"""
        
        verification_result = {
            'is_valid': True,
            'adjustment_factor': 1.2,  # 默认调整因子
            'reasons': []
        }
        
        # 1. 检查否定语境
        negation_context = self._check_negation_context(text, brand)
        if negation_context['has_negation']:
            verification_result['is_valid'] = False
            verification_result['reasons'].append('否定语境')
            return verification_result
        
        # 2. 检查是否有具体的负面指标
        negative_indicators = self._find_negative_indicators(text, brand, detailed_analysis)
        if negative_indicators['count'] == 0:
            verification_result['adjustment_factor'] = 0.6  # 降低强度
            verification_result['reasons'].append('缺乏具体负面指标')
        
        # 3. 检查对比语境
        comparison_context = self._check_comparison_context(text, brand)
        if comparison_context['is_comparison'] and not comparison_context['brand_is_worse']:
            verification_result['adjustment_factor'] = 0.7
            verification_result['reasons'].append('对比语境中非劣势方')
        
        # 4. 检查情感强度一致性
        emotion_consistency = self._check_emotion_consistency(detailed_analysis)
        if not emotion_consistency['is_consistent']:
            verification_result['adjustment_factor'] *= 0.8
            verification_result['reasons'].append('情感细分不一致')
        
        return verification_result

    def _check_negation_context(self, text: str, brand: str) -> dict:
        """检查否定语境"""
        
        negation_patterns = [
            rf'不是{brand}',
            rf'没有选{brand}',
            rf'除了{brand}',
            rf'不要{brand}',
            rf'别买{brand}',
            rf'不推荐{brand}',
            rf'{brand}不是',
            rf'并非{brand}'
        ]
        
        has_negation = any(re.search(pattern, text) for pattern in negation_patterns)
        
        return {
            'has_negation': has_negation,
            'patterns_found': [p for p in negation_patterns if re.search(p, text)]
        }

    def _find_negative_indicators(self, text: str, brand: str, detailed_analysis: dict) -> dict:
        """寻找具体的负面指标"""
        
        indicators = {
            'count': 0,
            'types': [],
            'evidence': []
        }
        
        # 1. 从关键词中寻找负面指标
        keywords = detailed_analysis.get('keywords', [])
        for keyword_item in keywords:
            if isinstance(keyword_item, dict):
                keyword = keyword_item.get('词语', '')
                score = keyword_item.get('得分', 0.0)
                
                # 明确的负面关键词
                negative_keywords = ['问题', '故障', '投诉', '差', '烂', '坑', '失望', '后悔']
                if any(neg_word in keyword for neg_word in negative_keywords):
                    indicators['count'] += 1
                    indicators['types'].append('负面关键词')
                    indicators['evidence'].append(keyword)
        
        # 2. 从文本中直接寻找负面表达
        direct_negative_patterns = [
            rf'{brand}.*?[有存在出现].*?问题',
            rf'{brand}.*?质量.*?[差不好]',
            rf'{brand}.*?[故障坏了]',
            rf'对{brand}.*?失望',
            rf'后悔.*?买.*?{brand}',
            rf'{brand}.*?[坑爹智商税]'
        ]
        
        for pattern in direct_negative_patterns:
            if re.search(pattern, text):
                indicators['count'] += 1
                indicators['types'].append('直接负面表达')
                indicators['evidence'].append(re.search(pattern, text).group())
        
        return indicators

    def _check_comparison_context(self, text: str, brand: str) -> dict:
        """检查对比语境"""
        
        comparison_result = {
            'is_comparison': False,
            'brand_is_worse': False,
            'comparison_evidence': ''
        }
        
        # 发现其他品牌
        other_brands = [b for b in self.automotive_brands if b != brand and b in text]
        
        if other_brands:
            comparison_result['is_comparison'] = True
            
            # 检查品牌在对比中的地位
            worse_patterns = [
                rf'{brand}.*?不如.*?({"|".join(other_brands)})',
                rf'({"|".join(other_brands)}).*?比.*?{brand}.*?好',
                rf'选择.*?({"|".join(other_brands)}).*?而不是.*?{brand}'
            ]
            
            for pattern in worse_patterns:
                match = re.search(pattern, text)
                if match:
                    comparison_result['brand_is_worse'] = True
                    comparison_result['comparison_evidence'] = match.group()
                    break
        
        return comparison_result

    def _check_emotion_consistency(self, detailed_analysis: dict) -> dict:
        """检查情感细分的一致性"""
        
        consistency_result = {
            'is_consistent': True,
            'inconsistency_score': 0.0
        }
        
        emotion_details = detailed_analysis.get('emotion_details', [])
        if emotion_details and len(emotion_details) > 0:
            top_emotion = emotion_details[0]
            emotion_label = top_emotion.get('label', '')
            emotion_score = top_emotion.get('score', 0.0)
            
            # 如果极性是负面，但情感细分的top情感是正面或中立
            if (detailed_analysis.get('global_sentiment') == '负面' and 
                emotion_label in ['中立', '赞同', '喜爱', '乐观'] and 
                emotion_score > 0.7):
                
                consistency_result['is_consistent'] = False
                consistency_result['inconsistency_score'] = emotion_score
        
        return consistency_result

    def _get_negative_amplifier(self, confidence: float, verification_result: dict) -> float:
        """根据验证结果获取负面情感放大系数"""
        
        base_amplifier = 1.0
        
        # 根据置信度调整
        if confidence >= 0.8:
            base_amplifier = 1.3
        elif confidence >= 0.6:
            base_amplifier = 1.1
        elif confidence >= 0.5:
            base_amplifier = 0.9
        else:
            base_amplifier = 0.7
        
        # 应用验证调整因子
        final_amplifier = base_amplifier * verification_result['adjustment_factor']
        
        # 确保不超过合理范围
        return max(0.5, min(1.5, final_amplifier))

    def _strength_to_label_conservative(self, strength: float, confidence: float) -> str:
        """更保守的强度到标签转换（特别针对负面）"""
        
        # 对负面情感使用更高的阈值
        if strength >= 0:
            # 正面情感阈值
            if confidence >= 0.7:
                pos_threshold = 0.10
            elif confidence >= 0.5:
                pos_threshold = 0.15
            else:
                pos_threshold = 0.22
            
            return "正面" if strength >= pos_threshold else "中立"
        else:
            # 负面情感阈值（更保守）
            if confidence >= 0.8:
                neg_threshold = -0.15  # 高置信度时稍微放松
            elif confidence >= 0.65:
                neg_threshold = -0.20  # 中高置信度
            elif confidence >= 0.5:
                neg_threshold = -0.30  # 中等置信度，较高阈值
            else:
                neg_threshold = -0.40  # 低置信度，很高阈值
            
            return "负面" if strength <= neg_threshold else "中立"

    def _calculate_conservative_sentiment_strength(self, detailed_analysis: dict, 
                                                attribution_result: dict, text: str, brand: str) -> float:
        """更保守的情感强度计算"""
        
        global_sentiment_score = detailed_analysis['global_score']
        attribution_score = attribution_result['attribution_score']
        confidence = detailed_analysis['confidence']
        
        # 对负面情感进行额外验证
        if global_sentiment_score < 0:
            verification = self._verify_negative_sentiment(text, brand, detailed_analysis)
            if not verification['is_valid']:
                global_sentiment_score *= 0.3  # 大幅降低无效负面情感
            else:
                global_sentiment_score *= verification['adjustment_factor']
        
        # 基础强度计算
        base_strength = global_sentiment_score * attribution_score
        
        # 置信度调整（对负面更保守）
        if base_strength < 0:
            # 负面情感需要更高置信度才能放大
            confidence_amplifier = 0.8 + (confidence - 0.5) * 0.4  # [0.6, 1.0]
        else:
            # 正面情感保持原有放大
            confidence_amplifier = 1.0 + (confidence - 0.5) * 0.6  # [0.7, 1.3]
        
        amplified_strength = base_strength * confidence_amplifier
        
        # 上下文调整
        context_adjustment = self._get_context_adjustment(text, brand) * 0.05
        
        final_strength = amplified_strength + context_adjustment
        
        # 对负面情感进行额外压缩
        if final_strength < 0:
            final_strength = final_strength * 0.9  # 负面情感整体压缩10%
        
        return max(-2.0, min(2.0, final_strength))
    

    def _should_use_direct_polarity(self, detailed_analysis: dict) -> bool:
        """判断是否应该直接使用极性分类结果"""
        confidence = detailed_analysis.get('confidence', 0.0)
        global_sentiment = detailed_analysis.get('global_sentiment', '中立')
        
        # 高置信度且非中性的结果直接采用
        return (confidence >= 0.5 and global_sentiment in ['正面', '负面'])

    def _simplified_single_brand_attribution(self, text: str, brand: str, detailed_analysis: dict) -> dict:
        """简化的单品牌归因分析"""
        
        # 单品牌场景下，简化归因逻辑
        attribution_score = 0.8  # 默认高归因
        attribution_confidence = 0.8
        
        # 只做基础的位置和距离检查
        brand_pos = text.find(brand)
        if brand_pos == -1:
            attribution_score = 0.3
            attribution_confidence = 0.3
        else:
            # 检查否定模式
            negation_patterns = [rf'不是{brand}', rf'没有{brand}', rf'除了{brand}', rf'不要{brand}']
            for pattern in negation_patterns:
                if re.search(pattern, text):
                    attribution_score = 0.2
                    break
            
            # 检查强调模式
            emphasis_patterns = [rf'就是{brand}', rf'只有{brand}', rf'特别是{brand}', rf'尤其是{brand}']
            for pattern in emphasis_patterns:
                if re.search(pattern, text):
                    attribution_score = 0.95
                    break
        
        return {
            'attribution_score': attribution_score,
            'attribution_confidence': attribution_confidence,
            'method': 'simplified_single_brand'
        }
    
    
    def _analyze_brand_positions(self, text: str, brand1: str, brand2: str) -> dict:
        """分析品牌在文本中的位置关系"""
        pos1 = text.find(brand1)
        pos2 = text.find(brand2)
        
        return {
            'brand1_first': pos1 < pos2 if pos1 >= 0 and pos2 >= 0 else None,
            'distance': abs(pos2 - pos1) if pos1 >= 0 and pos2 >= 0 else -1,
            'same_sentence': self._in_same_sentence(text, brand1, brand2)
        }
    
    def _in_same_sentence(self, text: str, brand1: str, brand2: str) -> bool:
        """判断两个品牌是否在同一句话中"""
        sentences = re.split(r'[。！？；]', text)
        
        for sentence in sentences:
            if brand1 in sentence and brand2 in sentence:
                return True
        return False
    
    def _extract_csv_sentiment_info(self, row: pd.Series) -> dict:
        """提取CSV中的情感信息"""
        sentiment_info = {
            'polarity': '中立',
            'polarity_score': 0.0,
            'emotion': '中立',
            'confidence': 0.5
        }
        
        # 提取极性分类
        if '极性分类' in row:
            polarity_data = row['极性分类']
            if isinstance(polarity_data, dict):
                sentiment_info['polarity'] = polarity_data.get('极性分类', '中立')
                score = polarity_data.get('极性置信度', 0.5)
                if sentiment_info['polarity'] == '正面':
                    sentiment_info['polarity_score'] = score
                elif sentiment_info['polarity'] == '负面':
                    sentiment_info['polarity_score'] = -score
            elif isinstance(polarity_data, str):
                sentiment_info['polarity'] = polarity_data
                sentiment_info['polarity_score'] = 1.0 if polarity_data == '正面' else -1.0 if polarity_data == '负面' else 0.0
        
        # 提取情感细分
        if '情感细分' in row:
            emotion_data = row['情感细分']
            if isinstance(emotion_data, list) and emotion_data:
                top_emotion = emotion_data[0]
                if isinstance(top_emotion, dict):
                    sentiment_info['emotion'] = top_emotion.get('label', '中立')
        
        # 提取总置信度
        if '总置信度' in row:
            confidence_data = row['总置信度']
            if isinstance(confidence_data, dict):
                sentiment_info['confidence'] = confidence_data.get('置信度评分', 0.5)
        
        return sentiment_info
    

    
    def _analyze_csv_keywords_impact(self, row: pd.Series, brand: str) -> dict:
        """分析CSV关键词对品牌的影响"""
        impact_score = 0.0
        relevant_keywords = []
        
        if '关键词' in row:
            keywords_data = row['关键词']
            if isinstance(keywords_data, list):
                for keyword_item in keywords_data:
                    if isinstance(keyword_item, dict):
                        keyword = keyword_item.get('词语', '')
                        score = keyword_item.get('得分', 0.0)
                        
                        # 判断关键词是否与品牌相关
                        if brand in keyword or self._is_brand_relevant_keyword(keyword):
                            relevant_keywords.append(keyword)
                            # 判断关键词情感倾向
                            keyword_sentiment = self._judge_keyword_sentiment(keyword)
                            impact_score += keyword_sentiment * score
        
        return {
            'impact_score': max(-2.0, min(2.0, impact_score)),
            'relevant_keywords': relevant_keywords
        }
    
    def _analyze_csv_auto_terms_impact(self, row: pd.Series, brand: str, text: str) -> dict:
        """分析汽车术语对品牌的影响"""
        terms_sentiment = 0.0
        triggered_terms = []
        
        if '汽车术语识别' in row:
            auto_terms_data = row['汽车术语识别']
            if isinstance(auto_terms_data, dict):
                terms_info = auto_terms_data.get('触发术语', {})
                
                for term_category, terms in terms_info.items():
                    triggered_terms.extend(terms)
                    
                    # 分析术语在品牌上下文中的情感
                    for term in terms:
                        if term in text:
                            term_context_sentiment = self._analyze_term_brand_context(
                                text, brand, term
                            )
                            terms_sentiment += term_context_sentiment
        
        return {
            'terms_sentiment': max(-1.0, min(1.0, terms_sentiment)),
            'triggered_terms': triggered_terms
        }
    
    def _is_brand_relevant_keyword(self, keyword: str) -> bool:
        """判断关键词是否与品牌相关"""
        brand_relevant_words = [
            "车主", "粉丝", "用户", "销量", "市场", "竞争", "对比", "选择",
            "配置", "性能", "价格", "服务", "质量", "体验", "口碑"
        ]
        return any(word in keyword for word in brand_relevant_words)
    
    def _judge_keyword_sentiment(self, keyword: str) -> float:
        """判断关键词的情感倾向"""
        positive_indicators = ["赠送", "优惠", "升级", "改进", "新增", "提升", "领先"]
        negative_indicators = ["取消", "减少", "问题", "故障", "投诉", "缺陷", "落后"]
        
        for pos_word in positive_indicators:
            if pos_word in keyword:
                return 1.0
        
        for neg_word in negative_indicators:
            if neg_word in keyword:
                return -1.0
        
        return 0.0
    
    def _analyze_term_brand_context(self, text: str, brand: str, term: str) -> float:
        """分析术语在品牌上下文中的情感"""
        brand_pos = text.find(brand)
        term_pos = text.find(term)
        
        if brand_pos == -1 or term_pos == -1:
            return 0.0
        
        # 获取品牌和术语之间的上下文
        start_pos = min(brand_pos, term_pos)
        end_pos = max(brand_pos + len(brand), term_pos + len(term))
        context = text[start_pos:end_pos]
        
        # 在上下文中寻找情感词汇
        sentiment = 0.0
        positive_words = ["好", "优秀", "不错", "满意", "推荐"]
        negative_words = ["差", "不好", "问题", "失望", "后悔"]
        
        for word in positive_words:
            if word in context:
                sentiment += 0.3
        
        for word in negative_words:
            if word in context:
                sentiment -= 0.3
        
        return max(-1.0, min(1.0, sentiment))
    
    def _strength_to_label(self, strength: float) -> str:
        """将强度数值转换为简化标签"""
        if strength >= 0.3:
            return "正面"
        elif strength <= -0.3:
            return "负面"
        else:
            return "中立"
    
    def _extract_brand_evidence(self, text: str, brand: str) -> str:
        """提取品牌相关证据"""
        sentences = re.split(r'[。！？；]', text)
        
        # 找到包含品牌或相关攻击词汇的句子
        relevant_sentences = []
        attack_keywords = ["摸黑", "反噬", "代差", "不值", "逆风", "完爆", "坑", "烂"]
        
        for sentence in sentences:
            s = sentence.strip()
            if len(s) > 5 and (brand in s or any(keyword in s for keyword in attack_keywords)):
                relevant_sentences.append(s)
        
        if relevant_sentences:
            # 返回最长的相关句子作为证据
            return max(relevant_sentences, key=len)
        
        return text[:50] + "..." if len(text) > 50 else text

    def _extract_relationship_evidence(self, text: str, brand1: str, brand2: str) -> str:
        """提取关系证据"""
        sentences = re.split(r'[。！？；]', text)
        
        # 找到同时包含两个品牌的句子
        for sentence in sentences:
            if brand1 in sentence and brand2 in sentence:
                return sentence.strip()
        
        return ""
    
    def _add_comprehensive_relationship_analysis(self, df: pd.DataFrame, 
                                               target_brand: str, related_brands: List[str]) -> pd.DataFrame:
        """添加综合关系分析"""
        
        # 添加关系统计列
        df[f'{target_brand}_关系统计'] = df.apply(
            lambda row: self._calculate_relationship_stats(row, target_brand, related_brands),
            axis=1
        )
        
        # 添加主要竞争对手列
        df[f'{target_brand}_主要对手'] = df.apply(
            lambda row: self._identify_main_competitor(row, target_brand, related_brands),
            axis=1
        )
        
        # 添加关系类型汇总
        df[f'{target_brand}_关系类型汇总'] = df.apply(
            lambda row: self._summarize_relationship_types(row, target_brand, related_brands),
            axis=1
        )
        
        return df
    
    def _add_attack_relationship_analysis(self, df: pd.DataFrame, 
                                          target_brand: str, related_brands: List[str]) -> pd.DataFrame:
        """添加攻击关系分析"""
        
        # 添加攻击关系统计列
        df[f'{target_brand}_攻击关系'] = df.apply(
            lambda row: self._analyze_attack_relationships(row, target_brand, related_brands),
            axis=1
        )
        
        # 添加攻击来源分析
        df[f'{target_brand}_攻击来源'] = df.apply(
            lambda row: self._identify_attack_sources(row, target_brand, related_brands),
            axis=1
        )
        
        # 添加攻击类型分析
        df[f'{target_brand}_攻击类型'] = df.apply(
            lambda row: self._classify_attack_types(row, target_brand),
            axis=1
        )
        
        return df
    
    def _calculate_relationship_stats(self, row: pd.Series, 
                                    target_brand: str, related_brands: List[str]) -> dict:
        """计算关系统计信息"""
        stats = {
            '涉及品牌数': 0,
            '正面关系数': 0,
            '负面关系数': 0,
            '中立关系数': 0,
            '总影响强度': 0.0
        }
        
        for related_brand in related_brands:
            strength_col = f'{target_brand}_vs_{related_brand}_强度'
            if strength_col in row:
                strength = row[strength_col]
                if strength != 0:
                    stats['涉及品牌数'] += 1
                    stats['总影响强度'] += strength
                    
                    if strength > 0.3:
                        stats['正面关系数'] += 1
                    elif strength < -0.3:
                        stats['负面关系数'] += 1
                    else:
                        stats['中立关系数'] += 1
        
        return stats
    
    def _identify_main_competitor(self, row: pd.Series, 
                                target_brand: str, related_brands: List[str]) -> str:
        """识别主要竞争对手"""
        max_negative_strength = 0
        main_competitor = "无"
        
        for related_brand in related_brands:
            strength_col = f'{target_brand}_vs_{related_brand}_强度'
            if strength_col in row:
                strength = row[strength_col]
                if strength < 0 and abs(strength) > max_negative_strength:
                    max_negative_strength = abs(strength)
                    main_competitor = related_brand
        
        return main_competitor
    
    def _summarize_relationship_types(self, row: pd.Series, 
                                    target_brand: str, related_brands: List[str]) -> List[str]:
        """汇总关系类型"""
        relationship_types = []
        
        for related_brand in related_brands:
            type_col = f'{target_brand}_vs_{related_brand}_类型'
            if type_col in row and row[type_col] != '无关系':
                relationship_types.append(f"{related_brand}: {row[type_col]}")
        
        return relationship_types
    
    def _analyze_attack_relationships(self, row: pd.Series, 
                                     target_brand: str, related_brands: List[str]) -> dict:
        """分析攻击关系"""
        attack_stats = {
            '被攻击次数': 0,
            '攻击他人次数': 0,
            '攻击强度': 0.0,
            '攻击来源': [],
            '攻击类型': defaultdict(int)
        }
        
        # 分析目标品牌被攻击的情况
        for related_brand in related_brands:
            type_col = f'{target_brand}_vs_{related_brand}_类型'
            strength_col = f'{target_brand}_vs_{related_brand}_强度'
            
            if type_col in row and strength_col in row:
                if '攻击' in row[type_col] and row[strength_col] < 0:
                    attack_stats['被攻击次数'] += 1
                    attack_stats['攻击强度'] += abs(row[strength_col])
                    attack_stats['攻击来源'].append(related_brand)
                    
                    # 解析攻击类型
                    if '产品攻击' in row[type_col]:
                        attack_stats['攻击类型']['产品攻击'] += 1
                    elif '服务攻击' in row[type_col]:
                        attack_stats['攻击类型']['服务攻击'] += 1
                    elif '价格攻击' in row[type_col]:
                        attack_stats['攻击类型']['价格攻击'] += 1
                    elif '安全攻击' in row[type_col]:
                        attack_stats['攻击类型']['安全攻击'] += 1
                    elif '技术攻击' in row[type_col]:
                        attack_stats['攻击类型']['技术攻击'] += 1
                    elif '品牌攻击' in row[type_col]:
                        attack_stats['攻击类型']['品牌攻击'] += 1
                    else:
                        attack_stats['攻击类型']['其他攻击'] += 1
        
        # 分析目标品牌攻击其他品牌的情况
        for related_brand in related_brands:
            type_col = f'{target_brand}_vs_{related_brand}_类型'
            strength_col = f'{target_brand}_vs_{related_brand}_强度'
            
            if type_col in row and strength_col in row:
                if '攻击' in row[type_col] and row[strength_col] > 0:
                    attack_stats['攻击他人次数'] += 1
        
        # 转换为标准字典
        attack_stats['攻击类型'] = dict(attack_stats['攻击类型'])
        
        return attack_stats
    
    def _identify_attack_sources(self, row: pd.Series, 
                               target_brand: str, related_brands: List[str]) -> List[str]:
        """识别攻击来源"""
        attack_sources = []
        
        for related_brand in related_brands:
            type_col = f'{target_brand}_vs_{related_brand}_类型'
            strength_col = f'{target_brand}_vs_{related_brand}_强度'
            
            if type_col in row and strength_col in row:
                if '攻击' in row[type_col] and row[strength_col] < 0:
                    attack_sources.append(related_brand)
        
        return attack_sources
    
    def _classify_attack_types(self, row: pd.Series, target_brand: str) -> dict:
        """分类攻击类型"""
        attack_types = defaultdict(int)
        
        # 检查目标品牌被攻击的情况
        for col in row.index:
            if col.startswith(f'{target_brand}_vs_') and col.endswith('_类型'):
                if '攻击' in row[col]:
                    # 解析攻击类型
                    if '产品攻击' in row[col]:
                        attack_types['产品攻击'] += 1
                    elif '服务攻击' in row[col]:
                        attack_types['服务攻击'] += 1
                    elif '价格攻击' in row[col]:
                        attack_types['价格攻击'] += 1
                    elif '安全攻击' in row[col]:
                        attack_types['安全攻击'] += 1
                    elif '技术攻击' in row[col]:
                        attack_types['技术攻击'] += 1
                    elif '品牌攻击' in row[col]:
                        attack_types['品牌攻击'] += 1
                    else:
                        attack_types['其他攻击'] += 1
        
        return dict(attack_types)
    
    
    def _calculate_relationship_network(self, df: pd.DataFrame, target_brand: str) -> dict:
        """计算关系网络"""
        network = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
        
        # 遍历所有相关品牌列
        for col in df.columns:
            if col.startswith(f'{target_brand}_vs_') and col.endswith('_强度'):
                related_brand = col.split('_vs_')[1].split('_强度')[0]
                
                for _, row in df.iterrows():
                    strength = row[col]
                    if strength > 0.3:
                        network[related_brand]["positive"] += 1
                    elif strength < -0.3:
                        network[related_brand]["negative"] += 1
                    elif strength != 0:
                        network[related_brand]["neutral"] += 1
        
        return dict(network)
    
    def _calculate_attack_analysis(self, df: pd.DataFrame, target_brand: str) -> dict:
        """计算攻击关系分析"""
        attack_stats = {
            "attacked_count": 0,  # 被攻击次数
            "attacking_count": 0,  # 攻击他人次数
            "attack_types": defaultdict(int)
        }
        
        # 分析攻击关系列
        for col in df.columns:
            if col.startswith(f'{target_brand}_vs_') and col.endswith('_类型'):
                related_brand = col.split('_vs_')[1].split('_类型')[0]
                
                for _, row in df.iterrows():
                    relationship_type = row[col]
                    strength_col = f'{target_brand}_vs_{related_brand}_强度'
                    strength = row[strength_col] if strength_col in row else 0
                    
                    if '攻击' in relationship_type:
                        if strength < 0:  # 目标品牌被攻击
                            attack_stats["attacked_count"] += 1
                            
                            # 记录攻击类型
                            if '产品攻击' in relationship_type:
                                attack_stats["attack_types"]["产品攻击"] += 1
                            elif '服务攻击' in relationship_type:
                                attack_stats["attack_types"]["服务攻击"] += 1
                            elif '价格攻击' in relationship_type:
                                attack_stats["attack_types"]["价格攻击"] += 1
                            elif '安全攻击' in relationship_type:
                                attack_stats["attack_types"]["安全攻击"] += 1
                            elif '技术攻击' in relationship_type:
                                attack_stats["attack_types"]["技术攻击"] += 1
                            elif '品牌攻击' in relationship_type:
                                attack_stats["attack_types"]["品牌攻击"] += 1
                            else:
                                attack_stats["attack_types"]["其他攻击"] += 1
                        elif strength > 0:  # 目标品牌攻击他人
                            attack_stats["attacking_count"] += 1
        
        # 转换为标准字典
        attack_stats["attack_types"] = dict(attack_stats["attack_types"])
        
        return attack_stats
    
    def _get_top_attack_sources(self, df: pd.DataFrame, target_brand: str) -> List[Tuple[str, int]]:
        """获取主要攻击来源"""
        attack_sources = defaultdict(int)
        
        # 分析攻击关系列
        for col in df.columns:
            if col.startswith(f'{target_brand}_vs_') and col.endswith('_类型'):
                related_brand = col.split('_vs_')[1].split('_类型')[0]
                
                for _, row in df.iterrows():
                    relationship_type = row[col]
                    strength_col = f'{target_brand}_vs_{related_brand}_强度'
                    strength = row[strength_col] if strength_col in row else 0
                    
                    if '攻击' in relationship_type and strength < 0:
                        attack_sources[related_brand] += 1
        
        # 按攻击次数排序
        return sorted(attack_sources.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_attack_type_distribution(self, df: pd.DataFrame, target_brand: str) -> dict:
        """获取攻击类型分布"""
        attack_types = defaultdict(int)
        
        # 分析攻击关系列
        for col in df.columns:
            if col.startswith(f'{target_brand}_vs_') and col.endswith('_类型'):
                for _, row in df.iterrows():
                    relationship_type = row[col]
                    
                    if '攻击' in relationship_type:
                        # 解析攻击类型
                        if '产品攻击' in relationship_type:
                            attack_types["产品攻击"] += 1
                        elif '服务攻击' in relationship_type:
                            attack_types["服务攻击"] += 1
                        elif '价格攻击' in relationship_type:
                            attack_types["价格攻击"] += 1
                        elif '安全攻击' in relationship_type:
                            attack_types["安全攻击"] += 1
                        elif '技术攻击' in relationship_type:
                            attack_types["技术攻击"] += 1
                        elif '品牌攻击' in relationship_type:
                            attack_types["品牌攻击"] += 1
                        else:
                            attack_types["其他攻击"] += 1
        
        return dict(attack_types)

    def _discover_brands_in_text(self, text: str) -> List[str]:
        """发现文本中的所有品牌"""
        found_brands = []
        
        # 方法1：使用实体识别器
        if self.entity_recognizer:
            entities = self.entity_recognizer.extract_entities(text)
            brand_entities = [e for e in entities.get('实体列表', []) 
                            if e.get('类别') == 'BRAND']
            for entity in brand_entities:
                found_brands.append(entity['文本'])
        
        # 方法2：基于预定义品牌列表匹配
        for brand in self.automotive_brands:
            if brand in text:
                found_brands.append(brand)
        
        return list(set(found_brands))  # 去重

    def _analyze_single_brand_in_row(self, row: pd.Series, brand: str, text: str) -> dict:
        """分析单行中单个品牌的情感"""
        analysis = self._analyze_brand_sentiment_with_csv(row, brand, text)
        
        return {
            '情感得分': analysis['strength'],
            '情感标签': analysis['sentiment_label'],
            '关键证据': analysis['evidence'],
            '置信度': abs(analysis['strength']) / 2.0  # 转换为0-1的置信度
        }

    def _analyze_brand_pair_in_row(self, row: pd.Series, brand1: str, brand2: str, text: str) -> dict:
        """分析单行中两个品牌的关系"""
        relationship_analysis = self._analyze_brand_pair_relationship(row, brand1, brand2, text)
        
        relationship_type = "中立"
        if relationship_analysis['strength'] > 0.3:
            relationship_type = "优于"
        elif relationship_analysis['strength'] < -0.3:
            relationship_type = "不如"
        elif 'attack' in relationship_analysis['relationship']:
            relationship_type = "攻击"
        
        return {
            '关系类型': relationship_type,
            '关系强度': relationship_analysis['strength'],
            '关系证据': relationship_analysis['evidence']
        }

    def _analyze_attack_direction_in_row(self, row: pd.Series, text: str, brands: List[str]) -> dict:
        """分析单行中的攻击指向"""
        attack_info = {
            '目标品牌': '',
            '攻击类型': '',
            '攻击强度': 0.0,
            '攻击证据': ''
        }
        
        max_attack_strength = 0
        
        for brand in brands:
            for attack_type, patterns in self.attack_patterns.items():
                for pattern in patterns:
                    # 改进：直接在原文中搜索，不替换(.+?)
                    if re.search(pattern, text):
                        # 检查匹配是否真的与当前品牌相关
                        match = re.search(pattern, text)
                        if match and brand in match.group(0):
                            # 计算攻击强度
                            strength = self._calculate_attack_strength(text, brand, attack_type)
                            
                            if strength > max_attack_strength:
                                max_attack_strength = strength
                                attack_info['目标品牌'] = brand
                                attack_info['攻击类型'] = attack_type
                                attack_info['攻击强度'] = strength
                                attack_info['攻击证据'] = self._extract_brand_evidence(text, brand)
        
        return attack_info

    def _calculate_attack_strength(self, text: str, brand: str, attack_type: str) -> float:
        """计算攻击强度"""
        base_strength = 1.0
        
        # 根据攻击类型调整基础强度
        type_weights = {
            "直接贬低": 1.5,
            "对比攻击": 1.3,
            "价值质疑": 1.4,
            "预测攻击": 1.2
        }
        
        strength = base_strength * type_weights.get(attack_type, 1.0)
        
        # 负面词汇加权
        negative_words = ["垃圾", "烂", "坑", "完爆", "摸黑", "反噬", "代差", "逆风"]
        for word in negative_words:
            if word in text:
                strength += 0.2
        
        return min(strength, 3.0)  # 最大强度限制为3.0

    # 新增
    def _extract_detailed_analysis(self, row: pd.Series) -> dict:
        """解析'分析详情'列，提取高质量情感信息"""
        
        detailed_info = {
            'global_sentiment': '中立',
            'global_score': 0.0,
            'confidence': 0.5,
            'emotion_details': [],
            'keywords': [],
            'auto_terms': {}
        }
        
        # 检查是否有分析详情列
        if '分析详情' not in row:
            return detailed_info
        
        try:
            analysis_data = row['分析详情']
            
            # 解析字符串格式的数据
            if isinstance(analysis_data, str):
                import ast
                analysis_data = ast.literal_eval(analysis_data)
            
            # 提取极性分类信息
            if '极性分类' in analysis_data:
                polarity_data = analysis_data['极性分类']
                detailed_info['raw_polarity_data'] = polarity_data
                
                detailed_info['global_sentiment'] = polarity_data.get('极性分类', '中立')
                raw_confidence = polarity_data.get('极性置信度', 0.5)  # 这里定义 raw_confidence
                detailed_info['confidence'] = raw_confidence
                
                # 改进的全局得分计算
                if detailed_info['global_sentiment'] == '正面':
                    # 对于正面情感，使用适度的得分计算
                    detailed_info['global_score'] = min(1.5, raw_confidence * 1.3)
                elif detailed_info['global_sentiment'] == '负面':
                    # 对于负面情感，使用更保守的得分计算
                    detailed_info['global_score'] = max(-1.2, -raw_confidence * 1.0)
                else:
                    # 中性情感保持较小得分
                    detailed_info['global_score'] = 0.0
            
            # 提取情感细分
            if '情感细分' in analysis_data:
                detailed_info['emotion_details'] = analysis_data['情感细分']
            
            # 提取关键词
            if '关键词' in analysis_data:
                detailed_info['keywords'] = analysis_data['关键词']
            
            # 提取汽车术语
            if '汽车术语识别' in analysis_data:
                detailed_info['auto_terms'] = analysis_data['汽车术语识别'].get('触发术语', {})
            
            # 提取总置信度
            if '总置信度' in analysis_data:
                total_confidence = analysis_data['总置信度'].get('置信度评分', 0.5)
                detailed_info['confidence'] = max(detailed_info['confidence'], total_confidence)
                
        except Exception as e:
            print(f"解析分析详情时出错: {e}")
        
        return detailed_info



    def _calculate_brand_position_weight(self, text: str, brand: str) -> float:
        """计算品牌位置权重（主语位置权重更高）"""
        
        sentences = re.split(r'[。！？；]', text)
        max_weight = 0.0
        
        for sentence in sentences:
            if brand in sentence:
                brand_pos = sentence.find(brand)
                sentence_len = len(sentence)
                
                # 位置权重：越靠前权重越高
                if sentence_len > 0:
                    position_ratio = 1 - (brand_pos / sentence_len)
                    
                    # 主语位置检测
                    if brand_pos < len(sentence) * 0.3:  # 前30%位置
                        position_weight = 0.8 + position_ratio * 0.2
                    else:
                        position_weight = 0.3 + position_ratio * 0.4
                    
                    max_weight = max(max_weight, position_weight)
        
        return max_weight

    def _analyze_brand_grammar_relation(self, text: str, brand: str, detailed_analysis: dict) -> float:
        """分析品牌与情感词的语法关系"""
        
        # 简化的语法关系分析
        grammar_patterns = {
            '直接修饰': [rf'{brand}[的]?[很真挺]?[好差不错糟糕]', rf'[好差不错糟糕][的]?{brand}'],
            '主语关系': [rf'{brand}[是就]', rf'^{brand}'],
            '宾语关系': [rf'[选择买推荐避免]{brand}', rf'{brand}[值得不值得]']
        }
        
        total_weight = 0.0
        pattern_count = 0
        
        for relation_type, patterns in grammar_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    pattern_count += 1
                    if relation_type == '直接修饰':
                        total_weight += 0.9
                    elif relation_type == '主语关系':
                        total_weight += 0.7
                    elif relation_type == '宾语关系':
                        total_weight += 0.5
        
        return min(1.0, total_weight / max(1, pattern_count))

    def _analyze_keyword_attribution(self, text: str, brand: str, keywords: list) -> float:
        """分析关键词与品牌的归属关系"""
        
        if not keywords:
            return 0.5
        
        brand_pos = text.find(brand)
        if brand_pos == -1:
            return 0.5
        
        attribution_weight = 0.0
        relevant_keywords = 0
        
        for keyword_item in keywords:
            if isinstance(keyword_item, dict):
                keyword = keyword_item.get('词语', '')
                score = keyword_item.get('得分', 0.0)
                
                keyword_pos = text.find(keyword)
                if keyword_pos != -1:
                    # 计算关键词与品牌的距离
                    distance = abs(keyword_pos - brand_pos)
                    
                    # 距离越近，归属性越强
                    if distance < 10:
                        distance_weight = 1.0
                    elif distance < 20:
                        distance_weight = 0.7
                    elif distance < 50:
                        distance_weight = 0.4
                    else:
                        distance_weight = 0.1
                    
                    attribution_weight += distance_weight * score
                    relevant_keywords += 1
        
        return min(1.0, attribution_weight / max(1, relevant_keywords))

    def _analyze_multi_brand_comparison(self, text: str, target_brand: str) -> float:
        """分析多品牌对比场景下的情感归属"""
        
        # 发现文本中的其他品牌
        other_brands = []
        for brand in self.automotive_brands:
            if brand != target_brand and brand in text:
                other_brands.append(brand)
        
        if not other_brands:
            return 0.6  # 只有目标品牌，中等归属
        
        # 对比模式检测
        comparison_patterns = {
            '优于模式': [rf'{target_brand}[比]([^比]*?)好', rf'{target_brand}[比]([^比]*?)强'],
            '不如模式': [rf'{target_brand}[比]([^比]*?)差', rf'{target_brand}不如([^，。！？]*)'],
            '排除模式': [rf'除了{target_brand}', rf'只有{target_brand}不'],
            '转折模式': [rf'虽然{target_brand}.*?但是', rf'{target_brand}.*?不过']
        }
        
        attribution_adjustment = 0.0
        
        for pattern_type, patterns in comparison_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if pattern_type == '优于模式':
                        attribution_adjustment += 0.8  # 强正面归属
                    elif pattern_type == '不如模式':
                        attribution_adjustment += 0.8  # 强负面归属
                    elif pattern_type == '排除模式':
                        attribution_adjustment -= 0.5  # 降低归属
                    elif pattern_type == '转折模式':
                        attribution_adjustment += 0.3  # 中等归属
        
        return min(1.0, 0.5 + attribution_adjustment)

    def _analyze_sentiment_word_distance(self, text: str, brand: str) -> float:
        """分析情感词与品牌的距离"""
        
        brand_pos = text.find(brand)
        if brand_pos == -1:
            return 0.5
        
        sentiment_words = []
        for sentiment_type, words in self.sentiment_patterns.items():
            sentiment_words.extend(words)
        
        min_distance = float('inf')
        closest_sentiment = None
        
        for word in sentiment_words:
            word_pos = text.find(word)
            if word_pos != -1:
                distance = abs(word_pos - brand_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_sentiment = word
        
        if min_distance == float('inf'):
            return 0.3
        
        # 距离越近，权重越高
        if min_distance < 5:
            return 1.0
        elif min_distance < 15:
            return 0.8
        elif min_distance < 30:
            return 0.5
        else:
            return 0.2

    def _multi_brand_sentiment_distribution(self, text: str, detailed_analysis: dict, mentioned_brands: list) -> dict:
        """在多品牌场景下分配全局情感"""
        
        if len(mentioned_brands) <= 1:
            return {mentioned_brands[0]: 1.0} if mentioned_brands else {}
        
        brand_weights = {}
        global_sentiment_score = detailed_analysis['global_score']
        
        # 1. 计算每个品牌的基础归因权重
        for brand in mentioned_brands:
            attribution = self._brand_attribution_analysis(text, brand, detailed_analysis)
            brand_weights[brand] = attribution['attribution_score']
        
        # 2. 检测对比关系并调整权重
        comparison_adjustments = self._detect_brand_comparisons(text, mentioned_brands)
        
        # 3. 应用对比调整
        for brand_pair, comparison_type in comparison_adjustments.items():
            brand1, brand2 = brand_pair
            
            if comparison_type == 'brand1_better':
                # brand1比brand2好
                if global_sentiment_score > 0:  # 全局正面
                    brand_weights[brand1] = max(brand_weights.get(brand1, 0), 0.8)
                    brand_weights[brand2] = min(brand_weights.get(brand2, 0), -0.3)
                else:  # 全局负面
                    brand_weights[brand1] = max(brand_weights.get(brand1, 0), 0.3)
                    brand_weights[brand2] = min(brand_weights.get(brand2, 0), -0.8)
                    
            elif comparison_type == 'brand2_better':
                # brand2比brand1好
                if global_sentiment_score > 0:  # 全局正面
                    brand_weights[brand2] = max(brand_weights.get(brand2, 0), 0.8)
                    brand_weights[brand1] = min(brand_weights.get(brand1, 0), -0.3)
                else:  # 全局负面
                    brand_weights[brand2] = max(brand_weights.get(brand2, 0), 0.3)
                    brand_weights[brand1] = min(brand_weights.get(brand1, 0), -0.8)
        
        # 4. 标准化权重
        total_abs_weight = sum(abs(w) for w in brand_weights.values())
        if total_abs_weight > 0:
            for brand in brand_weights:
                brand_weights[brand] = brand_weights[brand] / total_abs_weight
        
        return brand_weights

    def _detect_brand_comparisons(self, text: str, brands: list) -> dict:
        """检测品牌之间的对比关系"""
        
        comparisons = {}
        
        for i, brand1 in enumerate(brands):
            for j, brand2 in enumerate(brands):
                if i >= j:
                    continue
                
                # 检测对比模式
                comparison_patterns = {
                    'brand1_better': [
                        rf'{brand1}[比].*?{brand2}.*?[好强优秀]',
                        rf'{brand1}.*?胜过.*?{brand2}',
                        rf'选择{brand1}.*?而不是{brand2}',
                        rf'{brand1}.*?比{brand2}.*?[值得推荐]'
                    ],
                    'brand2_better': [
                        rf'{brand2}[比].*?{brand1}.*?[好强优秀]',
                        rf'{brand2}.*?胜过.*?{brand1}',
                        rf'选择{brand2}.*?而不是{brand1}',
                        rf'{brand2}.*?比{brand1}.*?[值得推荐]',
                        rf'{brand1}.*?不如.*?{brand2}'
                    ]
                }
                
                for comparison_type, patterns in comparison_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text):
                            comparisons[(brand1, brand2)] = comparison_type
                            break
        
        return comparisons

    def _get_text_hash(self, text: str) -> str:
        """生成文本的哈希值"""
        return hashlib.md5(str(text).encode('utf-8')).hexdigest()
    
    def _get_df_cache_key(self, df: pd.DataFrame, target_brand: str) -> str:
        """生成DataFrame缓存键"""
        df_hash = hashlib.md5(df.to_string().encode('utf-8')).hexdigest()
        return f"{df_hash}_{target_brand}"
    
    def _get_brand_cache_key(self, text_hash: str, brand1: str, brand2: str = None) -> str:
        """生成品牌特定分析的缓存键"""
        if brand2:
            return f"{text_hash}_{brand1}_{brand2}"
        return f"{text_hash}_{brand1}"

    
    def _fill_analysis_results(self, result_df, idx, analysis_result, target_brand):
        """填充分析结果到DataFrame"""
        # 填充提及品牌列表
        result_df.loc[idx, '提及品牌列表'] = ', '.join(analysis_result['提及品牌列表'])
        
        # 填充目标品牌分析结果
        if target_brand and target_brand in analysis_result['品牌分析']:
            brand_data = analysis_result['品牌分析'][target_brand]
            result_df.loc[idx, f'{target_brand}_情感得分'] = brand_data['情感得分']
            result_df.loc[idx, f'{target_brand}_情感标签'] = brand_data['情感标签']
            result_df.loc[idx, f'{target_brand}_置信度'] = brand_data['置信度']
            result_df.loc[idx, f'{target_brand}_关键证据'] = brand_data['关键证据']
        
        # 填充攻击指向分析结果
        attack_data = analysis_result['攻击指向']
        if attack_data and attack_data.get('目标品牌'):
            result_df.loc[idx, '攻击指向_目标品牌'] = attack_data['目标品牌']
            result_df.loc[idx, '攻击指向_类型'] = attack_data.get('攻击类型', '')
            result_df.loc[idx, '攻击指向_强度'] = attack_data.get('攻击强度', 0.0)
            result_df.loc[idx, '攻击指向_证据'] = attack_data.get('攻击证据', '')
    

    
    # 自动清理缓存方法
    def _auto_cleanup_if_needed(self):
        """自动清理：当缓存超过阈值时触发"""
        total_size = len(self._text_base_cache) + len(self._brand_specific_cache) + len(self._df_cache)
        
        if total_size > self.max_cache_size:
            self.logger.info(f"🧹 缓存达到阈值({total_size}/{self.max_cache_size})，触发自动清理")
            self._smart_cleanup()

    def _smart_cleanup(self):
        """智能清理：按价值保留缓存"""
        import time
        
        before_size = len(self._text_base_cache) + len(self._brand_specific_cache) + len(self._df_cache)
        
        # 按重要性清理
        self._cleanup_cache_by_ratio(self._df_cache, 'DataFrame', 0.3)      # 保留30%
        self._cleanup_cache_by_ratio(self._text_base_cache, '文本基础', 0.6)  # 保留60%  
        self._cleanup_cache_by_ratio(self._brand_specific_cache, '品牌特定', 0.8) # 保留80%
        
        after_size = len(self._text_base_cache) + len(self._brand_specific_cache) + len(self._df_cache)
        self._cache_stats['cleanups'] += 1
        self._cache_stats['last_cleanup_time'] = time.time()
        
        self.logger.info(f"✅ 智能清理完成: {before_size} → {after_size} (-{before_size-after_size})")

    def _cleanup_cache_by_ratio(self, cache: dict, cache_name: str, keep_ratio: float):
        """按比例清理单个缓存"""
        if not cache:
            return
            
        before_count = len(cache)
        keep_count = max(1, int(len(cache) * keep_ratio))
        
        # 保留最新的条目
        keys = list(cache.keys())
        keys_to_keep = keys[-keep_count:]
        new_cache = {k: cache[k] for k in keys_to_keep}
        cache.clear()
        cache.update(new_cache)
        
        self.logger.info(f"  └─ {cache_name}: {before_count} → {len(cache)}")

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            '文本基础分析缓存数量': len(self._text_base_cache),
            '品牌特定分析缓存数量': len(self._brand_specific_cache),
            'DataFrame缓存数量': len(self._df_cache),
            '预估内存使用': f"{(len(self._text_base_cache) + len(self._brand_specific_cache)) * 0.1:.1f}KB"
        }

    def clear_cache(self, cache_type: str = 'all'):
        """手动清理缓存"""
        before_stats = self.get_cache_stats()
        
        if cache_type in ['all', 'text']:
            self._text_base_cache.clear()
            
        if cache_type in ['all', 'brand']:
            self._brand_specific_cache.clear()
            
        if cache_type in ['all', 'df']:
            self._df_cache.clear()
        
        after_stats = self.get_cache_stats()
        self.logger.info(f"🧹 手动清理完成: {before_stats['总缓存数']} → {after_stats['总缓存数']}")

    def analyze_row_brand_sentiment(self, row: pd.Series, target_brand: str = None) -> dict:
        """使用分层缓存的品牌情感分析"""
        
            # 新增：检查是否需要自动清理
        self._auto_cleanup_if_needed()
        
        content_col = self._get_content_column(pd.DataFrame([row]))
        text = str(row[content_col])
        text_hash = self._get_text_hash(text)
        
        # 1. 检查或计算基础分析（品牌发现 + 攻击分析）
        if text_hash not in self._text_base_cache:
            # 执行基础分析（与品牌无关）
            mentioned_brands = self._discover_brands_in_text(text)
            attack_analysis = self._analyze_attack_direction_in_row(row, text, mentioned_brands)
            
            # 存入缓存
            self._text_base_cache[text_hash] = {
                'mentioned_brands': mentioned_brands,
                'attack_analysis': attack_analysis
            }
        
        # 获取缓存的基础分析结果
        base_data = self._text_base_cache[text_hash]
        mentioned_brands = base_data['mentioned_brands']
        attack_analysis = base_data['attack_analysis']
        
        # 2. 构建返回结果
        results = {
            '提及品牌列表': mentioned_brands,
            '品牌分析': {},
            '攻击指向': attack_analysis,
            '品牌关系': {}
        }
        
        # 3. 处理品牌特定分析（使用缓存）
        if target_brand and target_brand in mentioned_brands:
            # 分析目标品牌
            brand_cache_key = self._get_brand_cache_key(text_hash, target_brand)
            if brand_cache_key not in self._brand_specific_cache:
                brand_analysis = self._analyze_single_brand_in_row(row, target_brand, text)
                self._brand_specific_cache[brand_cache_key] = brand_analysis
            
            results['品牌分析'][target_brand] = self._brand_specific_cache[brand_cache_key]
            
            # 分析品牌关系
            for other_brand in mentioned_brands:
                if other_brand != target_brand:
                    relation_cache_key = self._get_brand_cache_key(text_hash, target_brand, other_brand)
                    if relation_cache_key not in self._brand_specific_cache:
                        relationship = self._analyze_brand_pair_in_row(row, target_brand, other_brand, text)
                        self._brand_specific_cache[relation_cache_key] = relationship
                    
                    results['品牌关系'][f'{target_brand}_vs_{other_brand}'] = self._brand_specific_cache[relation_cache_key]
        else:
            # 分析所有提及的品牌
            for brand in mentioned_brands:
                brand_cache_key = self._get_brand_cache_key(text_hash, brand)
                if brand_cache_key not in self._brand_specific_cache:
                    brand_analysis = self._analyze_single_brand_in_row(row, brand, text)
                    self._brand_specific_cache[brand_cache_key] = brand_analysis
                
                results['品牌分析'][brand] = self._brand_specific_cache[brand_cache_key]
        
        return results

# 目标函数（主要接口）
    def analyze_brand_relationships(self, df: pd.DataFrame, target_brand: str) -> pd.DataFrame:
        """分析以目标品牌为中心的关系网络"""
        
        # 复制数据避免修改原数据
        analysis_df = df.copy()
        
        # 1. 先用实体识别器找出所有相关品牌
        related_brands = self._discover_related_brands(df, target_brand)
        
        print(f"发现与'{target_brand}'相关的品牌: {related_brands}")
        
        # 2. 为目标品牌添加基础分析列
        analysis_df = self._add_target_brand_analysis(analysis_df, target_brand)
        
        # 3. 为每个发现的相关品牌添加关系分析列
        for related_brand in related_brands:
            if related_brand != target_brand:
                analysis_df = self._add_brand_relationship_columns(
                    analysis_df, target_brand, related_brand
                )
        
        # 4. 添加综合关系分析
        analysis_df = self._add_comprehensive_relationship_analysis(
            analysis_df, target_brand, related_brands
        )
        
        # 5. 添加攻击关系分析
        analysis_df = self._add_attack_relationship_analysis(
            analysis_df, target_brand, related_brands
        )
        
        return analysis_df
    
    def process_csv_brand_analysis(self, df: pd.DataFrame, target_brand: str = None) -> pd.DataFrame:
        """优化版本：使用完整缓存机制"""
        
        # 1. 检查DataFrame级缓存
        df_cache_key = self._get_df_cache_key(df, target_brand or "ALL")
        
        if df_cache_key in self._df_cache:
            self.logger.info(f"✅ 使用完整缓存 - 品牌: {target_brand or '全部'}")
            return self._df_cache[df_cache_key].copy()
        
        print(f"🔄 执行分析 - 品牌: {target_brand or '全部'}")
        
        # 2. 执行分析
        result_df = df.copy()
        
        # 初始化结果列
        if target_brand:
            result_df['目标品牌'] = target_brand
            result_df[f'{target_brand}_情感得分'] = 0.0
            result_df[f'{target_brand}_情感标签'] = '中立'
            result_df[f'{target_brand}_置信度'] = 0.0
            result_df[f'{target_brand}_关键证据'] = ''
        
        result_df['提及品牌列表'] = ''
        result_df['攻击指向_目标品牌'] = ''
        result_df['攻击指向_类型'] = ''
        result_df['攻击指向_强度'] = 0.0
        result_df['攻击指向_证据'] = ''
        
        # 3. 逐行处理
        total_rows = len(result_df)
        text_cache_hits = 0
        brand_cache_hits = 0
        
        progress_bar = tqdm(total=total_rows, desc=f"品牌分析 - {target_brand or '全部'}", unit="行")
        
        for idx, row in result_df.iterrows():
            try:
                # 获取缓存统计（处理前）
                old_text_cache_size = len(self._text_base_cache)
                old_brand_cache_size = len(self._brand_specific_cache)
                
                # 执行分析
                analysis_result = self.analyze_row_brand_sentiment(row, target_brand)
                
                # 统计缓存命中
                if len(self._text_base_cache) == old_text_cache_size:
                    text_cache_hits += 1
                if len(self._brand_specific_cache) == old_brand_cache_size:
                    brand_cache_hits += 1
                
                # 填充结果
                self._fill_analysis_results(result_df, idx, analysis_result, target_brand)
                
            except Exception as e:
                print(f"❌ 处理第{idx}行时出错: {e}")
                continue
            finally:
                progress_bar.update(1)
        
        progress_bar.close()
        
        # 4. 显示缓存统计
        self.logger.info(f"📊 缓存统计:")
        self.logger.info(f"   文本基础分析缓存命中: {text_cache_hits}/{total_rows} ({text_cache_hits/total_rows*100:.1f}%)")
        self.logger.info(f"   品牌特定分析缓存命中: {brand_cache_hits}/{total_rows} ({brand_cache_hits/total_rows*100:.1f}%)")
        
        # 5. 缓存完整结果
        self._df_cache[df_cache_key] = result_df.copy()
        
        return result_df

    def generate_brand_relationship_report(self, df: pd.DataFrame, target_brand: str) -> dict:
        """生成品牌关系分析报告"""
        
        # 过滤包含目标品牌的数据
        brand_data = df[df[f'{target_brand}_提及'] == True].copy()
        
        if len(brand_data) == 0:
            return {"error": f"没有找到关于'{target_brand}'的讨论"}
        
        # 计算整体情感分布
        sentiment_counts = brand_data[f'{target_brand}_情感倾向'].value_counts().to_dict()
        
        # 计算关系网络
        relationship_network = self._calculate_relationship_network(brand_data, target_brand)
        
        # 计算攻击关系分析
        attack_analysis = self._calculate_attack_analysis(brand_data, target_brand)
        
        # 生成报告
        report = {
            "target_brand": target_brand,
            "total_mentions": len(brand_data),
            "sentiment_distribution": sentiment_counts,
            "relationship_network": relationship_network,
            "attack_analysis": attack_analysis,
            "top_attack_sources": self._get_top_attack_sources(brand_data, target_brand),
            "attack_type_distribution": self._get_attack_type_distribution(brand_data, target_brand)
        }
        
        return report

    def get_brand_summary_stats(self, df: pd.DataFrame, target_brand: str) -> dict:
        """获取品牌的汇总统计信息（用于可视化）"""
        
        if f'{target_brand}_情感标签' not in df.columns:
            return {"error": "请先运行 process_csv_brand_analysis 方法"}
        
        # 过滤包含目标品牌的行
        brand_rows = df[df['提及品牌列表'].str.contains(target_brand, na=False)]
        
        stats = {
            "总提及次数": len(brand_rows),
            "情感分布": brand_rows[f'{target_brand}_情感标签'].value_counts().to_dict(),
            "平均情感得分": brand_rows[f'{target_brand}_情感得分'].mean(),
            "被攻击次数": len(df[df['攻击指向_目标品牌'] == target_brand]),
            "攻击类型分布": df[df['攻击指向_目标品牌'] == target_brand]['攻击指向_类型'].value_counts().to_dict(),
            "高置信度正面": len(brand_rows[(brand_rows[f'{target_brand}_情感标签'] == '正面') & 
                                        (brand_rows[f'{target_brand}_置信度'] > 0.7)]),
            "高置信度负面": len(brand_rows[(brand_rows[f'{target_brand}_情感标签'] == '负面') & 
                                        (brand_rows[f'{target_brand}_置信度'] > 0.7)])
        }
        
        return stats




if __name__ == "__main__":
# 使用示例
    import pandas as pd

    # 初始化分析器
    analyzer = BrandCentricAnalyzer()

    # 读取CSV数据
    df = pd.read_csv("D:\OneDrive\Desktop\weibo_8_15_label.csv")

    # 方法1: 针对特定品牌分析
    target_brand = "理想"
    result_df = analyzer.process_csv_brand_analysis(df, target_brand)

    # 方法2: 通用品牌分析（不指定目标品牌）
    # result_df = analyzer.process_csv_brand_analysis(df)

    # 获取统计信息用于可视化
    stats = analyzer.get_brand_summary_stats(result_df, "理想")
    print(stats)

    # 查看结果列
    print(result_df.columns.tolist())

    # 筛选正面评价
    positive_reviews = result_df[result_df['理想_情感标签'] == '正面']

    # 筛选攻击指向理想的评论
    attack_reviews = result_df[result_df['攻击指向_目标品牌'] == '理想']
    print(f"攻击 {target_brand} 的评论：{attack_reviews}")
    
    stats = analyzer.get_cache_stats()
            