from textblob import TextBlob
from typing import Dict, List, Tuple
import re

class SentimentAnalyzer:
    """情感分析器"""
    
    # 情感词典
    POSITIVE_WORDS = {
        'win', 'victory', 'success', 'excellent', 'brilliant', 'outstanding',
        'great', 'good', 'positive', 'strong', 'impressive', 'confident',
        'ready', 'fit', 'available', 'return', 'boost', 'improve'
    }
    
    NEGATIVE_WORDS = {
        'lose', 'loss', 'defeat', 'injury', 'injured', 'doubt', 'poor',
        'bad', 'negative', 'weak', 'disappointing', 'struggle', 'miss',
        'unavailable', 'suspended', 'ban', 'problem', 'concern'
    }
    
    def __init__(self):
        """初始化情感分析器"""
        pass
        
    def analyze_text(self, text: str) -> Dict:
        """分析文本情感
        
        Args:
            text: 要分析的文本
            
        Returns:
            包含情感分析结果的字典
        """
        # 使用TextBlob进行基础情感分析
        blob = TextBlob(text)
        
        # 提取关键词
        words = set(re.findall(r'\w+', text.lower()))
        
        # 计算正面和负面词出现次数
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        # 综合评分
        # TextBlob的极性（-1到1）权重0.6
        # 关键词匹配（归一化后的-1到1）权重0.4
        textblob_score = blob.sentiment.polarity
        keyword_score = (positive_count - negative_count) / (positive_count + negative_count + 1)
        
        final_score = textblob_score * 0.6 + keyword_score * 0.4
        
        return {
            'score': final_score,  # 综合情感得分（-1到1）
            'subjectivity': blob.sentiment.subjectivity,  # 主观性得分（0到1）
            'positive_words': positive_count,  # 正面词数量
            'negative_words': negative_count,  # 负面词数量
            'confidence': self._calculate_confidence(blob.sentiment.subjectivity, 
                                                  positive_count + negative_count)
        }
        
    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """分析多篇文章的情感
        
        Args:
            articles: 文章列表，每篇文章应包含'title'和'description'字段
            
        Returns:
            添加了情感分析结果的文章列表
        """
        for article in articles:
            # 分析标题（权重0.7）和描述（权重0.3）
            title_sentiment = self.analyze_text(article['title'])
            desc_sentiment = self.analyze_text(article.get('description', ''))
            
            # 计算加权平均
            article['sentiment'] = {
                'score': title_sentiment['score'] * 0.7 + desc_sentiment['score'] * 0.3,
                'subjectivity': title_sentiment['subjectivity'] * 0.7 + desc_sentiment['subjectivity'] * 0.3,
                'confidence': (title_sentiment['confidence'] * 0.7 + desc_sentiment['confidence'] * 0.3),
                'title_sentiment': title_sentiment,
                'description_sentiment': desc_sentiment
            }
            
        return articles
        
    def _calculate_confidence(self, subjectivity: float, keyword_count: int) -> float:
        """计算情感分析的置信度
        
        Args:
            subjectivity: 主观性得分
            keyword_count: 关键词数量
            
        Returns:
            置信度得分（0到1）
        """
        # 主观性越低，置信度越高
        subjectivity_confidence = 1 - subjectivity
        
        # 关键词数量越多，置信度越高（最多到5个词）
        keyword_confidence = min(keyword_count / 5, 1.0)
        
        # 综合计算
        return (subjectivity_confidence * 0.7 + keyword_confidence * 0.3) 