import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional

class FPLPreprocessor:
    """FPL数据预处理器"""
    def __init__(self):
        self.position_mapping = {
            'GKP': 'GK',
            'GK': 'GK',
            'DEF': 'DEF',
            'MID': 'MID',
            'FWD': 'FWD',
            'FOR': 'FWD'
        }
        self.required_columns = [
            'player_name',     # 球员名
            'pos',            # 位置
            'team',           # 球队
            'price',          # 价值
            'points',         # 总得分
            'minutes',        # 出场时间
            'goals_scored',   # 进球
            'assists',        # 助攻
            'clean_sheets',   # 零封
            'goals_conceded', # 丢球
            'season',         # 赛季
            'GW'             # 比赛周
        ]
    
    def preprocess(self, data, team_data=None):
        """预处理数据
        Args:
            data (pd.DataFrame): 原始数据
            team_data (pd.DataFrame, optional): 球队数据
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        try:
            # 检查数据列名
            print(f"Available columns: {data.columns.tolist()}")
            
            # 检查必要的列
            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                # 尝试映射列名
                column_mapping = {
                    'element': 'player_name',
                    'element_type': 'pos',
                    'round': 'GW',
                    'web_name': 'player_name',
                    'name': 'player_name',
                    'value': 'price',
                    'total_points': 'points'
                }
                
                # 重命名列
                data = data.rename(columns=column_mapping)
                
                # 再次检查必要的列
                missing_cols = [col for col in self.required_columns if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 复制数据以避免修改原始数据
            df = data.copy()
            
            # 标准化列名
            df = df.rename(columns={
                'GW': 'gw',
                'points': 'points',
                'price': 'price',
                'player_name': 'name',
                'pos': 'position'
            })
            
            # 标准化位置名称
            if 'position' in df.columns:
                df['position'] = df['position'].map(self.position_mapping).fillna(df['position'])
            
            # 处理缺失值
            df = self._handle_missing_values(df)
            
            # 标准化数值
            df = self._normalize_values(df)
            
            # 添加球队信息（如果提供）
            if team_data is not None:
                df = self._add_team_info(df, team_data)
            
            # 添加特征
            df = self._add_features(df)
            
            return df
            
        except Exception as e:
            print(f"❌ 数据预处理错误: {str(e)}")
            # 打印更多调试信息
            print(f"数据形状: {data.shape}")
            print(f"可用列: {data.columns.tolist()}")
            raise
    
    def _handle_missing_values(self, df):
        """处理缺失值"""
        # 数值型列用0填充
        numeric_cols = [
            'minutes', 'goals_scored', 'assists', 
            'clean_sheets', 'goals_conceded', 'points', 'price'
        ]
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # 分类列用众数填充
        categorical_cols = ['position', 'team']
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _normalize_values(self, df):
        """标准化数值"""
        # 价格转换为百万单位
        if 'price' in df.columns:
            df['price'] = df['price'] / 10
        
        # 确保所有数值列为float类型
        numeric_cols = [
            'minutes', 'goals_scored', 'assists', 
            'clean_sheets', 'goals_conceded', 'points', 'price'
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df
    
    def _add_team_info(self, df, team_data):
        """添加球队信息"""
        # 合并球队数据
        if 'team_id' in team_data.columns:
            df = df.merge(
                team_data[['team_id', 'name']],
                left_on='team',
                right_on='name',
                how='left'
            )
        return df
    
    def _add_features(self, df):
        """添加新特征"""
        # 计算每分钟得分
        df['points_per_minute'] = df.apply(
            lambda x: x['points'] / x['minutes'] if x['minutes'] > 0 else 0,
            axis=1
        )
        
        # 计算投资回报率
        df['roi'] = df['points'] / df['price']
        
        # 计算参与进球数
        df['goal_involvements'] = df['goals_scored'] + df['assists']
        
        return df
    
    def _standardize_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据类型"""
        type_mapping = {
            'value': 'float32',
            'total_points': 'int32',
            'minutes': 'int32',
            'goals_scored': 'int32',
            'assists': 'int32',
            'clean_sheets': 'int32',
            'goals_conceded': 'int32',
            'gw': 'int32'
        }
        
        for col, dtype in type_mapping.items():
            if col in data.columns:
                try:
                    data[col] = data[col].astype(dtype)
                except Exception as e:
                    print(f"⚠️ 列 {col} 类型转换警告: {str(e)}")
        
        return data
    
    def _process_gameweek(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理比赛周信息"""
        if 'gw' not in data.columns:
            if 'round' in data.columns:
                data['gw'] = data['round']
            elif 'gameweek' in data.columns:
                data['gw'] = data['gameweek']
        return data
    
    def _process_team(self, data: pd.DataFrame, team_list: Optional[pd.DataFrame]) -> pd.DataFrame:
        """处理球队信息（增强列识别能力）"""
        # 扩展球队列识别模式（包含不同赛季的列名变化）
        team_pattern = re.compile(r'team[_\-]?[xah]?', re.IGNORECASE)
        team_cols = [c for c in data.columns if team_pattern.search(c)]
        
        # 添加调试信息
        print("\n当前数据列：", data.columns.tolist())
        print("候选球队列：", team_cols)
        
        if not team_cols:
            # 尝试通过其他字段推断球队信息
            if 'was_home' in data.columns and 'opponent_team' in data.columns:
                print("通过主客场信息推断球队")
                data['team'] = np.where(data['was_home'], 
                                      data['team_h'].mode()[0], 
                                      data['opponent_team'])
            else:
                raise ValueError(f"无法识别球队列，最后尝试字段: {data.columns.tolist()[-5:]}")
        
        else:
            # 使用优先级排序选择最可能的球队列
            priority_order = ['team_x', 'team', 'team_h', 'team_a', 'team_id']
            team_col = next((c for c in priority_order if c in team_cols), team_cols[0])
            print(f"最终使用球队列: {team_col}")

            # 创建球队映射
            if team_list is not None:
                team_mapping = (
                    team_list.set_index('raw_name')['standard_name']
                    .str.strip().str.lower()
                    .to_dict()
                )
                
                # 统一球队名称（增强类型兼容性）
                data['team'] = (
                    data[team_col]
                    .astype(str).str.strip().str.lower()
                    .map(team_mapping)
                    .fillna(data[team_col])  # 保留原始值
                )
    
        return data
    
    def _process_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """终极位置处理方案"""
        # 扩展列名匹配模式
        position_patterns = [
            r'pos[ition]*', 
            r'element_?type',
            r'role',
            r'player_?type',
            r'position_id',
            r'^type$'
        ]
        
        position_cols = []
        for pattern in position_patterns:
            position_cols += data.columns[data.columns.str.contains(pattern, case=False)].tolist()
        
        position_cols = list(set(position_cols))  # 去重
        print("\n🔍 位置列识别阶段")
        print("候选列:", position_cols)
        print("前3行样例:", data[position_cols].head(3) if position_cols else "无候选列")

        if not position_cols:
            # 终极备用方案：通过数据内容推断
            print("⚠️ 进入备用位置识别模式")
            potential_cols = []
            for col in data.columns:
                unique_values = data[col].astype(str).str.upper().unique()
                if any(v in ['GK', 'DEF', 'MID', 'FWD'] for v in unique_values):
                    potential_cols.append(col)
            
            if potential_cols:
                print(f"通过内容识别到可能位置列: {potential_cols}")
                position_col = potential_cols[0]
            else:
                raise ValueError(
                    "无法识别位置列，请检查数据文件结构\n"
                    f"前5列样例:\n{data.iloc[:5,:5].to_markdown()}"
                )
        else:
            # 智能选择最优列
            priority_order = [
                'position', 'element_type', 'pos', 
                'player_type', 'role', 'type'
            ]
            position_col = next(
                (c for c in priority_order if c in position_cols),
                max(position_cols, key=lambda x: len(data[x].astype(str).unique()))
            )
        
        print(f"✅ 最终使用位置列: {position_col}")
        
        # 终极数据清洗
        data['position'] = (
            data[position_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r'[^A-Z]', '', regex=True)  # 移除非字母字符
            .replace({
                r'\bGK(P)?\b': 'GK',
                r'\bDEF(ENDER)?\b': 'DEF',
                r'\bMID(FIELDER)?\b': 'MID',
                r'\bFWD?(ARD)?\b': 'FWD',
                '1': 'GK', '2': 'DEF', '3': 'MID', '4': 'FWD'
            }, regex=True)
        )
        
        # 最终验证
        valid_positions = ['GK', 'DEF', 'MID', 'FWD']
        invalid = data[~data['position'].isin(valid_positions)]
        
        if not invalid.empty:
            print(f"⚠️ 发现非常规值: {invalid['position'].unique()}")
            print("尝试强制标准化...")
            data['position'] = data['position'].replace({
                r'.*GOAL.*': 'GK',
                r'.*DEFEN.*': 'DEF',
                r'.*MID.*': 'MID',
                r'.*FORW.*': 'FWD'
            }, regex=True)
        
        print("📊 最终位置分布:\n", data['position'].value_counts(dropna=False))
        return data
    
    def _process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数值特征"""
        # 基本特征
        numeric_cols = ['value', 'total_points', 'minutes', 'goals_scored', 
                       'assists', 'clean_sheets', 'goals_conceded']
        
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 填充缺失值
        data = data.fillna(0)
        
        return data

    def preprocess_team_data(self, raw_team: pd.DataFrame) -> pd.DataFrame:
        """预处理球队数据（适配当前数据格式）"""
        # 列名映射（根据实际CSV文件）
        column_mapping = {
            'team': 'raw_name',
            'team_name': 'standard_name'
        }
        
        # 重命名并生成必要列
        processed = raw_team.rename(columns=column_mapping)
        
        # 生成唯一ID（基于球队名称）
        processed['id'] = pd.factorize(processed['standard_name'])[0] + 1
        processed['short_name'] = processed['standard_name'].str[:3].str.upper()
        
        # 添加调试信息
        print("\n预处理后的球队数据列：", processed.columns.tolist())
        print("样例数据：\n", processed[['id', 'raw_name', 'standard_name', 'short_name']].head())
        
        return processed[['id', 'raw_name', 'standard_name', 'short_name']].drop_duplicates()

def build_synergy_matrix(players_df: pd.DataFrame) -> np.ndarray:
    """
    根据球员共同效力记录、国籍、俱乐部等信息生成协同矩阵
    players_df: 包含所有球员信息的 DataFrame
    返回: synergy_matrix, 形状 [num_players, num_players]
    """
    num_players = len(players_df)
    synergy_matrix = np.zeros((num_players, num_players), dtype=np.float32)
    
    for i in range(num_players):
        for j in range(num_players):
            if i == j:
                synergy_matrix[i, j] = 0.0
                continue
            
            # 示例：同队 +0.3，同国籍 +0.2，近几场共同首发X次 -> +X * 0.05
            same_team = (players_df.loc[i, 'team'] == players_df.loc[j, 'team'])
            if same_team:
                synergy_matrix[i, j] += 0.3
            
            same_nat = (players_df.loc[i, 'nationality'] == players_df.loc[j, 'nationality'])
            if same_nat:
                synergy_matrix[i, j] += 0.2
            
            synergy_matrix[i, j] += 0.05 * min(players_df.loc[i, 'shared_starts'][j], 5)
    
    # 归一化(可选)
    # synergy_matrix = synergy_matrix / synergy_matrix.max()
    return synergy_matrix