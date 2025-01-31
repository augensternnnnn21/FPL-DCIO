import pandas as pd
import sys
from pathlib import Path
import glob

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def safe_read_csv(file_path: str) -> pd.DataFrame:
    """安全读取CSV文件，尝试不同的编码"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时出错 ({encoding}): {str(e)}")
            continue
            
    raise ValueError(f"无法读取文件: {file_path}")

def get_team_mapping(season_path: Path) -> dict:
    """获取球队ID和名称的映射"""
    # 读取master_team_list.csv
    master_list = safe_read_csv(project_root / "data" / "master_team_list.csv")
    season = season_path.name  # 获取赛季名称
    
    # 获取当前赛季的球队映射
    season_teams = master_list[master_list['season'] == season]
    
    # 创建ID到名称的映射
    team_mapping = dict(zip(season_teams['team'], season_teams['team_name']))
    
    # 创建名称到ID的映射
    name_to_id = {}
    for _, row in season_teams.iterrows():
        team_id = row['team']
        team_name = row['team_name']
        
        # 添加原始名称
        name_to_id[team_name] = team_id
        
        # 添加一些常见的别名
        if team_name == "Brighton":
            name_to_id.update({
                "Brighton & Hove Albion": team_id,
                "Brighton and Hove Albion": team_id
            })
        elif team_name == "Man City":
            name_to_id["Manchester City"] = team_id
        elif team_name == "Man Utd":
            name_to_id.update({
                "Manchester Utd": team_id,
                "Manchester United": team_id
            })
        elif team_name == "Newcastle":
            name_to_id["Newcastle United"] = team_id
        elif team_name == "Norwich":
            name_to_id["Norwich City"] = team_id
        elif team_name == "Sheffield Utd":
            name_to_id["Sheffield United"] = team_id
        elif team_name == "Spurs":
            name_to_id.update({
                "Tottenham": team_id,
                "Tottenham Hotspur": team_id
            })
        elif team_name == "West Ham":
            name_to_id["West Ham United"] = team_id
        elif team_name == "Wolves":
            name_to_id.update({
                "Wolverhampton": team_id,
                "Wolverhampton Wanderers": team_id
            })
        elif team_name == "Leeds":
            name_to_id["Leeds United"] = team_id
        elif team_name == "Nott'm Forest":
            name_to_id.update({
                "Nottingham Forest": team_id,
                "Nottingham": team_id,
                "Forest": team_id
            })
        elif team_name == "Luton":
            name_to_id["Luton Town"] = team_id
        elif team_name == "West Brom":
            name_to_id.update({
                "West Bromwich": team_id,
                "West Bromwich Albion": team_id
            })
        elif team_name == "Cardiff":
            name_to_id["Cardiff City"] = team_id
        elif team_name == "Huddersfield":
            name_to_id["Huddersfield Town"] = team_id
        elif team_name == "Leicester":
            name_to_id["Leicester City"] = team_id
        elif team_name == "Hull":
            name_to_id["Hull City"] = team_id
        elif team_name == "Stoke":
            name_to_id["Stoke City"] = team_id
        elif team_name == "Swansea":
            name_to_id["Swansea City"] = team_id
            
    return team_mapping, name_to_id

def extract_fixtures_from_gw(gw_df: pd.DataFrame, gw: int, name_to_id: dict) -> list:
    """从一轮比赛数据中提取比赛信息"""
    fixtures = []
    processed_fixtures = set()
    
    # 检查是否是早期赛季的数据格式
    is_early_season = 'element' in gw_df.columns and 'team' not in gw_df.columns
    
    for _, row in gw_df.iterrows():
        fixture_id = row['fixture']
        
        # 避免重复处理同一场比赛
        if fixture_id in processed_fixtures:
            continue
            
        processed_fixtures.add(fixture_id)
        
        # 只处理主场比赛记录
        if row['was_home']:
            if is_early_season:
                # 早期赛季使用element作为team_id
                team = row['element']
                opponent_team = row['opponent_team']
            else:
                # 获取球队ID
                team_raw = row['team'] if 'team' in row else None
                opponent_team_raw = row['opponent_team'] if 'opponent_team' in row else None
                
                # 如果是字符串（球队名称），尝试转换为ID
                if isinstance(team_raw, str) and team_raw in name_to_id:
                    team = name_to_id[team_raw]
                else:
                    team = team_raw
                    
                if isinstance(opponent_team_raw, str) and opponent_team_raw in name_to_id:
                    opponent_team = name_to_id[opponent_team_raw]
                else:
                    opponent_team = opponent_team_raw
            
            if team is not None and opponent_team is not None:
                fixtures.append({
                    'id': fixture_id,
                    'gameweek': gw,
                    'kickoff_time': row['kickoff_time'],
                    'team_h': team,
                    'team_a': opponent_team,
                    'team_h_score': row['team_h_score'],
                    'team_a_score': row['team_a_score']
                })
            
    return fixtures

def rebuild_fixtures(season: str):
    """从gws目录重建fixtures数据
    
    Args:
        season: 赛季，格式如 "2016-17"
    """
    print(f"重建 {season} 赛季的fixtures数据...")
    
    # 读取所有gw文件
    season_path = project_root / "data" / season
    gw_files = sorted(glob.glob(str(season_path / "gws" / "gw*.csv")))
    
    if not gw_files:
        print(f"找不到赛季 {season} 的gw文件")
        return
        
    # 获取球队映射
    team_mapping, name_to_id = get_team_mapping(season_path)
    
    # 收集所有比赛数据
    all_fixtures = []
    for gw_file in gw_files:
        try:
            gw_df = safe_read_csv(gw_file)
            gw = int(Path(gw_file).stem[2:])  # 从文件名提取轮次
            fixtures = extract_fixtures_from_gw(gw_df, gw, name_to_id)
            all_fixtures.extend(fixtures)
            print(f"已读取 {Path(gw_file).name} ({len(fixtures)} 场比赛)")
        except Exception as e:
            print(f"处理文件 {gw_file} 时出错: {str(e)}")
            continue
            
    if not all_fixtures:
        print("没有成功读取任何比赛数据")
        return
        
    # 转换为DataFrame
    fixtures_df = pd.DataFrame(all_fixtures)
    
    # 创建teams数据
    teams = set()
    for _, row in fixtures_df.iterrows():
        # 确保team_id是整数类型
        team_h = int(row['team_h']) if isinstance(row['team_h'], str) else row['team_h']
        team_a = int(row['team_a']) if isinstance(row['team_a'], str) else row['team_a']
        teams.add(team_h)
        teams.add(team_a)
        
    teams_data = []
    for team_id in sorted(teams):
        teams_data.append({
            'team_id': team_id,
            'name': team_mapping.get(team_id, f'Unknown Team {team_id}')
        })
        
    teams_df = pd.DataFrame(teams_data)
    
    # 保存数据
    fixtures_df.to_csv(season_path / "fixtures.csv", index=False)
    print(f"已保存 {len(fixtures_df)} 场比赛的数据")
    
    teams_df.to_csv(season_path / "teams.csv", index=False)
    print(f"已保存 {len(teams_df)} 支球队的数据")

def main():
    """主函数"""
    # 处理所有赛季
    seasons = [
        "2016-17", "2017-18", "2018-19",
        "2019-20", "2020-21", "2021-22",
        "2022-23", "2023-24"
    ]
    
    for season in seasons:
        rebuild_fixtures(season)

if __name__ == "__main__":
    main() 