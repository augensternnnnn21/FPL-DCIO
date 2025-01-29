def check_formation_constraints(selected_players, constraints):
    """检查阵容约束
    Args:
        selected_players (pd.DataFrame): 选中的球员
        constraints (dict): 位置约束
    Returns:
        tuple: (是否满足约束, 违规数)
    """
    violations = 0
    position_counts = selected_players['position'].value_counts()
    
    for pos, (min_count, max_count) in constraints.items():
        count = position_counts.get(pos, 0)
        if count < min_count:
            violations += min_count - count
        elif count > max_count:
            violations += count - max_count
    
    return violations == 0, violations

def check_budget_constraint(selected_players, budget):
    """检查预算约束"""
    total_value = selected_players['price'].sum()
    return total_value <= budget, total_value 