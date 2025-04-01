import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# 创建结果输出目录
results_dir = r'C:\Users\he\Desktop\AKI\results_my'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 加载数据
data = pd.read_excel('my_data/all.xlsx')

# 检查数据基本情况
print(f"数据集形状: {data.shape}")
print(f"列名: {data.columns.tolist()}")
data.info()

# 保存数据基本信息
with open(os.path.join(results_dir, 'data_info.txt'), 'w', encoding='utf-8') as f:
    f.write(f"数据集形状: {data.shape}\n")
    f.write(f"列名: {data.columns.tolist()}\n")
    f.write("\n数据描述统计:\n")
    f.write(data.describe().to_string())

# 检查缺失值情况
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100
missing_info = pd.DataFrame({
    '缺失值数量': missing_values,
    '缺失百分比': missing_percentage
}).sort_values('缺失百分比', ascending=False)
print(missing_info[missing_info['缺失值数量'] > 0])

# 保存缺失值信息
missing_info.to_csv(os.path.join(results_dir, 'missing_values_info.csv'), encoding='utf-8-sig')

# 可视化缺失值
plt.figure(figsize=(12, 8))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('缺失值可视化')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'missing_values_heatmap.png'), dpi=300)

# 抗生素暴露定义
# 注意：原代码假设数据包含抗生素使用的列，如 'antibiotics_type', 'antibiotics_duration'
# 但实际数据集中没有这些列，因此我们使用'abx'列作为替代
def categorize_antibiotics(row):
    """基于抗生素使用情况分类"""
    if pd.isna(row['abx']) or row['abx'] == 0:
        return 'no_antibiotics'
    else:
        return 'antibiotics_used'

# 添加抗生素分类列
data['antibiotics_category'] = data.apply(categorize_antibiotics, axis=1)

# KDIGO标准的AKI定义
def define_aki(row):
    """基于KDIGO标准定义AKI"""
    # 检查数据集中是否有必要的列
    required_columns = ['baseline_scr', 'max_scr']
    for col in required_columns:
        if col not in row.index:
            print(f"警告: 数据集中缺少必要的列 '{col}'")
            return np.nan
    
    baseline_scr = row['baseline_scr']
    max_scr = row['max_scr']
    
    if pd.isna(baseline_scr) or pd.isna(max_scr):
        return np.nan
    
    # AKI阶段1: SCr增加≥0.3mg/dL或增加到基线的1.5-1.9倍
    if (max_scr - baseline_scr >= 0.3) or (max_scr / baseline_scr >= 1.5 and max_scr / baseline_scr < 2.0):
        return 1
    # AKI阶段2: SCr增加到基线的2.0-2.9倍
    elif max_scr / baseline_scr >= 2.0 and max_scr / baseline_scr < 3.0:
        return 2
    # AKI阶段3: SCr增加到基线的≥3.0倍或SCr≥4.0mg/dL或开始肾脏替代治疗
    elif max_scr / baseline_scr >= 3.0 or max_scr >= 4.0 or (('rrt' in row.index) and row['rrt'] == 1):
        return 3
    else:
        return 0  # 无AKI

data['aki_stage'] = data.apply(define_aki, axis=1)
data['aki'] = (data['aki_stage'] > 0).astype(int)  # 二分类AKI变量

# 保存AKI分布情况
aki_distribution = data['aki_stage'].value_counts().sort_index()
aki_distribution.to_csv(os.path.join(results_dir, 'aki_stage_distribution.csv'))

# 可视化AKI分布
plt.figure(figsize=(10, 6))
sns.countplot(x='aki_stage', data=data)
plt.title('AKI阶段分布')
plt.xlabel('AKI阶段')
plt.ylabel('患者数量')
plt.savefig(os.path.join(results_dir, 'aki_stage_distribution.png'), dpi=300)

# 检查数据集中的日期列
print("检查数据集中的日期列...")
date_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
print(f"找到可能的日期列: {date_columns}")

# 尝试将这些列转换为日期格式
for col in date_columns:
    try:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        print(f"成功将列 '{col}' 转换为日期格式")
    except Exception as e:
        print(f"无法将列 '{col}' 转换为日期格式: {e}")

# 创建时间相关特征
print("尝试创建时间相关特征...")
# 检查是否有必要的日期列来计算时间特征
required_date_columns = ['icu_admission_time', 'antibiotics_start_time']
if all(col in data.columns for col in required_date_columns):
    try:
        data['time_to_antibiotics'] = (data['antibiotics_start_time'] - data['icu_admission_time']).dt.total_seconds() / 3600  # 小时
        print("成功创建 'time_to_antibiotics' 特征")
        
        # 抗生素使用是否在AKI之前(排除AKI由其他原因导致的可能)
        if 'aki_detection_time' in data.columns:
            data['antibiotics_before_aki'] = (data['antibiotics_start_time'] < data['aki_detection_time']).astype(int)
            # 抗生素暴露到AKI发生的时间窗口
            data['hours_antibiotics_to_aki'] = (data['aki_detection_time'] - data['antibiotics_start_time']).dt.total_seconds() / 3600
            print("成功创建 'antibiotics_before_aki' 和 'hours_antibiotics_to_aki' 特征")
    except Exception as e:
        print(f"创建时间特征时出错: {e}")
else:
    print(f"警告: 缺少创建时间特征所需的列: {[col for col in required_date_columns if col not in data.columns]}")

# 保存处理后的数据
processed_data_path = os.path.join(results_dir, 'processed_data.csv')
data.to_csv(processed_data_path, index=False, encoding='utf-8-sig')
print(f"处理后的数据已保存至: {processed_data_path}")

# 使用causalml进行因果推断分析
from causalml.inference.meta import (
    BaseXRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
)
from causalml.match import NearestNeighborMatch, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.metrics import plot_gain, get_cumgain
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 准备因果推断所需的数据
# 选择特征、处理变量和结果变量
features = [col for col in data.columns if col not in ['aki', 'aki_stage', 'antibiotics_category', 
                                                      'icu_admission_time', 'antibiotics_start_time', 
                                                      'aki_detection_time']]
treatment = 'antibiotics_category'  # 处理变量
outcome = 'aki'  # 结果变量

# 处理缺失值
imputer = KNNImputer(n_neighbors=5)
X = data[features].copy()
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 标准化特征
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# 保存特征重要性
X_scaled.to_csv(os.path.join(results_dir, 'scaled_features.csv'), index=False)

# 计算倾向得分
p_model = ElasticNetPropensityModel()
# 将分类变量转换为二分类处理变量（有抗生素 vs 无抗生素）
data['treatment'] = (data[treatment] != 'no_antibiotics').astype(int)
p_score = p_model.fit_predict(X=X_scaled, y=data['treatment'])
data['propensity_score'] = p_score

# 进行匹配
matcher = NearestNeighborMatch(caliper=0.2, replace=False)
matched_data = matcher.match(
    data=data,
    treatment_col='treatment',
    score_cols=['propensity_score']
)

# 保存匹配后的数据
matched_data.to_csv(os.path.join(results_dir, 'matched_data.csv'), index=False)

# 检查匹配质量
balance_table = create_table_one(matched_data, 'treatment', features)
balance_table.to_csv(os.path.join(results_dir, 'balance_table.csv'))

# 使用不同的元学习器估计因果效应
learners = {
    'S-Learner (LR)': BaseSRegressor(LogisticRegression()),
    'T-Learner (LR)': BaseTRegressor(LogisticRegression()),
    'X-Learner (RF)': BaseXRegressor(RandomForestClassifier()),
    'R-Learner (GB)': BaseRRegressor(GradientBoostingClassifier())
}

# 评估不同学习器的性能
results = {}
for name, learner in learners.items():
    try:
        te_pred = learner.fit_predict(
            X=matched_data[features], 
            treatment=matched_data['treatment'], 
            y=matched_data[outcome],
            p=matched_data['propensity_score']
        )
    except TypeError:
        te_pred = learner.fit_predict(
            X=matched_data[features], 
            treatment=matched_data['treatment'], 
            y=matched_data[outcome]
        )
    
    results[name] = te_pred
    
    # 保存每个学习器的结果
    pd.DataFrame({
        'treatment_effect': te_pred
    }).to_csv(os.path.join(results_dir, f'{name.replace(" ", "_")}_results.csv'), index=False)
    
    # 计算平均处理效应
    ate = te_pred.mean()
    print(f"{name} 平均处理效应 (ATE): {ate}")
    
    # 保存ATE结果
    with open(os.path.join(results_dir, 'ate_results.txt'), 'a') as f:
        f.write(f"{name} 平均处理效应 (ATE): {ate}\n")

# 可视化累积增益
for name, te in results.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_gain(
        pd.DataFrame({
            'y': matched_data[outcome],
            'w': matched_data['treatment'],
            'treatment_effect': te
        }),
        outcome_col='y',
        treatment_col='w',
        treatment_effect_col='treatment_effect',
        ax=ax
    )
    plt.title(f'{name} 累积增益')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{name.replace(" ", "_")}_gain_curve.png'), dpi=300)
    plt.close()

# 抗生素类型与AKI的关系分析
if 'antibiotics_type' in data.columns:
    antibiotics_aki = pd.crosstab(data['antibiotics_type'], data['aki'])
    antibiotics_aki['aki_rate'] = antibiotics_aki[1] / (antibiotics_aki[0] + antibiotics_aki[1])
    antibiotics_aki.to_csv(os.path.join(results_dir, 'antibiotics_type_aki_rate.csv'))
    
    # 可视化不同抗生素类型的AKI发生率
    plt.figure(figsize=(12, 8))
    antibiotics_aki['aki_rate'].sort_values().plot(kind='bar')
    plt.title('不同抗生素类型的AKI发生率')
    plt.ylabel('AKI发生率')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'antibiotics_type_aki_rate.png'), dpi=300)

print(f"所有结果已保存至: {results_dir}")


