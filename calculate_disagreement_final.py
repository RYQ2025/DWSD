# 单次加权系统总分歧度输出
##

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def levenshtein_distance(s1, s2):
    """手动实现的Levenshtein编辑距离算法"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_standard_disagreement(output1, output2, task_id, vectorizer):
    """根据任务类型，计算标准分歧度"""
    if task_id == "T1":  # 编辑距离
        distance = levenshtein_distance(output1, output2)
        max_len = max(len(output1), len(output2))
        return distance / max_len if max_len > 0 else 0
    else:  # 余弦相似度
        try:
            tfidf = vectorizer.fit_transform([str(output1), str(output2)])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return 1 - similarity
        except ValueError:
            return 1.0 if str(output1) != str(output2) else 0.0

def calculate_weighted_disagreement(task_id, outputs, accs, vectorizer):
    """计算最终的加权总分歧度"""
    sd_ab = get_standard_disagreement(outputs['A'], outputs['B'], task_id, vectorizer)
    sd_ac = get_standard_disagreement(outputs['A'], outputs['C'], task_id, vectorizer)
    sd_bc = get_standard_disagreement(outputs['B'], outputs['C'], task_id, vectorizer)

    id_a = (sd_ab + sd_ac) / 2
    id_b = (sd_ab + sd_bc) / 2
    id_c = (sd_ac + sd_bc) / 2

    acc_a = accs['A'][task_id]
    acc_b = accs['B'][task_id]
    acc_c = accs['C'][task_id]
    acc_total = acc_a + acc_b + acc_c

    if acc_total == 0: return 0.0

    w_a = acc_a / acc_total
    w_b = acc_b / acc_total
    w_c = acc_c / acc_total

    weighted_disagreement = (id_a * w_a) + (id_b * w_b) + (id_c * w_c)
    return weighted_disagreement

# --- 主程序 ---
try:
    # =================================================================
    # ==      请在这里确认您本地的 .xlsx 文件名，并按需修改          ==
    # =================================================================
    file_llama = "llama3.1-8b_20.xlsx"
    file_qwen14b = "qwen2.5-14b_20.xlsx"
    file_qwen32b = "qwen2.5-32b_20.xlsx"
    # =================================================================

    # 初始化TF-IDF向量化器
    vectorizer = TfidfVectorizer()

    # 读取Excel文件，假设数据在第一个工作表且没有标题行
    df_llama = pd.read_excel(file_llama, header=None, engine='openpyxl')
    df_qwen14b = pd.read_excel(file_qwen14b, header=None, engine='openpyxl')
    df_qwen32b = pd.read_excel(file_qwen32b, header=None, engine='openpyxl')

    # 提取第2行 (iloc[1])
    row_llama = df_llama.iloc[1]
    row_qwen14b = df_qwen14b.iloc[1]
    row_qwen32b = df_qwen32b.iloc[1]

    # 定义任务与列的映射（按列的索引，B列是1，C列是2，以此类推）
    task_column_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6}

    tasks_data = {}
    for task_id, col_idx in task_column_map.items():
        tasks_data[task_id] = {
            "A": str(row_llama.iloc[col_idx]),
            "B": str(row_qwen14b.iloc[col_idx]),
            "C": str(row_qwen32b.iloc[col_idx])
        }

    # 使用您提供的最新准确率数据
    accuracies = {
        'A': {"T1": 0.500, "T2": 0.725, "T3": 0.725, "T4": 0.525, "T5": 0.500, "T6": 0.650},
        'B': {"T1": 0.475, "T2": 0.725, "T3": 0.900, "T4": 0.625, "T5": 0.700, "T6": 0.550},
        'C': {"T1": 0.600, "T2": 0.775, "T3": 0.875, "T4": 0.625, "T5": 0.700, "T6": 0.550}
    }
    
    task_names = {
        "T1": "命名规范审核", "T2": "学科分类准确性核查", "T3": "关键词准确性评估",
        "T4": "描述一致性与完整性检查", "T5": "数据来源类型识别", "T6": "数据类型验证"
    }

    print("----------- 基于 .xlsx 文件第2行及最新准确率的分歧度计算结果 -----------")
    for task_id in tasks_data:
        print(f"\n--- 任务: {task_id} ({task_names[task_id]}) ---")
        print(f"  [模型A - Llama 3.1 8B] 输出: {tasks_data[task_id]['A']}")
        print(f"  [模型B - Qwen2.5 14B] 输出: {tasks_data[task_id]['B']}")
        print(f"  [模型C - Qwen2.5 32B] 输出: {tasks_data[task_id]['C']}")
        
        disagreement_score = calculate_weighted_disagreement(task_id, tasks_data[task_id], accuracies, vectorizer)
        
        print(f"  => 计算出的“加权系统总分歧度”为: {disagreement_score:.4f}")

except FileNotFoundError as e:
    print(f"\n错误: 文件未找到。 {e}")
    print("请确认您的Python脚本和三个 .xlsx 文件都在同一个文件夹下，并且代码中的文件名与实际文件名完全一致。")
except Exception as e:
    print(f"\n发生未知错误: {e}")