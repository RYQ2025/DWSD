import requests
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import re

# ----------------------------------------------------------------------
#                         第一部分: 计算逻辑
# ----------------------------------------------------------------------

def get_standard_disagreement(output1, output2, task_id, vectorizer=None):
    s1, s2 = str(output1), str(output2)
    if task_id == "T1":
        distance = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return distance / max_len if max_len > 0 else 0
    else:
        if not s1 or not s2:
            return 0.0 if s1 == s2 else 1.0
        try:
            tfidf = vectorizer.transform([s1, s2])
            return 1 - cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except ValueError:
            return 1.0 if s1 != s2 else 0.0

def calculate_dwsd_score(task_id, outputs, accs, vectorizer):
    d_ab = get_standard_disagreement(outputs['A'], outputs['B'], task_id, vectorizer)
    d_ac = get_standard_disagreement(outputs['A'], outputs['C'], task_id, vectorizer)
    d_bc = get_standard_disagreement(outputs['B'], outputs['C'], task_id, vectorizer)

    id_a = (d_ab + d_ac) / 2
    id_b = (d_ab + d_bc) / 2
    id_c = (d_ac + d_bc) / 2

    acc_a, acc_b, acc_c = accs['A'][task_id], accs['B'][task_id], accs['C'][task_id]
    acc_total = acc_a + acc_b + acc_c
    if acc_total == 0:
        return 0.0, {'Llama': 0, 'Qwen14B': 0, 'Qwen32B': 0}

    w_a, w_b, w_c = acc_a / acc_total, acc_b / acc_total, acc_c / acc_total
    
    dwsd = w_a * w_b * d_ab + w_a * w_c * d_ac + w_b * w_c * d_bc
    weights = {'Llama': round(w_a, 2), 'Qwen14B': round(w_b, 2), 'Qwen32B': round(w_c, 2)}
    return dwsd, weights

# ----------------------------------------------------------------------
#                         第二部分: 组装Prompt
# ----------------------------------------------------------------------

def build_arbiter_prompt(row_index=19):
    try:
        file_llama = "llama3.1-8b_20.xlsx"
        file_qwen14b = "qwen2.5-14b_20.xlsx"
        file_qwen32b = "qwen2.5-32b_20.xlsx"
        
        df_llama = pd.read_excel(file_llama, header=None, engine='openpyxl')
        df_qwen14b = pd.read_excel(file_qwen14b, header=None, engine='openpyxl')
        df_qwen32b = pd.read_excel(file_qwen32b, header=None, engine='openpyxl')

        if row_index >= len(df_llama) or row_index >= len(df_qwen14b) or row_index >= len(df_qwen32b):
            print(f"错误: row_index {row_index} 超出数据范围。")
            return None

        row_llama = df_llama.iloc[row_index]
        row_qwen14b = df_qwen14b.iloc[row_index]
        row_qwen32b = df_qwen32b.iloc[row_index]
        
        task_column_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6}
        accuracies = {
            'A': {"T1": 0.500, "T2": 0.725, "T3": 0.725, "T4": 0.525, "T5": 0.500, "T6": 0.650},
            'B': {"T1": 0.475, "T2": 0.725, "T3": 0.900, "T4": 0.625, "T5": 0.700, "T6": 0.550},
            'C': {"T1": 0.600, "T2": 0.775, "T3": 0.875, "T4": 0.625, "T5": 0.700, "T6": 0.550}
        }
        
        task_info = {
            "T1": {"name": "数据集命名规范审核", "rule": "规则：必须严格遵循‘年度+地点+数据描述’的格式，不能包含冗余词语。"},
            "T2": {"name": "学科分类准确性核查", "rule": "规则：学科分类必须与内容的语义高度一致。"},
            "T3": {"name": "关键词准确性评估", "rule": "规则：必须是3到8个与核心内容紧密相关的术语。"},
            "T4": {"name": "描述一致性与完整性检查", "rule": "规则：必须逻辑通顺，信息完整，包含研究目的、方法、时空范围等。"},
            "T5": {"name": "数据来源类型识别", "rule": "规则：必须从预定义来源类型（如试验观测、调查分析、文献数据）中选择最恰当的一项。"},
            "T6": {"name": "数据类型验证", "rule": "规则：声明的数据类型必须与实际文件内容一致。"}
        }
        thresholds = {"T1": 0.15, "default": 0.30}

        print("正在构建全局TF-IDF词汇表...")
        corpus = []
        for col_idx in range(1, 7):
            corpus.append(str(row_llama.iloc[col_idx]))
            corpus.append(str(row_qwen14b.iloc[col_idx]))
            corpus.append(str(row_qwen32b.iloc[col_idx]))
        vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        print("全局词汇表构建完成。")
        
        prompt_payload_list = []
        for task_id, col_idx in task_column_map.items():
            outputs = {
                "A": str(row_llama.iloc[col_idx]),
                "B": str(row_qwen14b.iloc[col_idx]),
                "C": str(row_qwen32b.iloc[col_idx])
            }
            disagreement_score, weights = calculate_dwsd_score(task_id, outputs, accuracies, vectorizer)
            threshold = thresholds.get(task_id, thresholds["default"])
            
            if disagreement_score > threshold:
                level = "高分歧"
                instruction = "指令：检测到高分歧。请独立评估以下三个冲突建议，并生成一个最优的最终决定。在'justification'中，必须简要评判(critique)每个原始建议的优缺点，并阐述(rationale)你最终决策的形成逻辑。"
            else:
                level = "低分歧"
                instruction = "指令：检测到低分歧。请综合以下相似建议的优点，生成一个最完善的最终决定。在'justification'中，说明你主要采纳了哪个模型的建议，以及是否对其进行了优化。"

            prompt_payload_list.append({
                "task_id": task_id,
                "task_name": task_info[task_id]["name"],
                "audit_criteria": task_info[task_id]["rule"],
                "disagreement_score": round(disagreement_score, 4),
                "disagreement_level": level,
                "models": [
                    {"name": "Llama 3.1 8B", "output": outputs["A"], "weight": weights['Llama']},
                    {"name": "Qwen2.5 14B", "output": outputs["B"], "weight": weights['Qwen14B']},
                    {"name": "Qwen2.5 32B", "output": outputs["C"], "weight": weights['Qwen32B']}
                ],
                "final_instruction": instruction
            })
            
        return json.dumps(prompt_payload_list, ensure_ascii=False, indent=2)

    except FileNotFoundError as e:
        print(f"错误: 文件未找到。 {e}")
        return None
    except Exception as e:
        print(f"在构建prompt时发生错误: {e}")
        return None

# ----------------------------------------------------------------------
#                         第三部分: API调用逻辑
# ----------------------------------------------------------------------

def call_ollama_api(prompt_payload):
    ollama_api_url = "http://localhost:11434/api/generate"
    model_name = "deepseek-r1:32b-qwen-distill-q8_0"
#deepseek-r1:14b-qwen-distill-fp16 
#gemma3:27b-it-q8_0
#qwen2.5:72b-instruct-q4_0 
#deepseek-r1:32b-qwen-distill-q8_0
#run_arbiter_workflow_ok.py
#run_arbiter_workflow_ok.pyqwen3:32b-q8_0
    system_message = """
你是一位顶级的、严谨的农业科学元数据质量控制专家。你的工作核心是确保每一条元数据的质量。
你的核心原则是：1. **准确性** 2. **完整性** 3. **规范性** 4. **简洁性**。
你的任务是处理三个不同模型的输入，每个模型含有6个任务，T1-T6。其中每个对象代表一项独立的审核任务。对于数组中的每一项任务，你必须仔细阅读所有背景信息，不但分析每个任务的“output”，包括审核标准(audit_criteria)、各模型建议和分歧评估，并严格遵循最终指令(final_instruction)。
你的最终输出必须是一个JSON数组，每个对象对应一项任务的裁决结果。T1-T6，6个任务要处理完整，每个对象必须包含以下四个字段：
- 'taskId': 任务ID (例如T1)。
- 'final_decision': 你做出的最终裁决内容。
- 'status': 你的决策类型，必须是 '已修正' 或 '采纳最优' 中的一个。
- 'justification': 一个解释你决策的对象，必须包含 'critique' (对原始建议的简要评判) 和 'rationale' (你最终决策的形成理由) 两个子字段。
【重要指令】: 你的回答必须是从"T1": {"name": "数据集命名规范审核", "rule": "规则：必须严格遵循‘年度+地点+数据描述’的格式，不能包含冗余词语。"},
            "T2": {"name": "学科分类准确性核查", "rule": "规则：学科分类必须与内容的语义高度一致。"},
            "T3": {"name": "关键词准确性评估", "rule": "规则：必须是3到8个与核心内容紧密相关的术语。"},
            "T4": {"name": "描述一致性与完整性检查", "rule": "规则：必须逻辑通顺，信息完整，包含研究目的、方法、时空范围等。"},
            "T5": {"name": "数据来源类型识别", "rule": "规则：必须从预定义来源类型（如试验观测、调查分析、文献数据）中选择最恰当的一项。"},
            "T6": {"name": "数据类型验证", "rule": "规则：声明的数据类型必须与实际文件内容一致。"}，要分析完整，且仅是一个符合RFC 8259标准的JSON数组。不要包含任何解释、注释、思考过程或任何形式的额外文本。你的输出必须以 `[` 字符开始，以 `]` 字符结束。
"""

    data = {
        "model": model_name,
        "system": system_message,
        "prompt": prompt_payload,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    headers = {"Content-Type": "application/json"}

    print("\n----------- 正在向Ollama发送请求... -----------")
    print(f"裁决模型: {model_name}")
    
    try:
        response = requests.post(ollama_api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            raw_response_text = result.get("response", "")
            print("\n----------- 裁决大模型的响应 -----------")
            
            json_match = re.search(r'\[.*\]', raw_response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                try:
                    final_decision_json = json.loads(json_string)
                    print(json.dumps(final_decision_json, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(f"错误：提取的文本块不是有效的JSON。\n提取内容:\n{json_string}")
            else:
                print(f"错误：在模型响应中未能找到有效的JSON数组。\n原始响应:\n{raw_response_text}")
        else:
            print(f"\n请求失败，状态码: {response.status_code}\n错误信息: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"\n无法连接到 Ollama 服务器。请确保 Ollama 服务正在运行。")
    except Exception as e:
        print(f"\n调用API时发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    print("开始执行“三审一裁”工作流...")
    final_prompt = build_arbiter_prompt(row_index=2)  # 默认处理第5行，可修改

    if final_prompt:
        print("\n" + "="*50)
        print("      即将发送给裁决大模型的完整Prompt内容")
        print("="*50)
        prompt_object = json.loads(final_prompt)
        print(json.dumps(prompt_object, indent=2, ensure_ascii=False))
        print("="*50 + "\n")
        
        call_ollama_api(final_prompt)
    else:
        print("\n因错误导致工作流中断。")