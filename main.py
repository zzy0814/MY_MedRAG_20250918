# 导入必要的库
import os  # 用于操作系统相关功能，如文件和目录操作
import re  # 用于正则表达式操作
import json  # 用于JSON数据的编码和解码
import pandas as pd  # 用于数据处理和分析
from tqdm import tqdm  # 用于创建进度条
# 从main_MedRAG模块导入多个函数和类
from main_MedRAG import get_query_embedding, Faiss,  extract_diagnosis, documents, document_embeddings,generate_diagnosis_report, save_results_to_csv, get_additional_info_from_level_2,KG_preprocess, get_embeddings
# 从authentication模块导入路径变量
from authentication import ob_path,test_folder_path,ground_truth_file_path,augmented_features_path

# 定义疾病列表，包含各种疼痛相关疾病
disease_list = [
    "Head pain", "Migraine", "Trigeminal neuralgia", "Cervical spondylosis", "Chronic neck pain", "Neck pain",
    "Chest pain", "Abdominal pain", "Limb pain", "Shoulder pain", "Hip pain", "Knee pain", "Buttock pain",
    "Calf pain", "Low back pain", "Chronic low back pain", "Mechanical low back pain", "Upper back pain",
    "Degenerative disc disease", "Lumbar spondylosis", "Lumbar canal stenosis", "Spinal stenosis", "Foraminal stenosis",
    "Lumbar_radicular_pain", "Radicular pain", "Sciatica", "Lumbosacral pain", "Generalized body pain", "Fibromyalgia",
    "Musculoskeletal pain", "Myofascial pain syndrome", "Neuropathic pain", "Post-herpetic neuralgia"
]
# 读取真实诊断数据
ground_truth = pd.read_csv(ground_truth_file_path, header=0)

# 初始化结果列表
results = []
# 获取测试文件夹中的文件列表
file_paths = os.listdir(test_folder_path)

# 设置参数
topk=1  # 设置检索时返回的最相似文档数量
top_n=1  # 设置生成诊断报告时考虑的最相关信息数量
match_n=5  # 设置匹配的信息数量
samplerange=range(1,552)  # 设置采样范围

# 遍历采样范围内的每个样本
for i in tqdm(samplerange):

    print("topk:",topk)
    print("top_ns:",top_n)
    print("match_n:", match_n)
    print("i= ",i)
    # 构建文件路径
    file_path = os.path.join(test_folder_path, f"participant_{i}.json")
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f'{i} is not found')
        continue

    # 读取患者病例数据
    with open(file_path, 'r') as file:
        new_patient_case = json.load(file)
        print(new_patient_case)

    # 获取参与者编号
    participant_no = new_patient_case['Participant No.']
    # 将患者病例转换为JSON字符串格式
    query = json.dumps(new_patient_case)

    success = False
    # 使用循环直到处理成功
    while not success:
        try:
            # 获取查询嵌入向量
            query_embedding = get_query_embedding(query)
            # 使用Faiss进行相似性检索
            indices = Faiss(document_embeddings, query_embedding,k=topk)
            # 根据索引获取检索到的文档
            retrieved_documents = [documents[i] for i in indices[0]]
            final_retrieved_info =[]
            correct_count = 0
            # 处理每个检索到的文档
            for retrieved_document in retrieved_documents:
                with open(retrieved_document, 'r') as file:
                    patient_case = json.load(file)
                    patient_case_json = json.dumps(patient_case)
                    patient_case_dict = json.loads(patient_case_json)
                    # 过滤出需要的字段
                    filtered_patient_case_dict = {
                        key: patient_case_dict[key] for key in [
                            "Processed Diagnosis",
                            "Pain Presentation and Description Areas of pain as per physiotherapy input",
                            "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
                            "Pain/General Physiotherapist Treatments (Treatments\nSession No.: General Overview\n- Specific interventions/treatments)",
                            "Pain Psychologist Treatments (Treatments)",
                            "Pain Medicine Treatments (Treatments)",
                        ] if key in patient_case_dict
                    }
                    final_retrieved_info.append(filtered_patient_case_dict)

    # ——————————————————————————————————————————————————————————————————————————————————
            # 从真实诊断数据中查找当前患者的真实诊断
            true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == participant_no]
            # 检查是否找到真实诊断
            if true_diagnosis_row.empty:
                print(f"True diagnosis for patient_{participant_no} not found in ground truth data")
                break

            # 获取真实诊断信息
            true_diagnosis = true_diagnosis_row['Processed Diagnosis'].values[0]
            ori_truth = true_diagnosis_row['Diagnoses (related to pain)'].values[0]
            # 生成诊断报告
            generated_report_ori = generate_diagnosis_report(augmented_features_path,query, final_retrieved_info, i,top_n=top_n,match_n=match_n)
            print(generated_report_ori)

            # 从生成的报告中提取诊断信息
            generated_diagnosis = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report_ori)
            # 检查是否成功提取诊断信息
            if not generated_diagnosis:
                print("Generated diagnosis is either empty or not in the specified disease list. Retrying...")
                results.append([participant_no, '', true_diagnosis, ori_truth, generated_report_ori])
                break
            else:
                print("Success!!!")


            # 将结果添加到结果列表
            results.append([participant_no, generated_diagnosis[0], true_diagnosis, ori_truth,generated_report_ori])
            success = True
            print('________________________________________________________________')
        except Exception as e:
            # 捕获并处理异常
            print(f"Error processing patient_{participant_no}: {e}. ")

# 构建输出文件名
output_file = f"./test_results_topk{topk}_topn{top_n}_matchn{match_n}_{samplerange}_MedRAG.csv"

# 将结果转换为DataFrame并保存为CSV文件
df = pd.DataFrame(results, columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Ori Truth','Generated report'])
df.to_csv(output_file, index=False)
