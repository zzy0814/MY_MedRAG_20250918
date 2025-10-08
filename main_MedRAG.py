# 导入必要的库
import openai  # OpenAI API调用库，用于调用OpenAI的模型接口
import faiss  # 高效相似性搜索库，用于快速向量检索
import numpy as np  # 数值计算库，用于处理数组和矩阵
import os  # 操作系统接口库，用于文件和目录操作
import re  # 正则表达式操作库，用于文本模式匹配
import json  # JSON数据解析库，用于处理JSON格式数据
import pandas as pd  # 数据处理和分析库，用于表格数据处理
from tqdm import tqdm  # 进度条显示库，用于显示任务进度
from huggingface_hub import InferenceClient  # Hugging Face模型推理客户端，用于调用开源模型
from KG_Retrieve import main_get_category_and_level3  # 知识图谱检索模块，用于获取类别信息
from authentication import api_key,hf_token  # API密钥和令牌，用于身份验证

# 初始化OpenAI客户端
client = openai.OpenAI(api_key=api_key)

def get_embeddings(texts):
    """
    获取文本的嵌入向量

    
    该函数使用指定的模型将输入的文本列表转换为嵌入向量数组。
    使用tqdm库显示进度条，便于处理大量文本时了解进度。
    :param texts: 文本列表，可以是字符串或字符串列表

    :type texts: list or str
    :return: 文本的嵌入向量数组，形状为(len(texts), embedding_dim)
    :rtype: numpy.ndarray
    示例:
        >>> texts = ["Hello world", "How are you"]
        >>> embeddings = get_embeddings(texts)
        >>> print(embeddings.shape)  # 输出: (2, 3072)
    """
    # 初始化空列表用于存储嵌入向量
    embeddings = []
    # 遍历输入的文本列表，使用tqdm显示进度条
    for text in tqdm(texts):
        # 调用API获取文本的嵌入向量
        response = client.embeddings.create(
            input=text,           # 输入文本
            model="text-embedding-3-large"  # 使用的嵌入模型
        )
        # 将获取的嵌入向量添加到列表中
        # response.data[0].embedding 包含文本的嵌入向量
        embeddings.append(response.data[0].embedding)
    # 将列表转换为NumPy数组并返回
    return np.array(embeddings)


def get_query_embedding(query):
    """
    获取查询文本的嵌入向量

    
    该函数接收一个查询文本，通过调用get_embeddings函数获取其嵌入向量，
    并返回结果列表中的第一个元素。这通常用于文本相似度计算或
    自然语言处理任务中的文本表示。
    :param query: 查询文本，可以是任意字符串形式的输入
    :return: 查询文本的嵌入向量，通常是一个数值列表或数组
    """
    return get_embeddings([query])[0]  # 将查询文本放入列表并调用get_embeddings，返回结果中的第一个元素


# FAISS
def Faiss(document_embeddings, query_embedding, k):
    # index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    # index = faiss.IndexHNSWFlat(document_embeddings.shape[1])
    index.add(document_embeddings)
    _, indices = index.search(np.array([query_embedding]), k)
    print("index: ", indices)
    return indices

def extract_diagnosis(generated_text):
    diagnoses = re.findall(r'\*\*Diagnosis\*\*:\s(.*?)\n', generated_text)
    return diagnoses

import re  # 导入re模块，用于正则表达式操作
def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()

def KG_preprocess(file_path):
    kg_data = pd.read_excel(file_path, usecols=['subject', 'relation', 'object'])
    kg_data['subject'] = kg_data['subject'].apply(remove_parentheses)
    kg_data['object'] = kg_data['object'].apply(remove_parentheses)

    knowledge_graph = {}
    for index, row in kg_data.iterrows():
        subject = row['subject']
        relation = row['relation']
        obj = row['object']

        if subject not in knowledge_graph:
            knowledge_graph[subject] = []
        knowledge_graph[subject].append((relation, obj))

        if obj not in knowledge_graph:
            knowledge_graph[obj] = []
        knowledge_graph[obj].append((relation, subject))
    return knowledge_graph


def extract_features_from_json(file_path):
    with open(file_path, 'r') as file:
        patient_case = json.load(file)

    pain_location = patient_case.get("Pain Presentation and Description Areas of pain as per physiotherapy input", "")
    pain_symptoms = patient_case.get(
        "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
        "")

    return pain_location, pain_symptoms

level_3_to_level_2 = {
    # Here are subcategories: diseases
    # Examples: 
    
    # Respiratory System
    "acute_copd_exacerbation_infection": "respiratory_system",

    # Cardiovascular System
    "atrial_fibrillation": "cardiovascular_system",

}


def get_additional_info_from_level_2(participant_no,  kg_path,top_n,match_n):
    level_2_values=main_get_category_and_level3(match_n,participant_no,top_n)
    additional_info = []
    if not level_2_values:
        print(f"No data found for Participant No.: {participant_no}")
        return None
    for level_2_value in level_2_values:
        relevant_level_3_descriptions = [desc for desc, level2 in level_3_to_level_2.items() if level2 == level_2_value]
        print("Relevant Level 3 Descriptions:", relevant_level_3_descriptions)
        if not relevant_level_3_descriptions:
            print("No Level 3 descriptions found for Level 2:", level_2_value)
            continue

        kg_data = pd.read_excel(kg_path, usecols=['subject', 'relation', 'object'])
        if kg_data.empty:
            print("Knowledge graph data is empty.")
            return None

        merged_info = {}

        for level_3 in relevant_level_3_descriptions:
            related_info = kg_data[kg_data['subject'] == level_3]

            if related_info.empty:
                print(f"No related information found in KG for: {level_3}")
            else:
                for _, row in related_info.iterrows():
                    subject = row['subject']
                    relation = row['relation'].replace('_', ' ')
                    obj = row['object']

                    if (subject, relation) in merged_info:
                        merged_info[(subject, relation)].append(obj)
                    else:
                        merged_info[(subject, relation)] = [obj]

        # K
        additional_info = []
        for (subject, relation), objects in merged_info.items():
            sentence = f"{subject} {relation} {', '.join(objects)}"
            additional_info.append(sentence)

    if not additional_info:
        print("No additional information found.")
        return None

    final_info = ', '.join(additional_info)
    print("Additional Info:", final_info)
    return final_info


def get_system_prompt_for_RAGKG():
    """
    返回一个用于RAG（检索增强生成）知识图谱的系统提示，该提示指导AI作为医学助手进行疼痛管理。
    返回:
        str: 包含详细指导的系统提示文本，规定了AI在诊断、治疗和建议方面的行为准则。
    """
    return '''
        # 角色定义
        You are a knowledgeable medical assistant with expertise in pain management.
        # 任务描述
        Your tasks are:
        1. Analyse and refer to the retrieved similar patients' cases and knowledge graph which may be relevant to the diagnosis and assist with new patient cases.
2. Output of "Diagnoses" must come from : acute copd exacerbation infection, bronchiectasis, bronchiolitis, bronchitis, bronchospasm acute asthma exacerbation, pulmonary embolism, pulmonary neoplasm, spontaneous pneumothorax, urti, viral pharyngitis, whooping cough, acute laryngitis, acute pulmonary edema, croup, larygospasm, epiglottitis, pneumonia, atrial fibrillation, myocarditis, pericarditis, psvt, possible nstemi stemi, stable angina, unstable angina, gerd, boerhaave syndrome, pancreatic neoplasm, scombroid food poisoning, inguinal hernia, tuberculosis, hiv initial infection, ebola, influenza, chagas, acute otitis media, acute rhinosinusitis, allergic sinusitis, chronic rhinosinusitis, myasthenia gravis, guillain barre syndrome, cluster headache, acute dystonic reactions, sle, sarcoidosis, anaphylaxis, panic attack, spontaneous rib fracture, anemia.        3. You are given differences of diagnoses of similar symptoms or pain locations. Read that information as a reference to your diagnostic if applicable.
        4. Do mind the nuance between these factors of similar diagnosis with knowledge graph information and consider it when diagnose new patient's informtation.
        5. Ensure that the recommendations are evidence-based and consider the most recent and effective practices in pain management.
        6. The output should include four specific treatment-related fields:
           - "Diagnoses (related to pain)"
           - Explanations of diagnose
           - "Pain/General Physiotherapist Treatments\nSession No.: General Overview\n- Specific interventions/treatments"
           - "Pain Psychologist Treatments"
           - "Pain Medicine Treatments"
        7. In "Diagnoses", only output the diagnosis itself. Place all other explanations and analyses (if any) into "Explanations of diagnose".
        8. You can leave Psychologist Treatments blank if not applicable for the case, leaving text "Not applicable"
        9.If you think information is needed, guide the doctor to ask further questions which following areas to distinguish between the most likely diseases: Pain restriction; Location; Symptom. Seperate answers with ",". The output should only include aspects.
        10. The output should follow this structured format:
        

    ### Diagnoses
    1. **Diagnosis**: Answer.
    2. **Explanations of diagnose**: Answer.
    
    ### Instructive question
    1. **Questions**: Answer.
    
    ### Pain/General Physiotherapist Treatments
    1. **Session No.: General Overview**
        - **Specific interventions/treatments**:
        - **Goals**:
        - **Exercises**:
        - **Manual Therapy**:
        - **Techniques**:

    2. **Exercise Recommendations from the Exercise List**:

    ### Pain Psychologist Treatments(if applicable)
    1. **Treatment 1**: 
    
    ### Pain Medicine Treatments


    ### Recommendations for Further Evaluations
    1. **Evaluation 1**:
    '''


def generate_diagnosis_report(path, query, retrieved_documents, i,top_n,match_n,model):
    system_prompt_RAGKG = get_system_prompt_for_RAGKG()
    system_prompt=system_prompt_RAGKG
    additional_info= get_additional_info_from_level_2(i ,path,top_n=top_n,match_n=match_n)

    prompt = f"{query}\nRetrieved Documents: {retrieved_documents}\nInformation from knowledge graph about relevant diagnoses, if you think the patient's disease is relevant from the suggestions provided by the atlas please refer to thoses details to distinguish similar diagnoses : {additional_info} .Now complete the tasks in that format"


    ############################################################################################openai
    if model =='gpt-4o' or 'gpt-4o-mini' or 'gpt-3.5-turbo-0125':
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        prompt=f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"""
        LLMclient = InferenceClient(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Meta-Llama-3.1-70B-Instruct",
            # "meta-llama/Llama-2-13b-hf",
            # "Qwen/Qwen2-7B-Instruct",
            # "Qwen/Qwen2.5-0.5B-Instruct",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            token=hf_token
        )
        response = LLMclient.text_generation(prompt=prompt,max_new_tokens=400)
        return response

def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results,
                      columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Original Diagnosis'])
    df.to_csv(output_file, index=False)


folder_path=".dataset/df/train"
documents = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file_name))]

document_embeddings_file_path='./dataset/document_embeddings.npy'

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    return np.load(file_path)
if os.path.exists(document_embeddings_file_path):
    document_embeddings = load_embeddings(document_embeddings_file_path)
else:
    document_embeddings = get_embeddings(documents)
    save_embeddings(document_embeddings, document_embeddings_file_path)
