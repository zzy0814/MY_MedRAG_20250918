
# 导入必要的库
import openai  # OpenAI API接口
import pandas as pd  # 数据处理
import re  # 正则表达式
import numpy as np  # 数值计算
from tqdm import tqdm  # 进度条
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
import networkx as nx  # 图论分析
import matplotlib.pyplot as plt  # 数据可视化
import nltk  # 自然语言处理
from nltk.tokenize import word_tokenize  # 分词
from nltk.corpus import stopwords  # 停用词
import string  # 字符串处理
import os  # 操作系统接口
from collections import defaultdict  # 默认字典
from authentication import DEEPSEEK_API_KEY,DEEPSEEK_URL


# 下载NLTK所需资源
nltk.download('punkt')
nltk.download('stopwords')

# 设置OpenAI API密钥
#api_key = ''
api_key = DEEPSEEK_API_KEY
#初始化deepseek
#client = openai.OpenAI(api_key=api_key)
client =openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_URL)



# 定义文件路径
KG_file_path = './dataset/knowledge graph of chronic pain.xlsx'  # 知识图谱数据路径
file_path = './dataset/AI Data Set with Categories.csv'  # 数据集路径
embedding_save_path = './Embeddings_saved/CP_KG_embeddings'  # 词向量保存路径



def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = text.replace('_', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)


kg_data = pd.read_excel(KG_file_path, usecols=['subject', 'relation', 'object'])

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

kg_data['object_preprocessed'] = kg_data.apply(
    lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
    axis=1
)
symptom_nodes = kg_data['object_preprocessed'].dropna().unique().tolist()


def get_symptom_embeddings(symptom_nodes, save_path):
    embeddings_path = os.path.join(save_path, 'KG_embeddings.npy')
    if os.path.exists(embeddings_path):
        print("load existing embeddings...")
        return np.load(embeddings_path)
    else:
        print("generate new embeddings...")
        symptom_embeddings = []
        for symptom in tqdm(symptom_nodes):
            response = client.embeddings.create(
                input=symptom,
                #model="text-embedding-3-large"
                model="deepseek-embed"
            )
            symptom_embeddings.append(response.data[0].embedding)
        np.save(embeddings_path, symptom_embeddings)

        return np.array(symptom_embeddings)


symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path)


def find_top_n_similar_symptoms(query, symptom_nodes, symptom_embeddings, n):
    if pd.isna(query) or not query:
        return []
    query_preprocessed = preprocess_text(query)
    response = client.embeddings.create(
        input=query_preprocessed,
        model="text-embedding-3-large"
    )
    query_embedding = response.data[0].embedding
    if not query_embedding:
        return []

    if len(symptom_embeddings) > len(symptom_nodes):
        symptom_embeddings = symptom_embeddings[:len(symptom_nodes)]

    similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

    top_n_symptoms = []
    unique_symptoms = set()
    top_n_indices = similarities.argsort()[::-1]

    for i in top_n_indices:
        if similarities[i] > 0.5 and symptom_nodes[i] not in unique_symptoms:
            top_n_symptoms.append(symptom_nodes[i])
            unique_symptoms.add(symptom_nodes[i])
        if len(top_n_symptoms) == n:
            break

    return top_n_symptoms


def compute_shortest_path_length(node1, node2, G):
    try:
        return nx.shortest_path_length(G, source=node1, target=node2)
    except nx.NetworkXNoPath:
        return float('inf')

categories = [
    "thoracoabdominal_pain_syndromes",
    "neuropathic_pain_syndromes",
    "craniofacial_pain_syndromes",
    "cervical_spine_pain_syndromes",
    "limb_and_joint_pain_syndromes",
    "back_pain_syndromes",
    "lumbar_degenerative_and_stenosis_and_radicular_and_sciatic_syndromes",
    "generalized_pain_syndromes",

]
G = nx.Graph()
for node, edges in knowledge_graph.items():
    for relation, neighbor in edges:
        G.add_edge(node, neighbor, relation=relation)


def get_diagnoses_for_symptom(symptom):

    diagnoses = []
    if symptom in G:
        for neighbor in G.neighbors(symptom):
            edge_data = G.get_edge_data(neighbor, symptom)
            if edge_data and 'relation' in edge_data and edge_data['relation'] != 'is_a':
                diagnoses.append(neighbor)
    return diagnoses


def find_closest_category(top_symptoms, categories,top_n):
    if isinstance(top_symptoms, pd.Series) and top_symptoms.empty:
        print("Warning: top_symptoms is empty.")
        return None
    category_votes = {category: 0 for category in categories}
    for symptom in top_symptoms:
        top_symptoms = list(set(top_symptoms))

        # print('symptom: ',symptom)
        if symptom not in G:
            print(f"Symptom node not found in graph: {symptom}")
            continue

        diagnosis_nodes = get_diagnoses_for_symptom(symptom)
        for diagnosis in diagnosis_nodes:

            individual_diagnoses = diagnosis.split(',')

            for single_diagnosis in individual_diagnoses:
                single_diagnosis = single_diagnosis.strip().replace(' ', '_').lower()  # 去掉前后空格
                if single_diagnosis not in G:
                    print(f"Diagnosis node not found in graph: {single_diagnosis}")
                    continue

                min_distance = float('inf')
                closest_category = None

                for category in categories:
                    if category not in G:
                        print(f"Category node not found in graph: {category}")
                        continue

                    try:
                        distance = nx.shortest_path_length(G, source=single_diagnosis, target=category)
                    except nx.NetworkXNoPath:
                        distance = float('inf')

                    if distance < min_distance:
                        min_distance = distance
                        closest_category = category

                if closest_category:
                    category_votes[closest_category] += 1
    print("Category votes:", category_votes)

    sorted_categories = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)
    top_n_categories = [sorted_categories[i][0] for i in range(top_n)]
    return top_n_categories


def get_keyinfo_for_category(category, knowledge_graph):
    keyinfo_values = []
    for node, edges in knowledge_graph.items():
        if node == category:
            for relation, neighbor in edges:
                if relation == "is_a" and neighbor in knowledge_graph:
                    for rel, obj in knowledge_graph[neighbor]:
                        if rel == "has_keyinfo":
                            keyinfo_values.append(obj)
    return keyinfo_values



def get_subjects_for_objects(objects, knowledge_graph):
    subjects = []
    processed_objects = [obj.replace(' ', '_') for obj in objects]
    for obj in processed_objects:
        for index, row in knowledge_graph.iterrows():
            if row['object'] == obj:
                subjects.append(row['subject'])
    return subjects


def find_level3_for_symptoms(top_symptoms, knowledge_graph):
    level3_connections = {}
    for symptom in top_symptoms:
        subjects = get_subjects_for_objects([symptom], knowledge_graph)
        for subject in subjects:
            if subject in level3_connections:
                level3_connections[subject] += 1
            else:
                level3_connections[subject] = 1
    return level3_connections


def print_symptom_and_disease(symptom_nodes):
    for symptom in symptom_nodes:
        subjects = get_subjects_for_objects([symptom], kg_data)


def main_get_category_and_level3(n, participant_no,top_n):
    """
    根据参与者编号获取其诊断分类和第三层级信息
    
    参数:
        n: 未知参数（根据函数名推断可能与类别相关）
        participant_no: 参与者编号，用于在数据中查找特定参与者
        top_n: 未知参数（根据参数名推断可能与top-n选择相关）
    
    返回值:
        如果找到参与者，返回处理后的诊断信息；如果未找到参与者，返回None
    """
    # 读取CSV文件数据
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # 根据参与者编号查找对应行数据
    row = data.loc[data['Participant No.'] == str(participant_no)]
    if row.empty:
        print(f"Participant No. {participant_no} not found!")
        return None

    # 提取第二层级信息并分割处理
    tr = row["Level 2"].values[0]
    tr=tr.split(",")[0]

    # 提取处理后的诊断信息
    level3real = row["Processed Diagnosis"].values[0]

    # 提取疼痛相关信息
    pain_location = row["Pain Presentation and Description"].values[0]
    pain_symptoms = row["Pain descriptions and assorted symptoms (self-report)"].values[0]
    pain_restriction = row["Pain restriction"].values[0]
    print(f'pain_location: {pain_location}')
    print(f'pain_symptoms: {pain_symptoms}')
    print(f'pain_restrction: {pain_restriction}')
    
    # 处理空值情况，将NaN替换为空字符串
    if pd.isna(pain_location):
        pain_location = ''
    if pd.isna(pain_symptoms):
        pain_symptoms = ''
    if pd.isna(pain_restriction):
        pain_symptoms = ''


    def process_symptom_field(field_value, symptom_nodes, symptom_embeddings, n):
        if pd.isna(field_value) or field_value == '':
            return []
        return find_top_n_similar_symptoms(field_value, symptom_nodes, symptom_embeddings, n)

    top_5_location_nodes = process_symptom_field(pain_location, symptom_nodes, symptom_embeddings, n)
    top_5_symptom_nodes = process_symptom_field(pain_symptoms, symptom_nodes, symptom_embeddings, n)
    top_5_painrestriction_nodes = process_symptom_field(pain_restriction, symptom_nodes, symptom_embeddings, n)


    top_5_location_nodes_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_location_nodes), 'object'].drop_duplicates()
    top_5_symptom_nodes_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_symptom_nodes), 'object'].drop_duplicates()
    top_5_painrestriction_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_painrestriction_nodes), 'object'].drop_duplicates()


    most_similar_category = find_closest_category(
        list(top_5_location_nodes_original) + list(top_5_symptom_nodes_original)+ list(top_5_painrestriction_original),
        categories,
        top_n
    )
    return most_similar_category
