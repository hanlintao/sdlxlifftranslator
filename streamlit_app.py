import streamlit as st
import xml.etree.ElementTree as ET
import concurrent.futures
import time
import pandas as pd
from langchain_openai.chat_models.base import ChatOpenAI

# 确保安装并导入 openpyxl
try:
    import openpyxl
except ImportError:
    st.error("The `openpyxl` library is required to read Excel files. Please install it using `pip install openpyxl`.")

# Streamlit sidebar for API keys
st.sidebar.title("API Key Input")
api_keys = []

num_keys = st.sidebar.number_input("Number of API keys", min_value=1, max_value=10, value=1, step=1)

for i in range(num_keys):
    api_key = st.sidebar.text_input(f"API Key {i+1}", type="password")
    if api_key:
        api_keys.append(api_key.strip())

if not api_keys:
    st.sidebar.warning("Please enter at least one API key.")

# 读取术语表
def load_terms(file_path):
    df = pd.read_excel(file_path)
    terms = dict(zip(df['term_zh'], df['term_en']))
    return terms

# 匹配术语表
def match_terms(text, terms):
    matched_terms = {k: v for k, v in terms.items() if k in text}
    return matched_terms

# 定义翻译函数
def translate_sentence(sentence, api_key, matched_terms):
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    template = "你是一位资深翻译，是英中翻译专家，请将下面的英文翻译成适合外国专家阅读的中文，且仅输出最终的译文，如果原文有格式标记，也务必保留，翻译的时候严格参考以下术语表：{}。"

    term_pairs = ', '.join([f"{k}: {v}" for k, v in matched_terms.items()])
    full_template = template.format(term_pairs)

    for i in range(3):
        try:
            messages = [
                {"role": "system", "content": full_template},
                {"role": "user", "content": sentence}
            ]

            translation_output = llm.invoke(messages).content
            return translation_output
        except Exception as e:
            print(f"翻译异常: {e}")
            time.sleep(5)  # 等待5秒后重试

# 读取sdlxliff文件
def parse_sdlxliff(file):
    # 使用实际文件中的命名空间
    namespaces = {
        'xliff': 'urn:oasis:names:tc:xliff:document:1.2',
        'sdl': 'http://sdl.com/FileTypes/SdlXliff/1.0'
    }

    # 解析sdlxliff文件
    tree = ET.parse(file)
    root = tree.getroot()

    # 查找所有trans-units
    trans_units = root.findall('.//xliff:trans-unit', namespaces)

    to_translate_texts = []
    for unit in trans_units:
        source = unit.find('.//xliff:seg-source', namespaces)
        target = unit.find('.//xliff:target', namespaces)

        source_text = []
        target_text = []

        if source is not None:
            for source_seg in source.findall('.//xliff:mrk', namespaces):
                source_text.append(''.join(source_seg.itertext()))

        if target is not None:
            for target_seg in target.findall('.//xliff:mrk', namespaces):
                target_text.append(''.join(target_seg.itertext()))

        source_text_str = ' '.join(source_text)

        to_translate_texts.append(source_text_str)

    return to_translate_texts

# Streamlit app
st.title("SDLXLIFF Translator with Term Matching")

uploaded_sdlxliff = st.file_uploader("Upload SDLXLIFF File", type=["sdlxliff"])
uploaded_terms = st.file_uploader("Upload Terms File", type=["xlsx"])

if uploaded_sdlxliff and uploaded_terms and api_keys:
    try:
        to_translate_texts = parse_sdlxliff(uploaded_sdlxliff)
        terms = load_terms(uploaded_terms)

        # 统计待翻译的总数
        total_to_translate = len(to_translate_texts)
        st.write(f"总共有 {total_to_translate} 条待翻译的内容。")

        # 使用多线程进行翻译
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # 提交翻译任务
            futures = {
                executor.submit(translate_sentence, sentence, api_keys[i % len(api_keys)], match_terms(sentence, terms)): sentence
                for i, sentence in enumerate(to_translate_texts)
            }

            # 等待所有任务完成
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                source = futures[future]
                translation = future.result()
                matched_terms = match_terms(source, terms)
                term_pairs = ', '.join([f"{k}: {v}" for k, v in matched_terms.items()])
                results.append((source, translation, term_pairs))

                # 显示进度
                st.write(f"翻译进度: {i + 1}/{total_to_translate} ({(i + 1) / total_to_translate * 100:.2f}%)")

        # 创建包含原文、译文和参考术语的Excel文件
        df_results = pd.DataFrame(results, columns=['原文', '译文', '参考的术语'])
        df_results.to_excel('translation_results.xlsx', index=False)

        st.write("翻译结果已生成。")
        st.download_button("下载翻译结果", data=df_results.to_excel(index=False), file_name="translation_results.xlsx")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    if not api_keys:
        st.warning("请在左侧输入至少一个API密钥。")
