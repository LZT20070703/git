from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import requests


# 百度百科「词条摘要」接口（无需 key，国内网络可直接访问），替代原维基百科检索
BAIKE_API = "https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
BAIKE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://baike.baidu.com/",
}
# 精确匹配不到时，尝试去掉这些常见后缀再检索一次（如 "sora模型" -> "sora"）
_STRIP_SUFFIXES = ("模型", "技术", "系统", "平台", "公司", "软件", "工具")


def baidu_baike_search(keyword, timeout=15):
    """用百度百科词条摘要接口检索资料。

    返回拼接好的文本；查不到或网络异常时返回空字符串，不抛异常。
    """
    candidates = [keyword]
    for suffix in _STRIP_SUFFIXES:
        if keyword.endswith(suffix) and len(keyword) > len(suffix):
            candidates.append(keyword[: -len(suffix)])
            break

    for kw in candidates:
        try:
            resp = requests.get(
                BAIKE_API,
                params={"scope": 103, "format": "json", "appid": 379020,
                        "bk_key": kw, "bk_length": 600},
                headers=BAIKE_HEADERS,
                timeout=timeout,
            )
            data = resp.json()
        except Exception as e:
            print(f"[警告] 百度百科检索“{kw}”出错：{e}")
            continue

        abstract = data.get("abstract")
        if abstract:
            title = data.get("title", kw)
            return f"【百度百科·{title}】{abstract}"

    return ""


def generate_script(subject, video_length, creativity, api_key):
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human", "请为'{subject}'这个主题的视频想一个吸引人的标题")
        ]
    )
    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
             """你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
             视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
             要求开头抓住限球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
             整体内容的表达方式要尽量轻松有趣，吸引年轻人。
             脚本内容可以结合以下百度百科搜索出的信息，但仅作为参考，只结合相关的即可，对不相关的进行忽略：
             ```{baike_search}```""")
        ]
    )

    model = ChatOpenAI(
        model="deepseek-flash",
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        temperature=creativity,
        base_url="https://api.deepseek.com/v1",
    )

    title_chain = title_template | model
    script_chain = script_template | model

    title = title_chain.invoke({"subject": subject}).content

    # 百度百科检索失败时不影响脚本生成，仅跳过参考资料
    baike_result = baidu_baike_search(subject)
    if not baike_result:
        print("[警告] 百度百科未检索到相关内容，仅使用模型自身知识生成脚本。")

    script = script_chain.invoke({"title": title, "duration": video_length,
                                  "baike_search": baike_result}).content

    return baike_result, title, script


