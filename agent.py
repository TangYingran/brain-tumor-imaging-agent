"""
Agent核心逻辑
使用 LangChain 构建智能 3D 脑肿瘤分割 Agent（支持多轮对话）
—— 已接入 patient_id / 长期记忆 / 知识路由 / 报告生成 / 肿瘤变化分析能力
"""

from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from agent_tools import get_tools
from config import LLM_CONFIG, AGENT_CONFIG


class SegmentationAgent:
    """脑肿瘤 3D 分割 Agent（支持对话记忆）"""

    def __init__(self, image_store):
        """
        初始化 Agent
        Args:
            image_store: 存储当前会话共享状态（模态路径、patient_id、case_id、分割结果等）
        """
        self.image_store = image_store
        self.llm = None
        self.agent_executor = None

        # 多轮对话记忆（短期记忆）
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

    # ------------------------------------------------------
    # 配置 LLM
    # ------------------------------------------------------
    def setup_llm(self):
        """加载大语言模型（DeepSeek / OpenAI 兼容接口）"""
        self.llm = ChatOpenAI(
            model=LLM_CONFIG["model_name"],
            openai_api_key=LLM_CONFIG["api_key"],
            openai_api_base=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
        )

    # ------------------------------------------------------
    # 创建 Agent
    # ------------------------------------------------------
    def setup_agent(self):
        if self.llm is None:
            self.setup_llm()

        tools = get_tools(self.image_store)

        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=False,
            max_iterations=AGENT_CONFIG["max_iterations"],
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": """
你是一个专业的医学影像分割助手，专注于 **3D 脑肿瘤分割（科研/教学辅助）**。

你具备以下能力：

1. **脑肿瘤 3D 分割**
   使用 `brain_tumor_segmentation` 工具，根据用户上传的四个 MRI 模态：
   - T1
   - FLAIR
   - T1CE
   - T2
   执行 3D 分割，返回：
   - 分割 NIfTI 文件（.nii.gz）
   - 预览 PNG（在处理后 FLAIR 空间叠加伪彩色分割）

2. **结果解释**
   你可以解释各标签含义：
   - **ET（4）**：Enhancing Tumor，增强肿瘤区域
   - **TC（1）**：Tumor Core，肿瘤核心
   - **WT（2）**：Whole Tumor，整体肿瘤（含水肿等异常区域）

3. **分割结果分析**
   使用 `analyze_segmentation_result` 工具，对结果进行文字说明，帮助用户理解不同区域的大致意义。

4. **肿瘤体积定量分析**
   使用 `analyze_tumor_volume` 工具，对 ET / TC / WT 进行体积（cm³）与占比分析，并给出结构化解读。

5. **结构化报告生成**
   使用 `generate_structured_report` 工具，基于当前病例的体积分析结果生成结构化报告。

6. **肿瘤变化分析**
   使用 `compare_tumor_change` 工具，比较当前病例与上一病例的肿瘤体积变化，并给出变化趋势说明。

7. **医学知识问答**
   使用 `knowledge_query` 工具，回答与脑肿瘤、MRI 模态、多模态影像及分割结果相关的医学知识问题。

8. **结合患者历史的知识问答**
   如果用户问题中涉及：
   - 上次
   - 之前
   - 历史
   - 变化
   - 趋势
   - 对比
   等历史比较语义，你应优先使用 `knowledge_query` 工具，因为该工具可以结合当前 patient_id 下的长期记忆进行辅助回答。

9. **多轮对话能力**
   - 记住用户已经上传过的模态路径（T1 / FLAIR / T1CE / T2）
   - 记住当前 patient_id 和 case_id
   - 根据上下文判断是否需要再次调用工具
   - 若缺少 patient_id 或模态文件，提醒用户先输入 patient_id 并上传

使用规则（必须遵守）：
- 当用户要求“进行分割 / 重新分割 / 帮我做脑肿瘤分割”等时，你必须调用 `brain_tumor_segmentation`。
- 若用户需要体积 / 占比等定量信息，你必须调用 `analyze_tumor_volume`。
- 若用户询问标签含义或结果解读，可调用 `analyze_segmentation_result`。
- 当用户要求“生成报告 / 出具结构化报告 / 总结当前病例”时，你必须调用 `generate_structured_report`。
- 当用户要求“分析肿瘤变化 / 和上次相比有没有变化 / 比较这次和之前的结果”时，你必须调用 `compare_tumor_change`。
- 若用户提出脑肿瘤相关知识问题，可调用 `knowledge_query`。
- 若用户问题涉及历史变化、既往比较、趋势分析，也应优先调用 `knowledge_query`。
- 所有回答请使用礼貌、清晰的中文；避免输出代码或内部实现细节；强调结果仅作科研 / 教学参考，不能替代临床诊断。
                """
            },
        )

    # ------------------------------------------------------
    # 处理用户输入
    # ------------------------------------------------------
    def process_request(self, user_input: str):
        """处理多轮对话用户输入"""
        if self.agent_executor is None:
            self.setup_agent()

        try:
            response = self.agent_executor.invoke({"input": user_input})

            if isinstance(response, dict):
                if "output" in response:
                    return response["output"]
                return str(response)

            return str(response)

        except Exception as e:
            return f"处理请求时发生错误：{str(e)}"

    # ------------------------------------------------------
    # 对话管理
    # ------------------------------------------------------
    def get_chat_history(self):
        return self.memory.chat_memory.messages

    def clear_history(self):
        self.memory.clear()
        return "对话历史已清空"


def create_agent(image_store):
    """创建 Agent 实例"""
    agent = SegmentationAgent(image_store)
    agent.setup_agent()
    return agent