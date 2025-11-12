#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的LLM服务脚本
使用Ollama部署Qwen模型，提供HTTP API接口用于文本标准化
适配kimi-audio conda环境，不依赖deprecated目录
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import time

# 设置代理绕过，确保可以访问本地Ollama服务
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

# 尝试导入ollama库
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("警告: ollama库未安装，请运行: pip install ollama")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextNormalizationRequest(BaseModel):
    """文本标准化请求模型"""
    text1: str
    text2: str

class TextNormalizationResponse(BaseModel):
    """文本标准化响应模型"""
    normalized_text1: str
    normalized_text2: str
    success: bool
    error_message: Optional[str] = None

class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    model_name: str
    model_type: str
    ollama_host: str
    status: str

class OllamaLLMService:
    """Ollama LLM服务类"""
    
    def __init__(self, model_name: str = "qwen3:32b", model_type: str = "qwen3", ollama_host: str = "localhost:11434"):
        """
        初始化Ollama LLM服务
        
        Args:
            model_name: Ollama模型名称
            model_type: 模型类型，影响prompt策略
            ollama_host: Ollama服务地址
        """
        self.model_name = model_name
        self.model_type = model_type
        self.ollama_host = ollama_host
        self.ollama_url = f"http://{ollama_host}"
        self.client = None
        
        logger.info(f"Ollama LLM服务配置:")
        logger.info(f"  模型名称: {self.model_name}")
        logger.info(f"  模型类型: {self.model_type}")
        logger.info(f"  Ollama服务地址: {self.ollama_url}")
        
    def check_ollama_service(self, max_retries=5, retry_delay=2):
        """检查Ollama服务是否运行，支持重试"""
        session = requests.Session()
        session.proxies = {
            'http': None,
            'https': None
        }
        
        for attempt in range(max_retries):
            try:
                response = session.get(f"{self.ollama_url}/api/version", timeout=10)
                response.raise_for_status()
                version_info = response.json()
                logger.info(f"✓ Ollama服务正常运行，版本: {version_info.get('version', 'unknown')}")
                return True
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama服务检查失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"✗ Ollama服务在 {max_retries} 次尝试后仍不可用: {e}")
                    return False
            except Exception as e:
                logger.error(f"✗ Ollama服务检查出现意外错误: {e}")
                return False
    
    def load_model(self):
        """加载/检查模型"""
        try:
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama库未安装，请运行: pip install ollama")
            
            # 初始化ollama客户端，设置host
            self.client = ollama.Client(host=self.ollama_url)
            
            # 检查模型是否已加载
            models = self.client.list()
            # ollama库返回的是对象，不是字典，模型字段是'model'而不是'name'
            model_names = [model.model for model in models.models]
            
            if self.model_name not in model_names:
                logger.warning(f"模型 {self.model_name} 未找到，尝试拉取...")
                try:
                    self.client.pull(self.model_name)
                    logger.info(f"✓ 成功拉取模型: {self.model_name}")
                except Exception as e:
                    logger.error(f"✗ 拉取模型失败: {e}")
                    raise
            else:
                logger.info(f"✓ 模型已存在: {self.model_name}")
            
            # 测试模型
            test_response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': '你好'}
                ]
            )
            
            if test_response:
                logger.info(f"✓ 模型测试成功")
                return True
            else:
                logger.error("✗ 模型测试失败")
                return False
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def normalize_text_pair(self, text1: str, text2: str) -> tuple[str, str, bool, Optional[str]]:
        """
        使用LLM标准化文本对
        
        Returns:
            (normalized_text1, normalized_text2, success, error_message)
        """
        try:
            # 构造更详细的标准化提示词
            prompt = self._build_normalization_prompt(text1, text2)
            
            # 调用LLM
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 512,
                    "stop": ["<|endoftext|>", "<|im_end|>"]
                }
            )
            
            response_text = response['message']['content']
            
            # 使用改进的解析方法
            normalized_text1, normalized_text2 = self._extract_normalized_texts(response_text, text1, text2)
            
            logger.info(f"文本标准化完成")
            logger.info(f"原始文本1: '{text1}' -> 标准化: '{normalized_text1}'")
            logger.info(f"原始文本2: '{text2}' -> 标准化: '{normalized_text2}'")
            
            return normalized_text1, normalized_text2, True, None
            
        except Exception as e:
            logger.error(f"文本标准化失败: {e}")
            return text1, text2, False, str(e)
    
    def _build_normalization_prompt(self, text1: str, text2: str) -> str:
        """构造详细的标准化提示词"""
        # 根据模型类型决定是否添加直接回答指令
        direct_instruction = ""
        if self.model_type == "qwen3":
            direct_instruction = "\n请直接给出标准化结果，不要进行过多分析和思考。"
        
        prompt = f"""你是语音识别文本标准化专家。请将两段文本标准化为统一格式，用于WER/CER计算。{direct_instruction}

# 核心原则
- 只做格式标准化，不改变语义内容
- 保持原文的语言错误（如同音词误用）
- 两个文本必须使用完全相同的标准化规则

# 标准化规则

## 英语标准化：
**基础处理**:
- 转小写: "Hello" → "hello"
- 去标点: "Hello!" → "hello" 
- 去连字符: "twenty-one" → "twenty one"

**缩写展开**:
- "don't" → "do not"
- "I'm" → "i am" 
- "can't" → "cannot"
- "won't" → "will not"
- "it's" → "it is" (永远展开为"it is"，不是"its")
- "you're" → "you are"
- "we're" → "we are"
- "they're" → "they are"

**数字转换**:
- 单独数字: "3" → "three"
- 连续数字: "123" → "one hundred twenty three"
- 年份: "2023" → "two thousand twenty three"
- 序数词: "1st" → "first", "2nd" → "second"

**字母拼读统一规则** (重要):
- 检查两个文本中是否有字母序列
- 如果一个是拼读形式(如"a b c")，另一个是连写(如"abc")
- 统一转换为拼读形式: "abc" → "a b c"
- 示例: "My name is ABC" + "my name is a b c" → 都变成 "my name is a b c"

## 中文标准化：
- 去标点: "你好！" → "你好"
- 去空格: "你 好" → "你好" 
- 繁转简: "這個" → "这个", "機器" → "机器"
- 数字转换: 
  - "3个" → "三个"
  - "2023年" → "二零二三年"
  - "第1个" → "第一个"
  - "100元" → "一百元"
- **同音词保持原样**: "在家"保持"在家", "再家"保持"再家" (不纠错)
- 语气词统一: "嗯" → "嗯", "呃" → "呃"

## 中英混合处理：
- 英文部分按英语规则处理
- 中文部分按中文规则处理
- 数字在中英文语境中分别转换
- 保持语言切换的自然性

# 处理步骤示例

输入: "My name is A B C." + "my name is abc"
步骤:
1. 基础标准化: "my name is a b c" + "my name is abc"  
2. 检测拼读模式: 第一个已是拼读，第二个是连写
3. 统一为拼读: "my name is a b c" + "my name is a b c"

输入: "我在家里。" + "我再家里"
步骤:
1. 去标点: "我在家里" + "我再家里"
2. 保持同音词原样: "我在家里" + "我再家里" (不改变"再")

输入: "Hello, it's ok!" + "hello it is okay"  
步骤:
1. 基础处理: "hello its ok" + "hello it is okay"
2. 缩写展开: "hello it is ok" + "hello it is okay"  
3. 注意"it's"必须展开为"it is"

输入: "我有3个apple" + "我有三个苹果"
步骤:
1. 中文数字统一: "我有三个apple" + "我有三个苹果"
2. 英文小写: "我有三个apple" + "我有三个苹果"

---

待处理文本:
文本1: "{text1}"
文本2: "{text2}"

按规则处理后直接返回，注意格式：

标准化文本1: 处理结果
标准化文本2: 处理结果

重要：
- 只返回处理后的纯文本，不要加引号、方括号等任何标记
- 确保两个文本使用完全相同的处理规则"""
        
        return prompt
    
    def _extract_normalized_texts(self, response: str, original_text1: str, original_text2: str) -> tuple[str, str]:
        """从LLM响应中提取标准化后的文本，支持多种格式"""
        import re
        
        try:
            logger.info(f"LLM响应: {response}")
            
            # 查找标准化文本的模式，支持多种格式
            patterns = [
                # 标准格式（主要格式）
                (r'标准化文本1[:：]\s*(.+?)(?=\n标准化文本2|标准化文本2|$)', r'标准化文本2[:：]\s*(.+?)(?=\n|$)'),
                # 方括号格式
                (r'标准化文本1[:：]\s*\[(.+?)\]', r'标准化文本2[:：]\s*\[(.+?)\]'),
                # 简化格式
                (r'文本1[:：]\s*(.+?)(?=\n文本2|文本2|$)', r'文本2[:：]\s*(.+?)(?=\n|$)'),
                # 编号格式
                (r'1\.\s*(.+?)(?=\n2\.|2\.|$)', r'2\.\s*(.+?)(?=\n|$)'),
                # 英文格式
                (r'Normalized Text 1[:：]\s*(.+?)(?=\nNormalized Text 2|Normalized Text 2|$)', 
                 r'Normalized Text 2[:：]\s*(.+?)(?=\n|$)'),
                # 混合格式（处理多行响应）
                (r'(?:标准化文本1|Text 1)[:：]\s*(.+?)(?=\n(?:标准化文本2|Text 2)|(?:标准化文本2|Text 2)|$)', 
                 r'(?:标准化文本2|Text 2)[:：]\s*(.+?)(?=\n|$)'),
            ]
            
            for pattern1, pattern2 in patterns:
                match1 = re.search(pattern1, response, re.DOTALL | re.IGNORECASE)
                match2 = re.search(pattern2, response, re.DOTALL | re.IGNORECASE)
                
                if match1 and match2:
                    norm_text1 = match1.group(1).strip()
                    norm_text2 = match2.group(1).strip()
                    
                    # 清理各种可能的标记和格式
                    norm_text1 = re.sub(r'^\[|]$|^"|"$|^\'|\'$', '', norm_text1).strip()
                    norm_text2 = re.sub(r'^\[|]$|^"|"$|^\'|\'$', '', norm_text2).strip()
                    
                    # 验证提取的文本是否有效
                    if norm_text1 and norm_text2:
                        logger.info(f"成功使用正则匹配提取标准化文本")
                        return norm_text1, norm_text2
            
            # 如果没有匹配到，尝试更简单的提取方法
            lines = response.strip().split('\n')
            extracted_texts = []
            
            for line in lines:
                line = line.strip()
                # 寻找包含实际文本内容的行
                if line and not line.startswith(('标准化', '文本', '规则', '请', '以下', '格式', '步骤', '输入', '重要')):
                    # 移除可能的编号和标点
                    cleaned_line = re.sub(r'^\d+\.\s*|^[-*]\s*|^[:：]\s*', '', line).strip()
                    if cleaned_line:
                        extracted_texts.append(cleaned_line)
            
            if len(extracted_texts) >= 2:
                logger.info(f"使用简化提取方法成功")
                return extracted_texts[0], extracted_texts[1]
            
            # 最后的fallback：尝试按行解析原始响应
            lines = response.strip().split('\n')
            if len(lines) >= 2:
                # 清理可能的前缀
                result1 = lines[0].strip()
                result2 = lines[1].strip()
                
                for prefix in ["标准化文本1:", "标准化文本1：", "1.", "1、", "文本1:", "文本1："]:
                    if result1.startswith(prefix):
                        result1 = result1[len(prefix):].strip()
                        break
                
                for prefix in ["标准化文本2:", "标准化文本2：", "2.", "2、", "文本2:", "文本2："]:
                    if result2.startswith(prefix):
                        result2 = result2[len(prefix):].strip()
                        break
                
                if result1 and result2:
                    logger.info(f"使用fallback方法成功提取")
                    return result1, result2
            
            logger.error(f"无法从LLM响应中提取标准化文本")
            raise RuntimeError("LLM响应格式无法解析")
                
        except Exception as e:
            logger.error(f"提取标准化文本失败: {e}")
            # 如果所有方法都失败，返回原始文本
            logger.warning("返回原始文本作为fallback")
            return original_text1, original_text2

    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """保持向后兼容的解析方法"""
        return self._extract_normalized_texts(response_text, "", "")

# 全局LLM服务实例
ollama_service = None

# 创建FastAPI应用
app = FastAPI(
    title="LLM文本标准化服务",
    description="基于Ollama的文本标准化API服务",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """启动时初始化LLM服务"""
    global ollama_service
    if ollama_service:
        logger.info("LLM服务已初始化")
    else:
        logger.error("LLM服务未初始化")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "message": "LLM服务运行正常"}

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    global ollama_service
    if not ollama_service:
        raise HTTPException(status_code=500, detail="LLM服务未初始化")
    
    return ModelInfoResponse(
        model_name=ollama_service.model_name,
        model_type=ollama_service.model_type,
        ollama_host=ollama_service.ollama_host,
        status="running"
    )

@app.post("/normalize", response_model=TextNormalizationResponse)
async def normalize_text(request: TextNormalizationRequest):
    """文本标准化接口"""
    global ollama_service
    if not ollama_service:
        raise HTTPException(status_code=500, detail="LLM服务未初始化")
    
    try:
        normalized_text1, normalized_text2, success, error_message = ollama_service.normalize_text_pair(
            request.text1, request.text2
        )
        
        return TextNormalizationResponse(
            normalized_text1=normalized_text1,
            normalized_text2=normalized_text2,
            success=success,
            error_message=error_message
        )
        
    except Exception as e:
        logger.error(f"文本标准化请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"文本标准化失败: {str(e)}")

def main():
    """主函数"""
    global ollama_service
    
    parser = argparse.ArgumentParser(description="LLM文本标准化服务")
    parser.add_argument("--model_name", type=str, default="qwen3:32b",
                        help="Ollama模型名称")
    parser.add_argument("--model_type", type=str, default="qwen3",
                        choices=["qwen2.5", "qwen3"],
                        help="模型类型，影响prompt策略")
    parser.add_argument("--ollama_host", type=str, default="localhost:11434",
                        help="Ollama服务地址")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="HTTP服务绑定地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP服务端口")
    parser.add_argument("--workers", type=int, default=1,
                        help="服务工作进程数")
    
    args = parser.parse_args()
    
    # 初始化LLM服务
    logger.info("初始化LLM服务...")
    ollama_service = OllamaLLMService(
        model_name=args.model_name,
        model_type=args.model_type,
        ollama_host=args.ollama_host
    )
    
    # 检查Ollama服务
    if not ollama_service.check_ollama_service():
        logger.error("Ollama服务不可用，请先启动Ollama服务")
        return 1
    
    # 加载模型
    if not ollama_service.load_model():
        logger.error("模型加载失败")
        return 1
    
    # 启动HTTP服务
    logger.info(f"启动HTTP服务: http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
    
    return 0

if __name__ == "__main__":
    exit(main()) 