## 背景介绍

OpenAI的API提供了强大的工具调用功能，允许模型与外部函数或工具进行交互。这个功能最初被称为Function Calling，后来演变为更灵活的Tool Calling。

### 从Function Calling到Tool Calling

OpenAI最初引入了`function_call`和`functions`参数来支持函数调用。然而，随着功能的扩展，这些参数已被弃用，取而代之的是更通用的`tool_choice`和`tools`参数。

- `function_call` → `tool_choice`
- `functions` → `tools`

这一变化使得API能够支持更广泛的工具类型。

## 本仓库特点

在接入Didy等应用开发平台时,发现官方的`openai_api_server.py`在function call时如果是非Stream function calling,也就是非流式输出调用工具时,会正常输出，但是当选择Stream function calling时，输出为空,所以对此进行了优化。

本项目基于智谱AI公司发布的GLM4模型，成功实现了`tool_choice`功能的适配和优化。主要特点包括：

1. 支持OpenAI Tool Calling API规范。
2. 解决了官方`openai_api_server.py`在接入Didy等应用开发平台时function call不能正常调用的问题。
3. 经过测试，包括：
   - 本地环境测试
   - 在Dify等主流AI工作流平台上的集成测试

通过这些改进，用户可以更加稳定和高效地使用GLM4模型的工具调用功能，无论是在本地开发环境还是在生产级别的AI应用中。

## Dify接入第三方模型供应商

在OpenAI-API-compatible依次填入：

- Function calling 选择 Tool Call

- Stream function calling 选择 支持

## 安装

在安装本项目之前，推荐满足以下要求：

- Python版本：3.10
- 操作系统：Linux

安装本项目所需的依赖库，请运行以下命令：

```
pip install -r requirements.txt
```

## 使用方法

1. 克隆本项目到本地

2. 运行项目：

```
python openai_api_server.py
```

## 参考

1. [GLM-4/basic_demo/openai_api_server.py at main · THUDM/GLM-4 (github.com)](https://github.com/THUDM/GLM-4/blob/main/basic_demo/openai_api_server.py)
2. https://github.com/THUDM/GLM-4