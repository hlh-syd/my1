# CausalML 安装指南

## 安装要求

CausalML 需要 Python 3.7 或以上版本。

## 基本安装

### 使用pip安装（推荐）

一次性安装CausalML及其所有依赖项：

```bash
pip install causalml[all]
```

这将安装CausalML及其所有可选依赖项。

### 从源代码安装

克隆仓库并安装：

```bash
git clone https://github.com/uber/causalml.git
cd causalml
pip install -e .[all]
```

### 安装特定功能子集

根据您的需求安装特定功能：

1. 仅安装基本功能：
```bash
pip install causalml
```

2. 安装带有PyTorch支持的功能（如CEVAE）：
```bash
pip install causalml[torch]
```

3. 安装所有依赖项：
```bash
pip install causalml[all]
```

## 使用conda安装

CausalML 也可以通过 conda-forge 安装：

```bash
conda install -c conda-forge causalml
```

## 快速安装项目依赖

如果您已经克隆了项目代码，可以使用以下命令一次性安装所有依赖项：

```bash
pip install -r requirements.txt
```

## 开发环境设置

对于开发者，建议安装额外的开发工具：

```bash
pip install -e ".[dev]"
```

## 测试安装

安装后，您可以运行以下命令验证安装：

```python
import causalml
print(causalml.__version__)
```

## 常见问题

- **Cython相关错误**：如果在安装过程中遇到Cython相关错误，请先单独安装Cython：
  ```bash
  pip install Cython
  ```

- **多线程支持问题**：在Windows系统上，某些多线程功能可能需要额外配置：
  ```bash
  pip install causalml[all]
  ```

- **GPU支持**：要使用GPU加速（针对PyTorch模型），请安装相应的CUDA版本：
  ```bash
  pip install causalml[torch] torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```
