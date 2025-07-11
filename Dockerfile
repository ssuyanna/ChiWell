# 使用官方的 Python 3.10 slim 版本作为基础镜像
FROM python:3.10-slim

# 在容器中设置工作目录
WORKDIR /app

# 安装 Git 和 Git LFS，这对于拉取你的大文件模型至关重要
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# 复制 requirements.txt 文件到容器中
COPY requirements.txt .

# 安装所有依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 将你项目中的所有其他文件复制到容器中
COPY . .

# 暴露端口 7860，这是 Hugging Face Spaces 默认的 Web 服务端口
EXPOSE 7860

# 定义运行你应用的命令
# 使用 gunicorn 并在 0.0.0.0:7860 上监听
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]