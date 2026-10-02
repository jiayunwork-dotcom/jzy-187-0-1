FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CURRENT_DATA_DIR=/data

WORKDIR /srv

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷业务代码与测试
COPY app ./app
COPY tests ./tests
COPY pytest.ini ./pytest.ini

# 数据卷：SQLite 文件放在挂载卷上
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 只对外提供 HTTP 接口；容器内单进程即可，运维船数据量小
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
