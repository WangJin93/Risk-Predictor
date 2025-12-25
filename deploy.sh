#!/bin/bash

# 服务器部署脚本
echo "正在部署Mortality Risk Predictor..."

# 安装依赖
pip install -r requirements.txt

# 创建systemd服务文件
sudo tee /etc/systemd/system/mortality-predictor.service > /dev/null <<EOF
[Unit]
Description=Mortality Risk Predictor Streamlit App
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/path/to/your/app
ExecStart=/usr/local/bin/streamlit run app2.py --server.address=0.0.0.0 --server.port=8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable mortality-predictor
sudo systemctl start mortality-predictor

echo "部署完成！访问地址: http://your-server-ip:8501"