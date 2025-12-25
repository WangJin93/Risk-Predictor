# Mortality Risk Predictor - 部署指南

## 问题修复

### 修复的问题
部署后出现错误：`Failed to display SHAP visualizations: IPython must be installed to use initjs()!`

### 根本原因
在Streamlit环境中，不应该使用`shap.initjs()`，因为Streamlit有自己的组件系统来处理JavaScript可视化。

### 解决方案
从代码中移除了`shap.initjs()`调用。Streamlit的`st_shap`组件会自动处理SHAP可视化的JavaScript部分。

## 部署方法

### 1. Streamlit Cloud部署（推荐，最简单）

#### 步骤：
1. **准备GitHub仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/mortality-predictor.git
   git push -u origin main
   ```

2. **部署到Streamlit Cloud**
   - 访问 https://share.streamlit.io/
   - 连接GitHub账户
   - 选择你的仓库
   - 主文件路径设置为：`app2.py`
   - 点击"Deploy!"

#### 优点：
- 完全免费
- 自动HTTPS
- 自动扩展
- 无需服务器管理

### 2. Docker部署（推荐用于生产环境）

#### 构建和运行：
```bash
# 构建镜像
docker build -t mortality-predictor .

# 运行容器
docker run -p 8501:8501 mortality-predictor

# 或使用docker-compose
docker-compose up -d
```

#### 访问：
http://localhost:8501

### 3. 服务器部署

#### 使用部署脚本：
```bash
# 给脚本执行权限
chmod +x deploy.sh

# 运行部署脚本
./deploy.sh
```

#### 手动部署：
```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app2.py --server.address=0.0.0.0 --server.port=8501
```

## 文件清单

### 必需文件：
- `app2.py` - 主应用程序
- `death_model.pkl` - 训练好的模型
- `scaler_model.pkl` - 特征缩放器
- `requirements.txt` - Python依赖

### 配置文件：
- `.streamlit/config.toml` - Streamlit配置
- `Dockerfile` - Docker镜像配置
- `docker-compose.yml` - Docker Compose配置
- `deploy.sh` - 服务器部署脚本

## 依赖项

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
shap>=0.42.0
matplotlib>=3.7.0
joblib>=1.3.0
scikit-learn>=1.3.0
xgboost>=1.7.0
```

## 配置说明

### Streamlit配置（.streamlit/config.toml）
```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true
runOnSave = false

[theme]
base = "light"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 故障排除

### SHAP可视化问题
**问题：** `IPython must be installed to use initjs()!`

**解决方案：** 
- 已在代码中移除`shap.initjs()`调用
- Streamlit会自动处理SHAP可视化

### 模型加载问题
**问题：** 无法加载模型文件

**解决方案：**
- 确保`death_model.pkl`和`scaler_model.pkl`在正确的位置
- 检查文件权限

### 端口冲突
**问题：** 端口8501已被占用

**解决方案：**
```bash
# 使用其他端口
streamlit run app2.py --server.port=8502
```

## 性能优化

### 缓存策略
- 使用`@st.cache_resource`缓存模型加载
- 使用`@st.cache_data`缓存数据处理
- 使用参数前导下划线避免哈希不可哈希对象

### 内存管理
- 及时关闭matplotlib图形
- 使用`constrained_layout`避免警告
- 限制SHAP显示的特征数量（max_display=10）

## 安全建议

1. **模型安全**
   - 确保.pkl文件不包含敏感数据
   - 考虑使用模型加密

2. **网络安全**
   - 使用HTTPS
   - 添加身份验证
   - 实现访问控制

3. **数据保护**
   - 不记录用户输入数据
   - 遵守相关法规（GDPR/HIPAA）

## 监控和维护

### 日志监控
- 检查Streamlit日志
- 监控应用性能
- 跟踪错误率

### 更新维护
- 定期更新依赖项
- 监控模型性能
- 收集用户反馈

## 联系和支持

如有问题，请检查：
1. Streamlit文档：https://docs.streamlit.io/
2. SHAP文档：https://shap.readthedocs.io/
3. XGBoost文档：https://xgboost.readthedocs.io/

---

**注意：** 此应用程序仅用于教育和研究目的，不适用于临床决策。