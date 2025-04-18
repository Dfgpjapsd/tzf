教学课程画像分析系统使用说明
================================

1. 项目简介
本项目通过分析学生课程评价数据，自动构建课程特征画像，帮助识别不同类型的课程模式。系统使用机器学习技术对课程进行自动分类。

2. 数据说明
- 数据集包含278个课程评价文本文件(位于"数据集"文件夹)
- 每个文件对应一门课程的学生评价汇总
- 每条评价包含24个维度，如：
  * 教学方式评价
  * 学习效果评估
  * 课程内容掌握度
  * 教师表现等
- 评价使用程度副词表示：
  非常=5分 | 比较=4分 | 一般=3分 | 不太=2分 | 不=1分

3. 技术实现
3.1 技术栈
- 编程语言：Python 3.8+
- 核心库：
  • scikit-learn 1.3.0 (机器学习算法)
  • numpy 1.24.4 (数值计算)
  • matplotlib 3.7.2 (数据可视化) 
  • joblib 1.4.2 (模型持久化)
- 开发环境：VSCode/PyCharm

3.2 技术流程
3.2.1 数据预处理
- 文本清洗：去除空行和无效字符
- 程度词提取：识别"非常"/"比较"等程度副词
- 数值映射：转换为1-5分的数值评分
- 特征矩阵：构建24维特征向量

3.2.2 机器学习流程
- PCA降维：n_components=2，保留95%以上方差
- K-means参数：
  • n_clusters=3 (默认聚类数)
  • random_state=42 (随机种子)
  • n_init=10 (初始中心点次数)
- 模型评估：使用轮廓系数评估聚类效果

3.2.3 可视化流程
- 生成2D散点图展示聚类结果
- 使用viridis配色方案区分类别
- 添加PCA成分坐标轴标签
- 包含颜色图例说明类别

3.3 系统架构
```mermaid
graph TD
    A[原始评价数据] --> B[数据预处理]
    B --> C[特征工程]
    C --> D[PCA降维]
    D --> E[K-means聚类]
    E --> F[结果可视化]
    E --> G[模型持久化]
```

3.4 详细实现
3.4.1 数据预处理
```python
# 文本清洗和特征提取
def extract_features(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    features = []
    for line in lines:
        for level, score in RATING_MAP.items():
            if level in line:
                features.append(score)
                break
        else:
            features.append(3)  # 默认值
    return np.array(features)
```

3.4.2 特征工程
- 特征维度固定顺序存储
- 使用numpy构建特征矩阵
```python
X = np.array([extract_features(p) for p in file_paths])
```

3.4.3 PCA降维
```python
pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X)
print(f"解释方差比: {pca.explained_variance_ratio_}")
```

3.4.4 K-means聚类
```python
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42,
    max_iter=300,
    tol=1e-4
)
clusters = kmeans.fit_predict(X)
```

3.4.5 可视化实现
```python
plt.figure(figsize=(10, 6), dpi=300)
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.title('课程聚类分析结果', fontsize=14)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.savefig('course_clusters.png', bbox_inches='tight')
```

4. 使用指南
4.1 环境准备
- 安装Python 3.x
- 安装依赖库：
  pip install scikit-learn numpy matplotlib joblib

4.2 运行分析
- 执行命令：
  python course_profile_model.py

4.3 结果文件
- course_clusters.png: 课程聚类可视化图
- course_profile_model.pkl: 训练好的模型文件

5. 结果解读
- 可视化图中不同颜色代表不同课程类型
- 每个聚类的中心点代表该类课程的典型特征
- 可用于：
  * 识别优秀教学模式
  * 发现需要改进的课程
  * 评估教学改革效果

6. 进阶使用
- 修改course_profile_model.py中的n_clusters参数调整分类数量
- 可添加新数据继续训练模型
- 可扩展为课程质量评估系统

常见问题
--------
Q: 如何添加新的课程数据？
A: 在"数据集"文件夹中添加新的评价文本文件，保持相同格式

Q: 如何修改分类数量？
A: 编辑course_profile_model.py，修改build_model()函数中的n_clusters参数

联系方式
--------
如有问题请联系：your_email@example.com
