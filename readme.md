# Movie Rating Based Collaborative Filtering Recommendation System  
# 基于电影评分的协同过滤推荐系统  

## 📖 Project Overview | 项目概述  
This implementation builds UserCF and ItemCF recommendation systems based on the MovieLens dataset, containing data preprocessing, similarity calculation, recommendation generation and optimization strategies.  
基于MovieLens数据集构建用户协同过滤(UserCF)和物品协同过滤(ItemCF)推荐系统，包含数据预处理、相似度计算、推荐生成及优化策略。

---

## 📚 Dataset | 数据集  
- **Source**: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)  
- **Statistics**:  
  - 100,836 ratings  
  - 610 users  
  - 9,742 movies  
  - Time span: 1996-2018  

  - **来源**: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)  
  - **统计**:  
    - 100,836 条评分  
    - 610 个用户  
    - 9,742 部电影  
    - 时间跨度: 1996-2018  

---

## 🛠️ Features | 功能实现  

### UserCF Module | 用户协同过滤模块  
```python
# Core Pipeline
1. Build user-movie rating matrix
2. Calculate improved Pearson similarity
3. Filter Top-K similar users
4. Generate recommendations with weighted scores

# 核心流程
1. 构建用户-电影评分矩阵
2. 计算改进的皮尔逊相似度
3. 筛选Top-K相似用户
4. 加权生成推荐列表

### ItemCF Module | 物品协同过滤模块  
```python
# Core Pipeline
1. Build item-user inverted index
2. Calculate cosine similarity matrix
3. Recommend based on user history
4. Popularity penalty mechanism

# 核心流程
1. 构建物品-用户倒排表
2. 计算余弦相似度矩阵
3. 基于用户历史行为推荐
4. 热门物品惩罚机制
```

---

## 🚀 Quick Start | 快速开始  
### Requirements | 环境要求  
```bash
Python 3.8+
pandas==1.3.4
numpy==1.21.2
```

### Demo | 运行示例  
```python
# UserCF Recommendation
>>> generate_recommendations(1)
['The Shawshank Redemption (1994)', 'The Godfather (1972)', 'Forrest Gump (1994)'...]

# UserCF推荐
>>> generate_recommendations(1)
['肖申克的救赎 (1994)', '教父 (1972)', '阿甘正传 (1994)'...]
```

---

## 📈 Performance | 性能指标  
| Algorithm | MAE | RMSE | Coverage | Time Cost |  
|-----------|-----|------|----------|-----------|  
| UserCF    | 0.68 | 0.89 | 38%      | 1.57s     |  
| ItemCF    | 0.72 | 0.91 | 42%      | 0.89s     |  

| 算法     | 平均绝对误差 | 均方根误差 | 覆盖率 | 耗时  |  
|----------|------------|------------|-------|-------|  
| UserCF   | 0.68       | 0.89       | 38%   | 1.57s |  
| ItemCF   | 0.72       | 0.91       | 42%   | 0.89s |  

---

## 🎯 Optimization Strategies | 优化策略  
1. **Similarity Calculation Optimization**  
   - Popularity penalty factor: `1/log(1+N(i))`  
   - Dynamic similarity threshold (0.2~0.5)  

2. **Cold Start Handling**  
   ```python
   # Hybrid strategy
   Cold Start = 60% global popular + 30% recent popular + 10% random
   ```

3. **Efficiency Improvement**  
   - Inverted index acceleration  
   - Sparse similarity matrix storage  

1. **相似度计算优化**  
   - 引入流行度惩罚因子: `1/log(1+N(i))`  
   - 动态相似用户阈值 (0.2~0.5)  

2. **冷启动处理**  
   ```python
   # 混合推荐策略
   冷启动推荐 = 60%全局热门 + 30%近期热门 + 10%随机
   ```

3. **效率优化**  
   - 倒排表索引加速  
   - 相似度矩阵稀疏化存储  

---

## 📂 File Structure | 文件结构  
```
recommender-system/
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── UserCF.ipynb
├── ItemCF.ipynb
└── utils/
    ├── preprocessor.py
    └── evaluator.py
```

---

## 📌 Key Implementations | 关键实现  
### Improved Pearson Similarity | 改进的皮尔逊相似度  
```python
def improved_pearson(u1, u2):
    common = ratings_df.loc[u1].notna() & ratings_df.loc[u2].notna()
    if sum(common) < 5: return 0
    return np.corrcoef(ratings_df.loc[u1,common], ratings_df.loc[u2,common])[0,1]
```

### Weighted Recommendation Formula | 加权推荐公式  
$$
\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N(u)} sim(u,v)(r_{v,i}-\bar{r}_v)}{\sum_{v \in N(u)} |sim(u,v)|}
$$

---

## 📚 References | 参考文献  
1. [Recommender System Practice](https://book.douban.com/subject/10769749/)  
2. [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)  
3. [recommenderlab Framework](https://cran.r-project.org/web/packages/recommenderlab/)  

1. [推荐系统实践](https://book.douban.com/subject/10769749/)  
2. [矩阵分解推荐技术](https://ieeexplore.ieee.org/document/5197422)  
3. [recommenderlab框架](https://cran.r-project.org/web/packages/recommenderlab/)  

---

## 🛠️ Future Improvements | 后续改进  
- [ ] Integrate deep learning models  
- [ ] Build multi-dimensional evaluation system  

- [ ] 集成深度学习模型  
- [ ] 构建多维度评估体系  

> 🚧 Project under active development... Contributions welcome!  
> 🚧 项目持续开发中... 欢迎贡献代码和建议！
``` 
