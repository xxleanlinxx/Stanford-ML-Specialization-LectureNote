# Stanford Machine Learning Specialization - Lecture Notes

[![Course](https://img.shields.io/badge/Coursera-DeepLearning.AI-0056D2?style=flat&logo=coursera)](https://www.coursera.org/specializations/machine-learning-introduction)
[![Instructor](https://img.shields.io/badge/Instructor-Andrew%20Ng-red?style=flat)](https://www.andrewng.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Comprehensive lecture notes for the Stanford Machine Learning Specialization by Andrew Ng on Coursera (DeepLearning.AI)

## 📖 About

This repository contains detailed lecture notes covering all three courses of the Stanford Machine Learning Specialization. The notes are structured in Markdown format and optimized for use with [Obsidian](https://obsidian.md/), featuring extensive cross-references and a comprehensive knowledge graph.

**Total Duration**: 10 Weeks | **3 Courses** | **10 Weekly Modules**

## 🎯 Course Overview

### Course 1: Supervised Machine Learning - Regression and Classification

Foundational concepts in supervised learning, including linear regression, logistic regression, and fundamental optimization techniques.

**Topics Covered**:
- Introduction to Machine Learning (supervised/unsupervised learning)
- Linear Regression & Cost Functions
- Gradient Descent & Optimization
- Multiple Feature Regression & Vectorization
- Feature Scaling & Polynomial Regression
- Logistic Regression & Classification
- Decision Boundaries & Regularization

### Course 2: Advanced Learning Algorithms

Deep dive into neural networks, modern optimization techniques, and practical ML engineering.

**Topics Covered**:
- Neural Network Architecture & Forward Propagation
- TensorFlow Implementation
- Activation Functions (ReLU, Softmax)
- Backpropagation & Adam Optimizer
- Bias-Variance Tradeoff
- Train/CV/Test Split Strategies
- Error Analysis & Transfer Learning
- Decision Trees, Random Forests & XGBoost

### Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning

Advanced ML techniques for unsupervised scenarios and sequential decision-making.

**Topics Covered**:
- K-Means Clustering
- Anomaly Detection & Gaussian Density Estimation
- Collaborative Filtering & Content-Based Filtering
- Recommender System Architectures (Two-Tower Networks)
- Principal Component Analysis (PCA)
- Reinforcement Learning Fundamentals
- Markov Decision Processes (MDP)
- Q-Learning & Deep Q-Networks (DQN)

## 📁 Repository Structure

```
Stanford-ML-Specialization-LectureNote/
├── lecture_notes/
│   ├── Course 1 - Supervised Machine Learning/
│   │   ├── C1-W1 - Introduction to Machine Learning.md
│   │   ├── C1-W2 - Regression with Multiple Input Variables.md
│   │   ├── C1-W3 - Classification.md
│   │   └── Course 1 - Index.md
│   ├── Course 2 - Advanced Learning Algorithms/
│   │   ├── C2-W1 - Neural Networks.md
│   │   ├── C2-W2 - Neural Network Training.md
│   │   ├── C2-W3 - Advice for Applying ML.md
│   │   ├── C2-W4 - Decision Trees.md
│   │   └── Course 2 - Index.md
│   ├── Course 3 - Unsupervised Learning, Recommenders, Reinforcement Learning/
│   │   ├── C3-W1 - Clustering & Anomaly Detection.md
│   │   ├── C3-W2 - Recommender Systems & PCA.md
│   │   ├── C3-W3 - Reinforcement Learning.md
│   │   └── Course 3 - Index.md
│   ├── Knowledge Points/
│   │   ├── KP-01 - 超參數與學習率.md
│   │   ├── KP-02 - 現代優化器.md
│   │   ├── KP-03 - 損失函數.md
│   │   ├── KP-04 - 正則化技術.md
│   │   ├── KP-05 - 激活函數.md
│   │   ├── KP-06 - Attention 機制與 Transformer.md
│   │   ├── KP-07 - 縮放法則與湧現能力.md
│   │   ├── KP-08 - 自監督與對比學習.md
│   │   ├── KP-09 - RLHF 與現代強化學習.md
│   │   ├── KP-10 - 現代推薦系統.md
│   │   ├── KP-11 - 表格資料與現代決策樹.md
│   │   └── KP-Index - 知識點總索引.md
│   └── ML Specialization - Master Index.md
├── .obsidian/
│   └── (Obsidian workspace configuration)
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Recommended**: [Obsidian](https://obsidian.md/) (free, cross-platform note-taking app)
- Alternative: Any Markdown viewer/editor

### Usage with Obsidian

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xxleanlinxx/Stanford-ML-Specialization-LectureNote.git
   cd Stanford-ML-Specialization-LectureNote
   ```

2. **Open in Obsidian**:
   - Launch Obsidian
   - Click "Open folder as vault"
   - Select this repository folder

3. **Start with the Master Index**:
   - Navigate to `lecture_notes/ML Specialization - Master Index.md`
   - This contains the complete knowledge graph and navigation links

### Usage without Obsidian

Simply browse the markdown files in the `lecture_notes/` directory. Note that some Obsidian-specific features (like `[[WikiLinks]]`) may not render correctly in standard Markdown viewers.

## 🗺️ Knowledge Graph

The notes are organized with extensive cross-references following the Map of Content (MOC) methodology:

- **Master Index**: `ML Specialization - Master Index.md` - Complete course overview with Mermaid diagram
- **Course Indexes**: Each course has its own index with weekly breakdown
- **Knowledge Points**: Supplementary deep-dive documents on modern ML concepts
- **Cross-References**: Internal links connect related concepts across courses

## 📊 Key Algorithms Covered

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Linear Regression | Supervised/Regression | Continuous value prediction |
| Logistic Regression | Supervised/Classification | Binary classification |
| Neural Networks | Supervised | Complex tasks (images, text) |
| Decision Trees/XGBoost | Supervised/Ensemble | Tabular data competitions |
| K-Means | Unsupervised/Clustering | Market segmentation |
| Anomaly Detection | Unsupervised | Fraud detection, failure prediction |
| Collaborative Filtering | Unsupervised/Recommendation | Recommender systems |
| PCA | Unsupervised/Dimensionality Reduction | Visualization, compression |
| Deep Q-Network (DQN) | Reinforcement Learning | Game AI, robotics |

## 🔑 Core Concepts Index

### Optimization & Training
- Gradient Descent
- Adam Optimizer
- Backpropagation
- Learning Rate Scheduling

### Model Evaluation
- Bias-Variance Tradeoff
- Cross-Validation
- Precision, Recall & F1 Score
- Error Analysis

### Regularization
- L2 Regularization (Ridge)
- Regularization effects on Bias/Variance
- Dropout & Early Stopping

### Modern Extensions
- Attention Mechanisms & Transformers
- Self-Supervised Learning
- RLHF (Reinforcement Learning from Human Feedback)
- Scaling Laws & Emergent Abilities

## 🛠️ Technical Stack Mentioned

- **Frameworks**: TensorFlow, scikit-learn
- **Libraries**: NumPy, Pandas
- **Algorithms**: XGBoost, LightGBM
- **Concepts**: Vectorization, Matrix Computation

## 📝 Note-Taking Methodology

These notes follow best practices for technical learning:
- **Progressive Disclosure**: Start with intuition, then dive into mathematics
- **Cross-References**: Connect related concepts across courses
- **Code Examples**: Include practical implementation notes
- **Visual Aids**: Mermaid diagrams for concept relationships
- **Knowledge Points**: Supplementary documents bridging course content with modern research

## 🤝 Contributing

Contributions are welcome! If you find errors or want to add supplementary notes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (follow Conventional Commits format)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Andrew Ng** - Course Instructor
- **DeepLearning.AI** - Course Provider
- **Coursera** - Platform

## 📬 Contact

For questions or suggestions, please open an issue in this repository.

---

**Note**: These are personal lecture notes and are not officially affiliated with Stanford University, DeepLearning.AI, or Coursera. All course content belongs to the respective copyright holders.
