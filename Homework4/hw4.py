# -*- coding: utf-8 -*-
"""
SVM分类作业：Spambase垃圾邮件分类
参考课件SVM互动课件案例①
数据集：spambase.data （kaggle公开数据集，本地读取，不上传git）
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ---------------------- 1.读取数据 ----------------------
# spambase没有表头，共57特征+1标签，标签1=垃圾邮件，0=正常邮件
col_names = [f"feat_{i}" for i in range(57)] + ["label"]
df = pd.read_csv("spambase.data", names=col_names)

print("数据集大小：", df.shape)
print("标签分布：")
print(df["label"].value_counts(normalize=True).round(3))

X = df.drop("label", axis=1)
y = df["label"]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------- 2.标准化！SVM必须标准化【课件重点】 ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------3.网格搜索调参 ----------------------
## RBF核SVM
param_grid_rbf = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": [1e-4, 1e-3, 0.01, 0.1, 1]
}

print("\n======= RBF核SVM网格搜索开始 =======")
grid_rbf = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=param_grid_rbf,
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_rbf.fit(X_train_scaled, y_train)
best_rbf = grid_rbf.best_estimator_
print(f"RBF最佳参数：C={grid_rbf.best_params_['C']}, gamma={grid_rbf.best_params_['gamma']}")
print(f"交叉验证最佳得分：{grid_rbf.best_score_:.4f}")

## 线性核SVM
param_grid_linear = {"C":[0.01,0.1,1,10,100]}
print("\n======= 线性核SVM网格搜索开始 =======")
grid_linear = GridSearchCV(
    estimator=SVC(kernel="linear"),
    param_grid=param_grid_linear,
    cv=5,
    n_jobs=-1
)
grid_linear.fit(X_train_scaled, y_train)
best_linear = grid_linear.best_estimator_
print(f"Linear SVM最佳参数 C={grid_linear.best_params_['C']}")
print(f"交叉验证最佳得分：{grid_linear.best_score_:.4f}")

# ----------------------4.预测评估 ----------------------
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    sv_num = model.n_support_
    print(f"\n-------- {name} 测试集结果 --------")
    print(f"准确率Accuracy: {acc:.4f}")
    print(f"精确率Precision: {prec:.4f}")
    print(f"召回率Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"每类支持向量数量：{sv_num}")
    print(classification_report(y_test, y_pred))

evaluate(best_rbf, X_test_scaled, y_test, "RBF‑SVM")
evaluate(best_linear, X_test_scaled, y_test, "Linear‑SVM")

# ----------------------5.简单超参数定性实验（对应课件C与γ理论） ----------------------
print("\n========超参数C的影响实验========")
for c_test in [0.001, 1, 1000]:
    m = SVC(kernel="rbf", C=c_test, gamma=0.01)
    m.fit(X_train_scaled, y_train)
    tr_acc = m.score(X_train_scaled, y_train)
    te_acc = m.score(X_test_scaled, y_test)
    print(f"C={c_test:6} | 训练集acc:{tr_acc:.4f} | 测试集acc:{te_acc:.4f} | 支持向量数:{sum(m.n_support_)}")

print("\n========超参数gamma的影响实验========")
for g_test in [0.0001,0.01,10]:
    m = SVC(kernel="rbf", C=1, gamma=g_test)
    m.fit(X_train_scaled, y_train)
    tr_acc = m.score(X_train_scaled, y_train)
    te_acc = m.score(X_test_scaled, y_test)
    print(f"gamma={g_test:7} | 训练集acc:{tr_acc:.4f} | 测试集acc:{te_acc:.4f} | 支持向量数:{sum(m.n_support_)}")
