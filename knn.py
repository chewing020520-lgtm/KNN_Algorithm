import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 데이터 로드 및 전처리
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 메모리와 속도를 위해 데이터 일부만 사용 (샘플링)
num_training = 5000
num_test = 500
X_train = X_train[:num_training]
y_train = y_train[:num_training].flatten()
X_test = X_test[:num_test]
y_test = y_test[:num_test].flatten()

# KNN은 1차원 벡터를 입력으로 받으므로 3D 이미지를 1D로 평탄화(Flatten)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 2. Simple Train/Test Split (기본 성능 확인)
knn_simple = KNeighborsClassifier(n_neighbors=5)
knn_simple.fit(X_train_flat, y_train)
y_pred = knn_simple.predict(X_test_flat)

print("--- Simple Train/Test Split Result ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 3. 5-Fold Cross-Validation (최적의 k 찾기)
k_choices = [1, 3, 5, 8, 10, 12, 15, 20]
k_accuracy_averages = []
k_accuracy_stds = []

print("\n--- Performing 5-Fold Cross-Validation ---")
for k in k_choices:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5-fold 교차 검증 수행
    scores = cross_val_score(knn, X_train_flat, y_train, cv=5)
    k_accuracy_averages.append(np.mean(scores))
    k_accuracy_stds.append(np.std(scores))
    print(f"k = {k:2d}, Average Accuracy: {k_accuracy_averages[-1]:.4f}")

# 4. 성능 시각화 (Accuracy vs k)
plt.errorbar(k_choices, k_accuracy_averages, yerr=k_accuracy_stds, fmt='-o')
plt.title('5-fold Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation Accuracy')
plt.grid(True)
plt.show()

# 5. 최적의 k로 최종 평가 (Metrics Report)
best_k = k_choices[np.argmax(k_accuracy_averages)]
print(f"\nBest k found: {best_k}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_flat, y_train)
y_final_pred = knn_final.predict(X_test_flat)

# 성능 지표 리포트 (Precision, Recall, F1 등은 'macro' 평균 기준)
print("\n--- Final Performance Metrics (Test Set) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_final_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_final_pred, average='macro'):.4f}")
print(f"Recall:    {recall_score(y_test, y_final_pred, average='macro'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_final_pred, average='macro'):.4f}")