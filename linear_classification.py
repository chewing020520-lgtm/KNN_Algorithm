import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.datasets import cifar10

# 한글 폰트 설정 (Windows)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# CIFAR-10 클래스 이름
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ==========================================
# 1. 데이터 로드 및 평탄화 (Flatten)
# ==========================================
print(">>> CIFAR-10 데이터를 로드하는 중...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 훈련 속도를 위해 일부 샘플만 사용 (5000개 훈련, 500개 테스트)
num_training = 5000
num_test = 500
X_train = X_train[:num_training]
y_train = y_train[:num_training].flatten()
X_test = X_test[:num_test]
y_test = y_test[:num_test].flatten()

# 32x32x3 이미지를 3072차원의 열 벡터로 평탄화 (자료 Slide 7, 11 참고)
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0  # 정규화 포함
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

print(f"훈련 데이터 크기: {X_train_flat.shape}")  # (5000, 3072)
print(f"테스트 데이터 크기: {X_test_flat.shape}")  # (500, 3072)

# 무작위 가중치(W)와 편향(b) 초기화 (10개 클래스 x 3072 픽셀)
np.random.seed(42)
W = 0.001 * np.random.randn(10, 3072)
b = np.zeros((10, 1))

# ==========================================
# 2. 손실 함수 (Loss Functions) 구현
# ==========================================

def svm_hinge_loss(scores, y_true):
    """
    자료 Slide 16~18에 나온 다중 클래스 SVM Hinge Loss 계산 규칙 구현
    Li = Σ max(0, sj - sy_i + 1) (j != y_i)
    """
    num_classes = scores.shape[0]
    correct_class_score = scores[y_true]
    loss_i = 0

    for j in range(num_classes):
        if j == y_true:
            continue
        # 정답 클래스 스코어를 빼고 마진(1)을 더함
        margin = scores[j] - correct_class_score + 1
        if margin > 0:
            loss_i += margin
    return loss_i

def softmax_loss(scores, y_true):
    """
    자료 Slide 12~14에 나온 Softmax 확률 변환 및 Cross-Entropy Loss 구현
    1. 지수화(exp) -> 2. 정규화(normalize) -> 3. -log(정답 확률)
    """
    # 수치 안정성을 위해 최댓값을 빼줌 (Overflow 방지)
    scores -= np.max(scores)

    # 1. Exponentiate
    exp_scores = np.exp(scores)

    # 2. Normalize (Probabilities)
    probs = exp_scores / np.sum(exp_scores)

    # 3. -log P(Y=k|X)
    loss_i = -np.log(probs[y_true])
    return loss_i, probs

# ==========================================
# 3. 모델 실행 및 결과 확인
# ==========================================
print("\n>>> 선형 분류기 연산 시작 (f = Wx + b)...")

svm_losses = []
softmax_losses = []
predictions = []
all_probs = []

# 테스트 데이터 500개에 대해 연산 수행
for i in range(num_test):
    # x는 (3072, 1) 열 벡터 형태
    x = X_test_flat[i].reshape(-1, 1)

    # 순방향 전파: f(x, W) = Wx + b
    scores = np.dot(W, x) + b  # 결과는 (10, 1) 구조
    scores = scores.flatten()

    # 예측값 결정 (가장 높은 스코어를 가진 인덱스)
    predictions.append(np.argmax(scores))

    # 1) SVM Loss 계산
    l_svm = svm_hinge_loss(scores, y_test[i])
    svm_losses.append(l_svm)

    # 2) Softmax Loss 및 확률 계산
    l_softmax, probs = softmax_loss(scores, y_test[i])
    softmax_losses.append(l_softmax)
    all_probs.append(probs)

predictions = np.array(predictions)
all_probs = np.array(all_probs)

# 최종 평균 손실(Loss) 및 정확도(Accuracy) 출력
mean_svm_loss = np.mean(svm_losses)
mean_softmax_loss = np.mean(softmax_losses)
accuracy = np.mean(predictions == y_test)

print("\n==========================================")
print("             최종 모델 결과               ")
print("==========================================")
print(f"1) 전체 데이터셋 평균 SVM Hinge Loss: {mean_svm_loss:.4f}")
print(f"2) 전체 데이터셋 평균 Softmax Loss:     {mean_softmax_loss:.4f}")
print(f"3) 현재 초기 가중치 기준 예측 정확도:  {accuracy * 100:.2f}%")
print("==========================================")
print("※ 현재는 학습(Optimization) 전 초기 상태라 정확도가 낮게 나옵니다.")

# ==========================================
# 4. 시각화 (Visualization)
# ==========================================
print("\n>>> 그래프를 그리는 중...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Linear Classifier (Random Weights) - CIFAR-10 결과 분석', fontsize=16, fontweight='bold')

# ── 그래프 1: 샘플 이미지 예측 결과 (2행 5열 = 10장) ──────────────────────
print("  [1/4] 샘플 이미지 예측 결과...")
ax_imgs = [fig.add_subplot(4, 5, i + 1) for i in range(10)]

# 정답과 예측이 일치하는 것 / 틀린 것 각 5장씩 선택
correct_idx = np.where(predictions == y_test)[0]
wrong_idx   = np.where(predictions != y_test)[0]
sample_idx  = np.concatenate([correct_idx[:5], wrong_idx[:5]])

for ax, idx in zip(ax_imgs, sample_idx):
    ax.imshow(X_test[idx])
    pred_label  = CLASS_NAMES[predictions[idx]]
    true_label  = CLASS_NAMES[y_test[idx]]
    is_correct  = predictions[idx] == y_test[idx]
    color       = 'green' if is_correct else 'red'
    ax.set_title(f'예측: {pred_label}\n정답: {true_label}', fontsize=7, color=color)
    ax.axis('off')

# 상단 라벨
fig.text(0.18, 0.97, '✓ 정답 (상위 5개)', ha='center', color='green', fontsize=9)
fig.text(0.58, 0.97, '✗ 오답 (상위 5개)', ha='center', color='red',   fontsize=9)

# ── 그래프 2: SVM / Softmax Loss 분포 히스토그램 ──────────────────────────
print("  [2/4] Loss 분포 히스토그램...")
ax2 = fig.add_subplot(4, 2, 3)
ax2.hist(svm_losses, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
ax2.axvline(mean_svm_loss, color='red', linestyle='--', linewidth=1.5,
            label=f'평균: {mean_svm_loss:.2f}')
ax2.set_title('SVM Hinge Loss 분포')
ax2.set_xlabel('Loss')
ax2.set_ylabel('샘플 수')
ax2.legend()

ax3 = fig.add_subplot(4, 2, 4)
ax3.hist(softmax_losses, bins=40, color='darkorange', edgecolor='white', alpha=0.85)
ax3.axvline(mean_softmax_loss, color='red', linestyle='--', linewidth=1.5,
            label=f'평균: {mean_softmax_loss:.2f}')
ax3.set_title('Softmax Cross-Entropy Loss 분포')
ax3.set_xlabel('Loss')
ax3.set_ylabel('샘플 수')
ax3.legend()

# ── 그래프 3: 클래스별 정확도 막대 그래프 ────────────────────────────────
print("  [3/4] 클래스별 정확도...")
ax4 = fig.add_subplot(4, 2, (5, 6))
per_class_acc = []
per_class_count = []
for c in range(10):
    mask = (y_test == c)
    acc_c = np.mean(predictions[mask] == y_test[mask]) if mask.sum() > 0 else 0.0
    per_class_acc.append(acc_c * 100)
    per_class_count.append(mask.sum())

colors = ['#4CAF50' if a >= accuracy * 100 else '#F44336' for a in per_class_acc]
bars = ax4.bar(CLASS_NAMES, per_class_acc, color=colors, edgecolor='white', width=0.6)
ax4.axhline(accuracy * 100, color='navy', linestyle='--', linewidth=1.5,
            label=f'전체 평균 정확도: {accuracy*100:.2f}%')
ax4.set_title('클래스별 예측 정확도')
ax4.set_xlabel('클래스')
ax4.set_ylabel('정확도 (%)')
ax4.set_ylim(0, 35)
ax4.legend()
ax4.tick_params(axis='x', rotation=25)
for bar, acc in zip(bars, per_class_acc):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

# ── 그래프 4: Confusion Matrix ────────────────────────────────────────────
print("  [4/4] Confusion Matrix...")
ax5 = fig.add_subplot(4, 2, (7, 8))
conf_mat = np.zeros((10, 10), dtype=int)
for true, pred in zip(y_test, predictions):
    conf_mat[true][pred] += 1

im = ax5.imshow(conf_mat, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
ax5.set_xticks(range(10))
ax5.set_yticks(range(10))
ax5.set_xticklabels(CLASS_NAMES, rotation=40, ha='right', fontsize=8)
ax5.set_yticklabels(CLASS_NAMES, fontsize=8)
ax5.set_title('Confusion Matrix (행=정답, 열=예측)')
ax5.set_xlabel('예측 클래스')
ax5.set_ylabel('실제 클래스')

# 셀 안에 숫자 표시
thresh = conf_mat.max() / 2
for i in range(10):
    for j in range(10):
        ax5.text(j, i, str(conf_mat[i, j]),
                 ha='center', va='center', fontsize=7,
                 color='white' if conf_mat[i, j] > thresh else 'black')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('linear_classification_result.png', dpi=150, bbox_inches='tight')
print("\n>>> 그래프 저장 완료: linear_classification_result.png")
plt.show()
print(">>> 완료!")
