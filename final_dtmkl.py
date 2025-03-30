# ============== 1. 依赖库导入 ==============
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.preprocessing import Normalizer

solvers.options['show_progress'] = False

# ============== 2. 特征处理器类 ==============
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from sentence_transformers import SentenceTransformer
from feature_extractor import BoWFeatureExtractor, FeatureExtractor, BetterFeatureExtractor
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 分词
    words = nltk.word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # 词干提取
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # 合并并去除多余空格
    return ' '.join(words).strip()


class FeatureProcessor:
    """确保训练/测试使用相同的特征转换器"""

    def __init__(self, max_features=15000, svd_dim=5000):
        # self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        # self.svd = TruncatedSVD(n_components=svd_dim)
        # self.pipeline = Pipeline([
        #     ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
        #     ('svd', TruncatedSVD(svd_dim))
        # ])
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('normalizer', Normalizer(norm='l2'))
        ])

    def fit_transform(self, texts):
        return self.pipeline.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.pipeline.transform(texts).toarray()


# ============== 3. 数据加载函数 ==============
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
import os
from sklearn.utils import shuffle


def load_data_ori(config, m=5, random_state=42):
    """返回原始文本数据（未转换）"""
    # 加载辅助域数据
    aux_pos = fetch_20newsgroups(
        subset='train',
        categories=config["auxiliary"]["positive"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    aux_pos2 = fetch_20newsgroups(
        subset='test',
        categories=config["auxiliary"]["positive"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )

    aux_neg = fetch_20newsgroups(
        subset='train',
        categories=config["auxiliary"]["negative"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    aux_neg2 = fetch_20newsgroups(
        subset='test',
        categories=config["auxiliary"]["negative"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    aux_pos.data = [clean_text(text) for text in aux_pos.data]
    aux_neg.data = [clean_text(text) for text in aux_neg.data]

    X_aux_text = aux_pos.data + aux_pos2.data + aux_neg.data + aux_neg2.data
    y_aux = np.array([1] * (len(aux_pos.data) + len(aux_pos2.data)) + [-1] * (len(aux_neg.data) + len(aux_neg2.data)))

    # 加载目标域全量数据
    target_pos = fetch_20newsgroups(
        subset='train',
        categories=config["target"]["positive"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    target_neg = fetch_20newsgroups(
        subset='train',
        categories=config["target"]["negative"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    target_pos.data = [clean_text(text) for text in target_pos.data]
    target_neg.data = [clean_text(text) for text in target_neg.data]
    # 随机抽取m个正/负样本作为标记数据
    np.random.seed(random_state)
    pos_idx = np.random.choice(len(target_pos.data), m, replace=False)
    neg_idx = np.random.choice(len(target_neg.data), m, replace=False)

    # 目标域标记文本和标签
    X_target_labeled_text = [target_pos.data[i] for i in pos_idx] + \
                            [target_neg.data[i] for i in neg_idx]
    y_target_labeled = np.array([1] * m + [-1] * m)

    # 目标域未标记文本（剩余样本）
    X_target_unlabeled_text = [
                                  target_pos.data[i] for i in range(len(target_pos.data)) if i not in pos_idx
                              ] + [
                                  target_neg.data[i] for i in range(len(target_neg.data)) if i not in neg_idx
                              ]

    # 测试集数据
    X_test_text = X_target_unlabeled_text
    y_test = np.array([1] * (len(target_pos.data) - m) + [-1] * (len(target_neg.data) - m))

    # 合并目标域所有文本
    X_target_text = X_target_labeled_text + X_target_unlabeled_text

    return X_aux_text, y_aux, X_target_text, y_target_labeled, X_test_text, y_test


def load_data(config, m=5, random_state=42):
    """从本地20news-bydate文件夹加载数据"""
    # 定义数据路径
    base_path = "20news-bydate"
    train_path = os.path.join(base_path, "20news-bydate-train")
    test_path = os.path.join(base_path, "20news-bydate-test")

    # 加载辅助域数据（从训练集）
    def load_category(categories, subset='train'):
        data = []
        for category in categories:
            dir_path = os.path.join(train_path if subset == 'train' else test_path, category)
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.isdigit()]
            for file_path in files:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                    text = clean_text(text)
                    data.append(text)
        return data

    # 辅助域正负样本
    aux_pos = load_category(config["auxiliary"]["positive"], subset='train')
    aux_neg = load_category(config["auxiliary"]["negative"], subset='train')
    X_aux_text = aux_pos + aux_neg
    y_aux = np.array([1] * len(aux_pos) + [-1] * len(aux_neg))

    # 目标域全量数据（从训练集）
    target_pos = load_category(config["target"]["positive"], subset='train')
    target_neg = load_category(config["target"]["negative"], subset='train')

    # 随机抽取m个正/负样本作为标记数据
    np.random.seed(random_state)
    pos_idx = np.random.choice(len(target_pos), m, replace=False)
    neg_idx = np.random.choice(len(target_neg), m, replace=False)

    # 目标域标记文本和标签
    X_target_labeled_text = [target_pos[i] for i in pos_idx] + [target_neg[i] for i in neg_idx]
    y_target_labeled = np.array([1] * m + [-1] * m)

    # 目标域未标记文本（剩余样本）
    X_target_unlabeled_text = [
                                  target_pos[i] for i in range(len(target_pos)) if i not in pos_idx
                              ] + [
                                  target_neg[i] for i in range(len(target_neg)) if i not in neg_idx
                              ]

    # 测试集数据
    X_test_text = X_target_unlabeled_text
    y_test = np.array([1] * (len(target_pos) - m) + [-1] * (len(target_neg) - m))

    # 合并目标域所有文本
    X_target_text = X_target_labeled_text + X_target_unlabeled_text

    return X_aux_text, y_aux, X_target_text, y_target_labeled, X_test_text, y_test


# ============== 4. DTMKL_f模型类 ==============
class DTMKL_f():
    """使用现有基分类器的DTMKL_f"""

    def __init__(self, C=1.0, lr=0.05, zeta=0.1, theta=1e-5, max_iter=6, lambda_=0.5, tol=1e-3):
        self.base_kernels = []  # 基核列表
        self.C = C  # SVM正则化参数
        self.zeta = zeta  # 目标域标记数据正确和未标记数据上分类器相似度正则项权重
        self.theta = theta  # J(d)正则项权重
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.d = None  # 核组合系数
        self.svr = None  # SVR
        self.lr = lr  # SVR学习率
        self.base_classifiers = None
        self.lambda_ = lambda_  # 未标记数据上分类器相似度正则项权重

    def generate_base_kernels(self, X_aux, X_target):
        """生成多个基核矩阵（线性核、多项式核等）"""
        base_kernels = []
        # 线性核
        K_linear = linear_kernel(np.concatenate([X_aux, X_target], axis=0))
        base_kernels.append(K_linear)
        # 多项式核（不同次数）
        for degree in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            K_poly = polynomial_kernel(np.concatenate([X_aux, X_target], axis=0), degree=degree)
            base_kernels.append(K_poly)
        return base_kernels

    def base_classifier_fit(self, y):
        base_classifiers = []
        for index, K_m in enumerate(self.base_kernels):
            svm = SVR(kernel='precomputed', C=self.C)
            svm.fit(K_m[:len(y), :len(y)], y)
            base_classifiers.append(svm)
        return base_classifiers

    def _compute_mmd(self, K, s):
        """K: (n_total, n_total), s: (n_total,)"""
        return np.trace(K @ np.outer(s, s))

    def fit(self, X_aux, y_aux, X_labeled, y_labeled, X_unlabeled):
        X_target = np.concatenate([X_labeled, X_unlabeled], axis=0)
        X_train = np.concatenate([X_aux, X_labeled], axis=0)
        y_train = np.concatenate([y_aux, y_labeled], axis=0)
        self.base_kernels = self.generate_base_kernels(X_aux, X_target)

        self.base_classifiers = self.base_classifier_fit(y_train)

        X_target_combined = np.vstack([X_labeled, X_unlabeled])
        n_A = X_aux.shape[0]
        n_Tl = X_labeled.shape[0]
        n_Tu = X_unlabeled.shape[0]
        self.n_A = n_A
        self.n_Tl = n_Tl
        self.n_Tu = n_Tu
        self.n_T = self.n_Tu + self.n_Tl
        n_T = n_Tl + n_Tu
        n_labeled = X_labeled.shape[0]
        n_unlabeled = X_unlabeled.shape[0]
        n_train = X_aux.shape[0] + X_labeled.shape[0]
        M = len(self.base_classifiers)
        self.d = np.ones(M) / M

        K_combined = sum(self.d[m] * self.base_kernels[m] for m in range(M))

        # 计算MMD向量 p = [tr(K_m S)]
        s = np.concatenate([np.ones(n_A) / n_A, -np.ones(n_T) / n_T])
        S = np.outer(s, s)

        self.base_classifiers = self.base_classifier_fit(y_train)

        # 迭代优化
        for iter in range(self.max_iter):
            # 预计算基分类器在未标记数据上的决策值
            f_base = np.zeros((n_unlabeled, M))

            #kernel_test = np.dot(X_unlabeled, X_train.T)
            for m in range(M):
                K_test = self.base_kernels[m][n_train:, :n_train]  # 未标记数据与训练数据的核矩阵块
                f_base[:, m] = self.base_classifiers[m].predict(K_test)
                #f_base[:, m] = self.base_classifiers[m].predict(kernel_test)

            y_virtual = f_base @ self.d

            mmd_values = [np.trace(K_m @ S) for K_m in self.base_kernels]

            p = np.array(mmd_values)
            #K_combined = np.einsum('m,mij->ij', self.d, self.base_kernels)
            #mmd = self._compute_mmd(K_combined, s)

            # 步骤1：训练SVR（使用标记数据）
            combined_kernel = sum(self.d[m] * self.base_kernels[m] for m in range(M))
            self.svr = SVR(kernel='precomputed', C=self.C, epsilon=0.1)
            self.svr.fit(combined_kernel, np.concatenate([y_train, y_virtual], axis=0))

            # 步骤2：计算梯度并更新d
            alpha_diff = self.svr.dual_coef_  # 形状 (1, n_SV)
            alpha_sum = np.abs(alpha_diff).sum()

            grad_J = self._compute_grad_J_new(alpha_diff, alpha_sum, f_base)
            grad_mmd = p.T @ p * self.d  # 同DTMKL_AT的MMD梯度计算
            grad_total = grad_mmd + self.theta * grad_J

            self.d -= self.lr * grad_total

            self.d = self._project_to_simplex(self.d)
            #print(f"Iteration {iter + 1}: d = {self.d}")

        combined_kernel = sum(self.d[m] * self.base_kernels[m] for m in range(M))
        self.svr.fit(combined_kernel[:n_A + n_Tl, :n_A + n_Tl], y_train)

    def _compute_mmd_grad(self, mmd, s):
        grad_mmd = np.zeros(len(self.base_classifiers))
        for m in range(len(self.base_classifiers)):
            grad_mmd[m] = 1 / 2 * mmd * np.trace(self.base_kernels[m] @ np.outer(s, s))
        return grad_mmd

    def _project_to_simplex(self, d):
        """投影到单纯形约束：d >=0, sum(d)=1"""
        d = np.where(d < 0, 1 / len(d), d)
        d_sorted = np.sort(d)[::-1]
        cum_sum = np.cumsum(d_sorted) - 1
        idx = np.arange(1, len(d) + 1)
        rho = np.where(d_sorted - cum_sum / idx > 0)[0][-1]
        theta = cum_sum[rho] / (rho + 1)
        return np.maximum(d - theta, 0)

    def generate_diagonal_matrix(self, n, u, lambda_value):
        # 矩阵的总大小
        size = n + u
        # 前n行是1，后u行是1/lambda
        diagonal_values = [1] * n + [1 / lambda_value] * u
        # 生成对角矩阵
        matrix = np.diag(diagonal_values)
        return matrix

    def _compute_grad_J_new(self, alpha_diff, alpha_sum, f_base):
        sv_indices = self.svr.support_
        grad = np.zeros(len(self.base_kernels))
        for m in range(len(self.base_kernels)):
            K_m_grad = self.base_kernels[m]
            K_grad_matrix = self.generate_diagonal_matrix(self.n_A + self.n_Tl, self.n_Tu, lambda_value=self.lambda_)
            K_grad = 1 / self.zeta * 2 * self.d[m] * K_grad_matrix
            K_hat_grad = K_m_grad + K_grad
            K_hat_grad_sv = K_hat_grad[sv_indices][:, sv_indices]
            first_term = -0.5 * alpha_diff @ K_hat_grad_sv @ alpha_diff.T

            y_hat = np.concatenate([np.zeros(self.n_A + self.n_Tl), f_base[:, m]], axis=0)
            second_term = - alpha_diff @ y_hat[sv_indices]
            grad[m] = first_term + second_term

        return grad

    def _compute_grad_J_svr(self, alpha, K_combined, y_aux, y_labeled, f_base):
        """计算SVR结构风险项的梯度"""
        grad = np.zeros(len(self.base_kernels))
        n_labeled = len(y_labeled)
        n_aux = len(y_aux)
        sv_indices = self.svr.support_  # 支持向量的索引

        # 标记数据梯度项
        for m in range(len(self.base_kernels)):
            K_m_labeled = self.base_kernels[m][n_aux:n_aux + n_labeled, n_aux:n_aux + n_labeled]
            K_m_sv = K_m_labeled[sv_indices][:, sv_indices]
            grad[m] = - 0.5 * alpha @ K_m_sv @ alpha.T / 2 / self.d[m] / self.d[m]

        # 未标记数据正则项梯度
        y_virtual = f_base @ self.d
        for m in range(len(self.base_kernels)):
            K_m_unlabeled = self.base_kernels[m][n_aux + n_labeled:, n_aux + n_labeled:]
            residual = y_virtual - self.svr.predict(K_combined[n_aux + n_labeled:, n_aux:n_aux + n_labeled])
            grad[m] += self.lambda_ * self.d[m] * np.dot(residual, K_m_unlabeled @ residual)
        return grad

    def predict(self, n_labeled, n_aux):
        """预测测试集"""

        combined_kernel_test = sum(self.d[m] * self.base_kernels[m] for m in range(len(self.base_kernels)))
        return self.svr.predict(combined_kernel_test[n_aux + n_labeled:, :n_aux + n_labeled])


# ============== extra. 对比模块 ==============
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
class FeatureReplicationSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.clf = SVC(C=C, kernel='linear')

    def _expand_features(self, X, domain='aux'):
        """特征扩展：辅助域[原特征, 原特征, 0], 目标域[原特征, 0, 原特征]"""
        n_samples, n_features = X.shape
        if domain == 'aux':
            return np.hstack([X, X, np.zeros((n_samples, n_features))])
        elif domain == 'target':
            return np.hstack([X, np.zeros((n_samples, n_features)), X])

    def fit(self, X_aux, y_aux, X_target_labeled, y_target_labeled):
        # 合并数据并扩展特征
        X_aux_expanded = self._expand_features(X_aux, domain='aux')
        X_target_expanded = self._expand_features(X_target_labeled, domain='target')
        X_combined = np.vstack([X_aux_expanded, X_target_expanded])
        y_combined = np.concatenate([y_aux, y_target_labeled])

        self.clf.fit(X_combined, y_combined)

    def predict(self, X_test):
        X_test_expanded = self._expand_features(X_test, domain='target')
        return self.clf.predict(X_test_expanded)


from sklearn.svm import SVC
class AdaptiveSVM:
    def __init__(self, C=1.0):
        self.base_clf = SVC(C=C, kernel='linear', probability=True)  # 启用概率输出
        self.delta_clf = SVR(C=C, kernel='linear')  # 使用回归模型拟合连续残差

    def fit(self, X_aux, y_aux, X_target_labeled, y_target_labeled):
        # 训练基分类器
        self.base_clf.fit(X_aux, y_aux)

        # 获取基分类器在目标域标记数据上的概率预测（映射到[-1, 1]）
        f_base_prob = self.base_clf.predict_proba(X_target_labeled)[:, 1] * 2 - 1

        # 计算残差（连续值）
        residual = y_target_labeled - f_base_prob

        # 训练调整模型
        self.delta_clf.fit(X_target_labeled, residual)

    def predict(self, X_test):
        # 基分类器的概率预测
        base_prob = self.base_clf.predict_proba(X_test)[:, 1] * 2 - 1

        # 调整项的预测
        delta = self.delta_clf.predict(X_test)

        # 综合预测
        return np.sign(base_prob + delta)

from sklearn.neighbors import NearestNeighbors


class CDSVM:
    def __init__(self, C=1.0, k=5):
        self.C = C
        self.k = k
        self.clf = SVC(C=C, kernel='linear', class_weight='balanced')

    def fit(self, X_aux, y_aux, X_target_labeled, y_target_labeled):
        # 合并所有目标域标记数据作为参考
        X_target_all = np.vstack([X_target_labeled])

        # 计算辅助域样本的权重：与目标域的相似度
        if len(y_target_labeled) == 2:
            knn = NearestNeighbors(n_neighbors=1).fit(X_target_all)
        else:
            knn = NearestNeighbors(n_neighbors=self.k).fit(X_target_all)
        distances, _ = knn.kneighbors(X_aux)
        weights = 1.0 / (np.mean(distances, axis=1) + 1e-6)
        weights /= np.max(weights)  # 归一化

        # 合并数据并加权训练
        X_train = np.vstack([X_aux, X_target_labeled])
        y_train = np.concatenate([y_aux, y_target_labeled])
        sample_weight = np.concatenate([weights, np.ones(len(y_target_labeled))])

        self.clf.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.clf.predict(X_test)


from cvxopt import matrix, solvers


class KMM:
    def __init__(self, kernel='rbf', gamma=0.1, B=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.B = B  # 权重边界

    def fit(self, X_aux, X_target_labeled, X_target_unlabeled):
        # 合并目标域数据（标记+未标记）
        X_target = np.vstack([X_target_labeled, X_target_unlabeled])

        # 计算核矩阵
        K = rbf_kernel(X_aux, X_target, gamma=self.gamma)
        n_aux = X_aux.shape[0]
        n_target = X_target.shape[0]

        # 优化目标：最小化 ||Phi(aux)*beta - Phi(target)||
        # 转换为二次规划问题
        K_aux = rbf_kernel(X_aux, gamma=self.gamma)
        K_target = rbf_kernel(X_target, gamma=self.gamma)

        P = matrix(K_aux)
        q = -matrix(np.mean(K, axis=1))
        G = matrix(np.vstack([-np.eye(n_aux), np.eye(n_aux)]))  # 0 <= beta_i <= B
        h = matrix(np.hstack([np.zeros(n_aux), np.ones(n_aux) * self.B]))
        A = matrix(np.ones((1, n_aux)), (1, n_aux))
        b = matrix(n_target * 1.0, (1, 1))

        sol = solvers.qp(P, q, G, h, A, b)
        self.beta = np.array(sol['x']).flatten()

    def get_weights(self):
        return self.beta


# ================== extra. 绘图 ==================
def draw_pic(models, results, C_values,m):
    # ================== 1. 数据 ==================
    if m != 5 and m != 10:
        return
    # ================== 2. 绘图配置 ==================
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})

    # 颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # ================== 3. 绘制折线图 ==================
    for idx, model in enumerate(models):
        means, stds = results[model]

        # 绘制带误差线的折线
        plt.errorbar(
            x=C_values,
            y=means,
            yerr=stds,
            color=colors[idx],
            marker=markers[idx],
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            elinewidth=2,
            label=model
        )

    # ================== 4. 图形美化 ==================
    #plt.xscale('log')  # C值通常用对数刻度
    plt.xticks(C_values, labels=[str(c) for c in C_values])
    plt.xlabel('Regularization Parameter C', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Classification Performance Comparison with Varying C', fontsize=16)
    plt.legend(loc='lower right', frameon=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.5, 1)  # 根据实际数据调整范围

    # ================== 5. 保存/显示图像 ==================
    plt.tight_layout()
    plt.savefig('accuracy_vs_C_rec_'+str(m)+'.png', dpi=300)
    plt.show()

def compute_A_T_AT(data,label,X_target_unlabeled,y_test,C,A_T_acc,name):
    print(f"\n===== Testing {name}_C={C} =====")
    try_K_li = linear_kernel(data)

    try_svm = SVC(kernel='precomputed', C=C)
    try_svm.fit(try_K_li, label)

    try_kernel_test = np.dot(X_target_unlabeled, data.T)
    try_result = try_svm.predict(try_kernel_test)
    best = accuracy_score(y_test,try_result)
    print(f"try:{accuracy_score(y_test, try_result)}")
    for de in [1.5, 2.0]:
        try_K_li = polynomial_kernel(data, degree=de)

        try_svm = SVC(kernel='precomputed', C=C)
        try_svm.fit(try_K_li, label)

        try_kernel_test = np.dot(X_target_unlabeled, data.T)
        try_result = try_svm.predict(try_kernel_test)
        if accuracy_score(y_test, try_result) > best:
            best = accuracy_score(y_test, try_result)
        print(f"try:{accuracy_score(y_test, try_result)}")
    A_T_acc[name].append(best)

# ============== 5. 完整实验流程 ==============
if __name__ == "__main__":
    np.random.seed(42)
    # 配置实验参数
    config = {
        "auxiliary": {
            "positive": ["comp.windows.x"],
            "negative": ["rec.sport.hockey"]
        },
        "target": {
            "positive": ["comp.sys.ibm.pc.hardware"],
            "negative": ["rec.motorcycles"]
        }
    }
    m_test = [1,3,5,7,10]  # 每个类别的标记样本数
    n_runs = 3  # 实验重复次数
    svd_dim = 500
    C_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    #C_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]  # 需要测试的C值
    results = {C: [] for C in C_values}
    models_list = ['DTMKL_f', 'FR', 'A-SVM', 'CD-SVM', 'KMM']
    multi_results = {C: {} for C in C_values}
    A_T_results = {C: {} for C in C_values}
    A_T_list = ['A','T','AT']
    compare = 1 #要不要对比试验
    SVM_A = 0 #要不要基础SVM对比
    for m in m_test:
        for C in C_values:
            print(f"\n===== Testing C={C} =====")
            # 运行实验
            accuracies = []
            multi_acc = {name: [] for name in models_list}
            A_T_acc = {name: [] for name in A_T_list}
            for seed in range(n_runs):
                print(f"\n===== Run {seed + 1}/{n_runs} =====")

                # 1. 加载原始文本数据
                X_aux_text, y_aux, X_target_text, y_target_labeled, X_test_text, y_test = \
                    load_data_ori(config, m=m, random_state=seed)

                # 2. 特征处理（确保一致性）
                processor = FeatureProcessor(svd_dim=svd_dim)
                all_text = X_aux_text + X_target_text + X_test_text
                processor.fit_transform(all_text)

                # 转换各部分数据
                X_aux_feat = processor.transform(X_aux_text)
                X_target_feat = processor.transform(X_target_text)
                X_test_feat = processor.transform(X_test_text)

                # model = SentenceTransformer('all-mpnet-base-v2')
                # X_aux_feat = model.encode(X_aux_text, show_progress_bar=True)
                # X_target_feat = model.encode(X_target_text, show_progress_bar=True)
                # X_test_feat = model.encode(X_test_text, show_progress_bar=True)
                # feat_extractor = BetterFeatureExtractor
                # if feat_extractor:
                #     X_aux_feat = feat_extractor.extract_feature_for_multiple_exs(X_aux_text)
                #     X_target_feat = feat_extractor.extract_feature_for_multiple_exs(X_target_text)
                #     X_test_feat = feat_extractor.extract_feature_for_multiple_exs(X_test_text)
                # else:
                #     train_feat = None

                # 3. 划分目标域标记/未标记数据
                n_labeled = 2 * m
                X_target_labeled = X_target_feat[:n_labeled]
                X_target_unlabeled = X_target_feat[n_labeled:]

                #try 模型对比

                if SVM_A == 1:
                    compute_A_T_AT(X_aux_feat,y_aux,X_target_unlabeled,y_test,C,A_T_acc,'A')
                    compute_A_T_AT(X_target_labeled, y_target_labeled, X_target_unlabeled, y_test, C,A_T_acc,'T')
                    compute_A_T_AT(np.concatenate([X_aux_feat, X_target_labeled], axis=0),np.concatenate([y_aux, y_target_labeled], axis=0),X_target_unlabeled, y_test,C,A_T_acc,'AT')
                if compare == 1:
                    # 训练和评估 FR
                    fr_model = FeatureReplicationSVM(C=C)
                    fr_model.fit(X_aux_feat, y_aux, X_target_labeled, y_target_labeled)
                    y_pred_fr = fr_model.predict(X_test_feat)
                    multi_acc['FR'].append(accuracy_score(y_test, y_pred_fr))
                    print("FR Accuracy:", accuracy_score(y_test, y_pred_fr))

                    # 训练和评估 A-SVM
                    a_svm = AdaptiveSVM(C=C)
                    a_svm.fit(X_aux_feat, y_aux, X_target_labeled, y_target_labeled)
                    y_pred_asvm = a_svm.predict(X_test_feat)
                    multi_acc['A-SVM'].append(accuracy_score(y_test, y_pred_asvm))
                    print("A-SVM Accuracy:", accuracy_score(y_test, y_pred_asvm))

                    # 训练和评估 CD-SVM
                    cd_svm = CDSVM(C=C, k=5)
                    cd_svm.fit(X_aux_feat, y_aux, X_target_labeled, y_target_labeled)
                    y_pred_cdsvm = cd_svm.predict(X_test_feat)
                    multi_acc['CD-SVM'].append(accuracy_score(y_test, y_pred_cdsvm))
                    print("CD-SVM Accuracy:", accuracy_score(y_test, y_pred_cdsvm))

                    # 训练和评估 KMM（需结合SVM）
                    kmm = KMM(gamma=0.1, B=0.9)
                    kmm.fit(X_aux_feat, X_target_labeled, X_target_unlabeled)
                    weights = kmm.get_weights()

                    # 使用KMM权重训练SVM
                    svm = SVC(C=C, kernel='linear')
                    X_train = np.vstack([X_aux_feat, X_target_labeled])
                    y_train = np.concatenate([y_aux, y_target_labeled])
                    sample_weight = np.concatenate([weights, np.ones(len(y_target_labeled))])
                    svm.fit(X_train, y_train, sample_weight=sample_weight)
                    y_pred_kmm = svm.predict(X_test_feat)
                    multi_acc['KMM'].append(accuracy_score(y_test, y_pred_kmm))
                    print("KMM Accuracy:", accuracy_score(y_test, y_pred_kmm))

                # 4. 训练模型
                model = DTMKL_f(C=C)
                model.fit(
                    X_aux_feat,
                    y_aux,
                    X_target_labeled,
                    y_target_labeled,
                    X_target_unlabeled
                )

                n_labeled = len(y_target_labeled)
                n_aux = len(y_aux)
                # 5. 预测评估
                y_pred_continuous = model.predict(n_labeled, n_aux)
                y_pred = np.where(y_pred_continuous >= 0, 1, -1)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)
                multi_acc['DTMKL_f'].append(acc)
                print(f"Run {seed + 1} Accuracy: {acc:.4f}")
            results[C] = accuracies
            if compare == 1:
                multi_results[C] = multi_acc
            if SVM_A == 1:
                A_T_results[C] = A_T_acc

        plot_result = {name: ([], []) for name in models_list}
        print("\n===== Final Results =====")
        for C in C_values:

            mean_acc = np.mean(results[C])
            std_acc = np.std(results[C])
            print(f"C={C}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
            if compare == 1:
                for r in models_list:
                    mean_acc_m = np.mean(multi_results[C][r])
                    std_acc_m = np.std(multi_results[C][r])

                    plot_result[r][0].append(mean_acc_m)
                    plot_result[r][1].append(std_acc_m)
                    print(f"C={C}: model-{r}Accuracy = {mean_acc_m:.4f} ± {std_acc_m:.4f}")
            if SVM_A == 1:
                for r in A_T_list:
                    mean_acc_at = np.mean(A_T_results[C][r])
                    std_acc_at = np.std(A_T_results[C][r])
                    print(f"C={C}: model-{r}Accuracy = {mean_acc_at:.4f} ± {std_acc_at:.4f}")
        if compare == 1:
            draw_pic(models_list, plot_result, C_values,m)
