import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def one_population_mean_t_square_test(x, expect_mean):
    """
    进行单总体均值向量检验（未知协方差）
    :param x: 数据
    :param expect_mean: 期望均值
    :return: F统计量，显著性概率p
    """
    x = np.matrix(x)
    n, dim = x.shape
    x_mean = np.matrix(np.mean(x, axis=0)).T  # 均值
    A_X = (x-x_mean.T).T.dot(x-x_mean.T)  # 样本离差阵
    T_square = n * (n - 1) * np.matmul(np.matmul((x_mean - expect_mean).T, np.linalg.inv(A_X)), (x_mean - expect_mean))
    F = (n - dim) / ((n - 1) * dim) * T_square
    P = 1 - stats.f.cdf(F, dim, n - dim)
    return F, P


def two_population_mean_t_square_test(X11, X12):
    """
    进行两总体均值向量检验（未知协方差）
    :param X11: 第一个总体的数据
    :param X12: 第二个总体的数据
    :return: F统计量，显著性概率p
    """
    n = X11.shape[0]
    m = X12.shape[0]
    p = X11.shape[1]

    X11_mean = np.mean(X11, axis=0)
    X12_mean = np.mean(X12, axis=0)
    A_X11 = np.matrix(np.cov(X11.T))
    A_X12 = np.matrix(np.cov(X12.T))
    T_square = (n + m - 2) * n * m / (n + m) * np.matmul(np.matmul((X11_mean - X12_mean).T,
                                                                   np.linalg.inv(A_X11 + A_X12)), (X11_mean - X12_mean))
    F = ((n + m - 2) - p + 1) / ((n + m - 2) * p) * T_square
    P = 1 - stats.f.cdf(F, p, ((n + m - 2) - p + 1))
    return F, P


def three_population_mean_wilks_test(X1, X2, X3, alpha=0.01):
    X = np.concatenate((X1, X2, X3), axis=0)
    A = (X1.shape[0]-1)*np.cov(X1, rowvar=False) + (X2.shape[0]-1)*np.cov(X2, rowvar=False) + (X3.shape[0]-1)*np.cov(X3, rowvar=False)
    T = (X.shape[0]-1)*np.cov(X, rowvar=False)
    wilks_value = np.linalg.det(A) / np.linalg.det(T)
    F_value = (X.shape[0]-3-X.shape[1]+1) / X.shape[1] * (1-np.sqrt(wilks_value)) / np.sqrt(wilks_value)

    F_critical = stats.f.ppf(1-alpha, 2*X.shape[1], 2*(X.shape[0]-3-X.shape[1]))
    p_value = 1 - stats.f.cdf(F_value, 2*X.shape[1], 2*(X.shape[0]-3-X.shape[1]))

    return F_critical, p_value


def LDA_reduce_dim(x, y, reduced_dim):
    lda = LinearDiscriminantAnalysis(n_components=reduced_dim)
    x_lda = lda.fit_transform(x, y)

    return x_lda


def PCA_reduce_dim(x, reduced_dim):
    pca = PCA(n_components=reduced_dim)
    x_pca = pca.fit_transform(x)

    return x_pca


def FA_reduce_dim(x, reduced_dim):
    fa = FactorAnalyzer(n_factors=reduced_dim, rotation='varimax')
    fa.fit(x)
    x_fa = fa.transform(x)

    return x_fa


def distance_classifier(x_train, y_train, x_test):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)

    # 先验概率、方差和类别中心
    # prior_probs = qda.priors_
    # class_means = qda.means_
    # class_covariances = qda.covariance_

    y_pred = qda.predict(x_test)

    return y_pred


def kmeans_classifier(x_train, y_train, x_test):
    class_0_center = np.mean(x_train[y_train == 0], axis=0)
    class_1_center = np.mean(x_train[y_train == 1], axis=0)

    # 将中心作为初始聚类中心
    initial_centers = np.vstack([class_0_center, class_1_center])
    kmeans = KMeans(n_clusters=2, random_state=6, init=initial_centers)
    kmeans.fit(x_test)
    y_pred = kmeans.labels_

    return y_pred


def svm_classfier(x_train, y_train, x_test, kernel='linear'):
    svm_cf = SVC(kernel=kernel, C=1.0)  # kernel: 'linear', 'poly', 'rbf', 'sigmoid'
    svm_cf.fit(x_train, y_train)

    y_pred = svm_cf.predict(x_test)

    return y_pred


def rf_classifier(x_train, y_train, x_test):
    rf_cf = RandomForestClassifier(n_estimators=100, random_state=6)
    rf_cf.fit(x_train, y_train)

    y_pred = rf_cf.predict(x_test)

    return y_pred
