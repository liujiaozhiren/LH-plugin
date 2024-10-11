import numpy as np
def sorted_indices(matrix):
    n = matrix.shape[0]
    result = []

    for i in range(n):
        # 获取当前行
        row = matrix[i]
        # 获取升序排列后的索引
        sorted_indices = np.argsort(row)
        # 移除对角元素的索引
        adjusted_indices = [index for index in sorted_indices if index != i]
        result.append(adjusted_indices)
    return np.array(result)

def calculate_ndcg(ground_truth, search_results, k=5):
    # 只考虑前 k 个搜索结果
    search_results = search_results[:k]
    k_ground = ground_truth[:k]
    # search_rel = [1 if item in k_ground else 0 for item in search_results]
    # 计算 DCG
    dcg = 0.0
    for i, result in enumerate(search_results):
        rel = 1 if result in k_ground else 0  # 相关性评分
        dcg += rel / np.log2(i + 2)  # +2 是因为索引从0开始

    # 计算 IDCG (理想 DCG)
    idcg = sum(1 / np.log2(i + 2) for i, rel in enumerate(range(k)))

    # 计算 NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def ndcg_(ground_truth_matrix, embedding_dist_matrix):
    gtm = sorted_indices(ground_truth_matrix)
    edm = sorted_indices(embedding_dist_matrix)
    # ndcg = np.mean([calculate_ndcg(gtm[i], edm[i], k) for i in range(len(gtm))])
    mrr = calculate_mrr(gtm, edm)
    #err = calculate_err(gtm, edm, k)

    ndcg10 = np.mean([calculate_ndcg(gtm[i], edm[i], 10) for i in range(len(gtm))])
    ndcg100 = np.mean([calculate_ndcg(gtm[i], edm[i], 100) for i in range(len(gtm))])
    ndcg50= np.mean([calculate_ndcg(gtm[i], edm[i], 50) for i in range(len(gtm))])
    err10 = calculate_err(gtm, edm, 10)
    err100 = calculate_err(gtm, edm, 100)
    err50 = calculate_err(gtm, edm, 50)
    return f"ndcg10:{ndcg10:.4f},ndcg50:{ndcg50:.4f},ndcg100:{ndcg100:.4f}, err10:{err10:.4f},err50:{err50:.4f},err100:{err100:.4f},mrr:{mrr:.4f}"

def calculate_mrr(ground_truth, search_results):
    reciprocal_ranks = []
    for i in range(len(search_results)):
        search_line = search_results[i]
        ground_line = ground_truth[i]
        for rank, result in enumerate(search_line):
            if result == ground_line[0]:
                reciprocal_ranks.append(1 / (rank+1))
                break
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def calculate_err(ground_truth, search_results, k):
    reciprocal_ranks = []
    for i in range(len(search_results)):
        search_line = search_results[i]
        ground_line = ground_truth[i]
        ground_top_k = ground_line[:k]

        relevance_scores = [1 if result in ground_top_k else 0 for result in search_line]
        err = 0.0
        total_relevance = sum(relevance_scores)

        for i, rel in enumerate(relevance_scores[:k]):
            if rel > 0:
                click_prob = (1 / (i + 1)) * (rel / total_relevance)
                not_click_prob = 1.0

                for j in range(i):
                    if relevance_scores[j] > 0:
                        not_click_prob *= (1 - (1 / (j + 1)) * (relevance_scores[j] / total_relevance))

                err += click_prob * not_click_prob

        return err



if __name__ == '__main__':
    ground_truth_matrix = np.array([[0, 0.01, 2, 5, 3],
                                    [0.2, 0, 0.3, 0.4, 0.6],
                                    [1, 0, 0, 3, 4],
                                    [3, 1, 4, 0, 2],
                                    [0.5, 0.2, 0.3, 0.1, 0]])
    embedding_dist_matrix = np.array([[0, 0.01, 2, 5, 3],
                                    [0.2, 0, 0.3, 0.4, 0.6],
                                    [1, 0, 0, 3, 4],
                                    [3, 1, 4, 0, 2],
                                    [0.5, 0.2, 0.3, 0.1, 0]])

    embedding_dist_matrix[3], embedding_dist_matrix[0] = embedding_dist_matrix[0], embedding_dist_matrix[3]

    k = 3
    print(ndcg_k(ground_truth_matrix, embedding_dist_matrix, k))
