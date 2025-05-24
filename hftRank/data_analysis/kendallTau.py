from scipy.stats import kendalltau


def kendall_tau(ranking_ids_1, ranking_ids_2):
    """
    Calculates the Kendall's Tau distance between two rankings
    :param ranking_ids_1: list of positive integers → ranking of items represented by corresponding ID numbers
    :param ranking_ids_2: list of positive integers → re-ranking of ranking_ids_1
    :return: float value → Kendall's Tau distance
    """

    # check if the rankings are of the same length
    if len(ranking_ids_1) != len(ranking_ids_2):
        return None, "X and Y are not the same length"
    # number of concordant pairs
    c = 0

    n = len(ranking_ids_1)

    for i in range(n - 1):
        # Check if the i-th element of ranking_ids_2 is in ranking_ids_1
        if ranking_ids_2[i] in ranking_ids_1:
            # Calculate position in ranking_ids_1 of the i-th element of ranking_ids_2
            index1 = ranking_ids_1.index(ranking_ids_2[i])
            for j in range(i + 1, n):
                # Check if the j-th element of ranking_ids_2 is in ranking_ids_1
                if ranking_ids_2[j] in ranking_ids_1:
                    # Compare positions in ranking_ids_1 of the i-th and j-th elements of ranking_ids_2
                    if ranking_ids_1.index(ranking_ids_2[j]) > index1:
                        c += 1  # ranking_ids_2[i] and ranking_ids_2[j] are a concordance

    # total pairs of elements in one ranking is n choose 2
    total_pairs = n * (n - 1) / 2

    # calculate number of discordant pairs, i.e. non-concordant pairs
    d = total_pairs - c

    if total_pairs == 0:
        return 0.0, None  # or another fallback value
    return (c - d) / total_pairs, None


def kT(X, Y):
    """
    calculate kendall tau correlation coefficient between two rankings
    :param X: rank 1. use either the unique_ids or the index of the ranking
    :param Y: rank 2. use either the unique_ids or the index of the ranking
    :return: kendall tau correlation coefficient or a message if X and Y are not the same length
    """
    if len(X) != len(Y):
        return None, "X and Y are not the same length"

    corr, p_value = kendalltau(X, Y, variant='c')
    return corr, None
