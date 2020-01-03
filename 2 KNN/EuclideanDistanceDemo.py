import math


def ComputeEuclideanDistance(x1, y1, x2, y2):
    # y² = x1² + x2²
    distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))  # 指数表达式
    return str(distance)

res = ComputeEuclideanDistance(1, 1, 4, 5)
print(res)

