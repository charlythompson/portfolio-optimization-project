# Recursive function to generate weights for the portfolios
def generate_weights(step=0.01, current=0.0, weights=None):
    if weights is None:
        weights = []
    if current > 1:
        return weights
    weight1 = round(current, 4)
    weight2 = round(1 - current, 4)
    weights.append([weight1, weight2])
    return generate_weights(step, current + step, weights)
