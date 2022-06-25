function g_prime = sigmoid_prime(z)
g_prime = sigmoid(z) .* (1 - sigmoid(z));