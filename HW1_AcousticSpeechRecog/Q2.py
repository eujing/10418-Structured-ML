import numpy as np

def relu(v):
    return np.maximum(np.zeros_like(v), v)

def softmax(v):
    tmp = np.exp(v)
    return tmp / np.sum(tmp)

A = np.array([
    [1, 1],
    [1, 1]])

B = np.array([
    [0.5, 0.5],
    [0.5, 0.5]])

C = np.array([
    [1, 1],
    [2, 1]])

d = np.array([0, 0]).reshape((2, 1))

e = np.array([0, 0]).reshape((2, 1))

b_0 = np.array([0, 0]).reshape((2, 1))

bs = []
cs = []

# zs = np.array([
#     [1, 0, 1],
#     [0, 1, 0]])

zs = np.array([0, 1]).reshape((2, 1))

b = b_0
for z in zs.T:
    b = relu(B.T @ b + A.T @ z.reshape(2, 1) + d)
    c = softmax(C.T @ b + e)

    bs.append(b.reshape(2))
    cs.append(c.reshape(2))

print(zs)
print(np.array(bs).T)
print(np.array(cs).T)

p = 1
for i in range(zs.shape[1]):
    pi = zs[:, i].T @ cs[i]
    print(f"p_i = {pi}")
    p = p * pi

print(p)



