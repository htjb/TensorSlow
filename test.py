from tensorslow.tensor import Tensor

a = Tensor(2.0)
b = Tensor(3.0)

c = a * b + a
c.backward()

print(c.data, type(c), c.grad)
print(a.grad, b.grad)