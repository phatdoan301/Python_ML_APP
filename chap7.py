class Cong:
  def __init__(self):
    self.a = None
    self.b = None
  def thuan(self, a, b):
    self.a = a
    self.b = b
    return a + b
  def nguoc(self, out):
    return out, out
class Nhan:
  def __init__(self):
    self.a = None
    self.b = None
  def thuan(self, a, b):
    self.a = a
    self.b = b
    return a * b
  def nguoc(self, out):
    da = out * self.b
    db = out * self.a
    return da, db
class Luythua:
  def __init__(self, mu):
    self.a = None
    self.mu = mu
  def thuan(self, a):
    self.a = a
    return a ** self.mu
  def nguoc(self, out):
    return out * self.mu * (self.a ** (self.mu - 1))

node_cong1 = Cong()
node_cong2 = Cong()
node_nhan1 = Nhan()
node_nhan2 = Nhan()
node_luythua = Luythua(2)

w = 2
b = 8
x = -2
y = 2

# Lan truyền thuận
c = node_nhan1.thuan(w, x)
a = node_cong1.thuan(c, b)
d = node_cong2.thuan(a, -y)
e = node_luythua.thuan(d)
loss = node_nhan2.thuan(0.5, e)
print('Loss:', loss)

# Lan truyền ngược
_, A = node_nhan2.nguoc(1)
print('A:', A)
B = node_luythua.nguoc(A)
print('B:', B)
C, _ = node_cong2.nguoc(B)
print('C:', C)
D, E = node_cong1.nguoc(B)
print('D:', D)
print('E:', E)
F, _ = node_nhan1.nguoc(D)
print('F:', F)