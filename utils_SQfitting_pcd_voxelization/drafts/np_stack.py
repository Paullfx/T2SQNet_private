import numpy as np

x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
print(x.shape)

a = x[1:2:1]
print("a:",a)




# a = np.array([1,2,3])
# b = np.array([4,5,6])

# c = np.stack([a,b])

# d = np.stack([a,b],axis = -1)
 
# e = np.stack([a,b],axis = 0)

# f = np.stack([a,b],axis = 1)

# print("c: ",c)
# print(c.shape)
# print("d: ",d)
# print(d.shape)
# print("e: ",e)
# print(e.shape)
# print("f: ",f)
# print(f.shape)

