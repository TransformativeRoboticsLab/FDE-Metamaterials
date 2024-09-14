import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.pt = np.array([x, y])
        
    @property
    def x(self):
        return self.pt[0]
    
    @property
    def y(self):
        return self.pt[1]

p = Point(0.3, 0.1)
p1 = Point(p.y, p.x)
p2 = Point(-p.y, p.x)
p3 = Point(-p.x, p.y)
p4 = Point(-p.x, -p.y)
p5 = Point(-p.y, -p.x)
p6 = Point(p.y, -p.x)
p7 = Point(p.x, -p.y)

points = [p.pt, p1.pt, p2.pt, p3.pt, p4.pt, p5.pt, p6.pt, p7.pt, p.pt]
points = np.array(points)
plt.scatter(points[:,0], points[:,1])
# plt.plot(points[:,0], points[:,1])
plt.show()
