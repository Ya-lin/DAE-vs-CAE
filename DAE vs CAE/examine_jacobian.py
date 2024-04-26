# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:44:22 2024

@author: yalin
"""

import torch
from torch.autograd.functional import jacobian


#%% example one

x = torch.rand(10, 2)
def test1(x):
    return x.exp().sum(dim=1)

y = test1(x)
# y.shape=10 and x.shape=(10,2)==>dy_dx.shape=(10, (10,2))
dy_dx = jacobian(test1, x)
print("\njacobian shape", dy_dx.shape)

c = 0
for j in range(10):
    for i in range(10):
        if i!=j:
            c += 1
            print("\n", dy_dx[i,j])
            # All must be zero as jth output does not rely on ith input.
print("\nnumber of availabel elements in jacobian:", 100-c)


#%% example two
x = torch.rand(10, 2, 2)
def test2(x):
    y1 = x.exp()
    y2 = x**3
    y3 = x**2
    y = y1+y2-y3/5
    return y.sum(dim=(1,2))

y = test2(x)
dy_dx = jacobian(test2, x)
print("\njacobian shape", dy_dx.shape)

c = 0
for j in range(10):
    for i in range(10):
        if i!=j:
            c += 1
            print("\n", dy_dx[i,j])
print("\nnumber of availabel elements in jacobian:", 100-c)


#%% example three
x = torch.rand(10, 2, 2)
def test3(x):
    y1 = x.exp()
    y2 = x**3
    y3 = x**2
    y = y1+y2-y3/5
    return y.sum(dim=1)

y = test3(x)
dy_dx = jacobian(test3, x)
print("\njacobian shape", dy_dx.shape)

c = 0
for j in range(10):
    for i in range(10):
        if i!=j:
            c += 1
            print("\n", dy_dx[i,:,j].shape)
            print(dy_dx[i,:,j])
print("\nnumber of availabel elements in jacobian:", 100-c)



