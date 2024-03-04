v1 = Simplex((1,), index=0)
v2 = Simplex((2,), index=1)
v3 = Simplex((3,), index=3)
simplex1 = Simplex((1,2), index=2)
simplex2 = Simplex((2,3), index=4)

chain = Chain([simplex1, simplex2], [1, 1])

cmplx = FilteredComplex([v1, v2, simplex1, v3, simplex2])

print(simplex1, simplex1.dim, asdict(simplex1))
print(chain)
print(cmplx)

v1 = Simplex((1,), index=0)
v2 = Simplex((2,), index=1)
simplex1 = Simplex((1,2), index=2)

cmplx = [v1, v2, simplex1]

print(cmplx)