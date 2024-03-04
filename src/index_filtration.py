from harmonic.harmonic import IndexFiltration, Simplex, FilteredComplex, VietorisRipsFiltration

v1 = Simplex((1,), index=0)
v2 = Simplex((2,), index=1)
v3 = Simplex((3,), index=3)
simplex1 = Simplex((1,2), index=2)
simplex2 = Simplex((1,3), index=4)

cmplx = [v1, v2, v3, simplex1, simplex2]
index_filtration = IndexFiltration(cmplx)

filtered_complex = index_filtration()

for simplex in filtered_complex.filtration:
    print(simplex.vertices, simplex.index, simplex.time)

filtered_complex = index_filtration(identity=True)

for simplex in filtered_complex.filtration:
    print(simplex.vertices, simplex.index, simplex.time)

persistence_diagram = filtered_complex.persistence_diagram

print(persistence_diagram.as_numpy())