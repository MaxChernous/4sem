import gmsh
import sys

import gmsh
import sys

gmsh.initialize()
gmsh.model.add("torus_mesh")

# torus: (x, y, z, R, r)
R = 0.5
r = 0.2

torus_tag = gmsh.model.occ.addTorus(0, 0, 0, R, r)
gmsh.model.occ.synchronize()

# Настройка плотности сетки
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)

gmsh.model.mesh.generate(3)
gmsh.write("tor.msh")

# Запуск визуализации
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

