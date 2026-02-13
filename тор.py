import gmsh
import sys

import gmsh
import sys

gmsh.initialize()
gmsh.model.add("torus_mesh")

# torus: (x, y, z, R, r)
# R - радиус от центра тора до центра сечения (кольца)
# r - радиус самого сечения (толщина "бублика")
R = 0.5
r = 0.2

# Добавляем тор
torus_tag = gmsh.model.occ.addTorus(0, 0, 0, R, r)

# Синхронизируем геометрию с моделью
gmsh.model.occ.synchronize()

# Настройка плотности сетки
# Задаем размер ячейки через "Field" (поле), чтобы сетка была равномерной
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)

# Генерируем 3D сетку
gmsh.model.mesh.generate(3)

# Сохраняем результат
gmsh.write("tor.msh")

# Запуск визуализации, если не указан флаг -nopopup
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

