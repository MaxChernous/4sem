import gmsh
import sys

import gmsh
import sys

gmsh.initialize()
gmsh.model.add("torus_mesh")

# Внешний и внутренний торы
torus_1 = gmsh.model.occ.addTorus(0, 0, 0, 1, 0.2)
torus_2 = gmsh.model.occ.addTorus(0, 0, 0, 1, 0.3)
gmsh.model.occ.cut([(3, torus_2)], [(3, torus_1)])

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

