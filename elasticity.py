from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import (Expression, Function, FunctionSpace, dirichletbc,
                         form, functionspace, locate_dofs_topological)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary

dtype = PETSc.ScalarType

def build_nullspace(V: FunctionSpace):
    """
    Создаем 'околонулевое пространство' (near-nullspace).
    Для задач упругости это критически важно: это векторы перемещений тела как целого
    (3 переноса и 3 вращения), которые не вызывают внутренних напряжений.
    Многосеточный решатель (AMG) использует их, чтобы быстрее сходиться.
    """
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Задаем три моды поступательного движения (вдоль осей X, Y, Z)
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Задаем три моды вращения вокруг осей
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis) # Ортонормируем базис для стабильности расчетов

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm) for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)

# --- Определение задачи ---

# Создаем сетку-параллелепипед 2x1x1, разбитую на тетраэдры
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
    (16, 16, 16),
    CellType.tetrahedron,
    ghost_mode=GhostMode.shared_facet,
)

x = ufl.SpatialCoordinate(msh)
# Создаем крутящий момент вокруг оси X
f = ufl.as_vector((0.0, -x[2], x[1])) * 5.0e7

# Параметры материала (сталь или что-то очень жесткое)
E = 1.0e9 # Модуль Юнга
ν = 0.3   # Коэффициент Пуассона
μ = E / (2.0 * (1.0 + ν)) # Параметр Ламе (сдвиг)
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν)) # Параметр Ламе (объемное сжатие)

def σ(v):
    """Закон Гука: вычисляем тензор напряжений через деформации"""
    return 2.0 * μ * ufl.sym(ufl.grad(v)) + λ * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v))

# Пространство функций Лагранжа (степень 1 — линейные элементы)
V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Вариационная форма: внутренняя энергия (напряжения) = внешняя работа (силы f)
a = form(ufl.inner(σ(u), ufl.grad(v)) * ufl.dx)
L = form(ufl.inner(f, v) * ufl.dx)

# Граничные условия: жестко фиксируем торцы балки при x=0 и x=1
facets = locate_entities_boundary(
    msh, dim=2, marker=lambda x: np.isclose(x[0], 0.0)
)
bc = dirichletbc(
    np.zeros(3, dtype=dtype), locate_dofs_topological(V, entity_dim=2, entities=facets), V=V
)

# --- Сборка и решение ---

A = assemble_matrix(a, bcs=[bc])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]]) # Корректируем правую часть с учетом фиксации границ
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bc.set(b.array_w)

# Подключаем наше 'нулевое пространство' к матрице, чтобы помочь AMG-решателю
ns = build_nullspace(V)
A.setNearNullSpace(ns)
A.setOption(PETSc.Mat.Option.SPD, True) # Матрица симметрична и положительно определена

# Настройка решателя: метод сопряженных градиентов (cg) + алгебраический многосеточный метод (gamg)
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["pc_type"] = "gamg"
solver = PETSc.KSP().create(msh.comm)
solver.setFromOptions()
solver.setOperators(A)

uh = Function(V)
solver.solve(b, uh.x.petsc_vec) # Главный момент: находим перемещения uh
uh.x.scatter_forward()

# --- Пост-процессинг ---

# Считаем напряжение по Мизесу (показывает, где материал скорее всего "порвется")
sigma_dev = σ(uh) - (1 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))

# Переносим результаты в пространство кусочно-постоянных функций (для каждой ячейки)
W = functionspace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points)
sigma_vm_h = Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)

# Сохраняем всё для ParaView
with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)