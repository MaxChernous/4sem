from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import (Expression, Function, dirichletbc,
                         form, functionspace, locate_dofs_topological)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from dolfinx.fem import Constant

from dolfinx import fem
from dolfinx.io import VTXWriter

dtype = PETSc.ScalarType

# --- 1. Геометрия берестинки ---
length, width, thickness = 2.0, 1.0, 0.025
msh = create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([length, width, thickness])],
                 (60, 20, 5), CellType.tetrahedron)

msh.topology.create_entities(1)
msh.topology.create_connectivity(1, msh.topology.dim)

V = functionspace(msh, ("Lagrange", 1, (3,)))

# --- 2. Физическа берестинки ---
E, ν = 0.5e9, 0.3
μ, λ = E / (2*(1+ν)), E*ν / ((1+ν)*(1-2*ν))
dT = 300.0

x = ufl.SpatialCoordinate(msh)
z_norm = x[2] - thickness / 2.0
x_norm = x[0] - length / 2.0

# Ориентация волокон (чтобы спиралька)
theta = 0.65 * x[1]

R = ufl.as_tensor([
    [ufl.cos(theta), -ufl.sin(theta), 0],
    [ufl.sin(theta),  ufl.cos(theta), 0],
    [0, 0, 1]
])

alpha_max = 5e-4

alpha_local = ufl.as_tensor([
    [alpha_max * (z_norm / (thickness / 2.0)) * (1 + 0.1 * x_norm / (length / 2.0)), 0, 0],
    [0, 1e-6, 0],
    [0, 0, 1e-6]
])

# Поворот α вместе с системой координат
alpha_tensor = R * alpha_local * R.T

# ассиметрия по ширине
alpha_tensor = alpha_tensor * (1 + 0.2 * x[1] / width)

# --- Нелинейная кинематика ---
I = ufl.Identity(3)

def F_def(u):
    return I + ufl.grad(u)

def E_GL(u):
    F = F_def(u)
    return 0.5 * (F.T * F - I)

# --- 3. Граничные условия ---
msh.topology.create_entities(2)
msh.topology.create_connectivity(2, msh.topology.dim)

def left_face(x):
    return np.isclose(x[0], 0.0)

face_entities = locate_entities_boundary(msh, dim=2, marker=left_face)
dofs = locate_dofs_topological(V, entity_dim=2, entities=face_entities)
bc = dirichletbc(np.zeros(3, dtype=dtype), dofs, V)

# --- 4. Нелинейная задача ---
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
dx = ufl.dx(domain=msh)

u = Function(V)
v = ufl.TestFunction(V)

dT_eff = Constant(msh, PETSc.ScalarType(0.0))

# Энергия деформации
def psi_step(u):
    E = E_GL(u) - alpha_tensor * dT_eff
    return μ * ufl.inner(E, E) + 0.5 * λ * (ufl.tr(E))**2

Pi = psi_step(u) * dx
F_res = ufl.derivative(Pi, u, v)
J = ufl.derivative(F_res, u, ufl.TrialFunction(V))

problem = NonlinearProblem(
    F_res, u,
    bcs=[bc],
    J=J
)

# --- 5. Параметры солвера ---
solver = NewtonSolver(msh.comm, problem)
solver.max_it = 50
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
ksp.setType("preonly")
ksp.getPC().setType("lu")
opts = PETSc.Options()
opts["pc_factor_mat_solver_type"] = "mumps" 
ksp.setFromOptions()

solver.relaxation_parameter = 1.0
nsteps = 200

# Проверка связи (нулевой шаг)
dT_eff.value = 0.0
it, converged = solver.solve(u)
if converged:
    print(f"Нулевой шаг сошелся за {it} итераций")
else:
    print("Капец, dT=0 не сходится!")

# --- 7. Расчет напряжений и сохранение ---
W_stress = fem.functionspace(msh, ("Lagrange", 1))
stress_field = fem.Function(W_stress)
stress_field.name = "von_Mises_Stress"
u.name = "Deformation"

# Формула для Мизеса
E_val = E_GL(u) - alpha_tensor * dT_eff
S = λ * ufl.tr(E_val) * ufl.Identity(3) + 2.0 * μ * E_val 
s = S - (1./3) * ufl.tr(S) * ufl.Identity(3)
von_Mises_expr_ufl = ufl.sqrt(1.5 * ufl.inner(s, s))

# Создаем выражение для интерполяции один раз
stress_expr = fem.Expression(von_Mises_expr_ufl, W_stress.element.interpolation_points())

# Инициализируем VTXWriter (имя файла теперь будет папкой .bp)
vtx = VTXWriter(msh.comm, "birch_animation.bp", [u, stress_field], engine="BP4")

print("--- Основной цикл ---")
for i in range(1, nsteps + 1):
    t = i / nsteps  # Условное время для ParaView
    dT_eff.value = dT * t
    
    try:
        n_it, converged = solver.solve(u)
        u.x.scatter_forward()
        
        # Обновляем карту напряжений для текущего шага
        stress_field.interpolate(stress_expr)
        
        # Записываем в файл VTX
        vtx.write(t)
        print(f"Шаг {i}/{nsteps} записан")
        
    except RuntimeError:
        print(f"Ошибка на {i}-ом шаге!")
        break
vtx.close()

print("gg wp!")
