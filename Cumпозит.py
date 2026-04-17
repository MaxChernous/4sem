from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import fem, mesh, la
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTXWriter

# --- 1. ГЕОМЕТРИЯ (в метрах) ---
# 10x10 мм, толщина 1.2 мм
L, W = 10.0e-3, 10.0e-3
th_Mo, th_Cer = 0.2e-3, 0.3e-3
total_thickness = 3*th_Mo + 2*th_Cer

# Создаем сетку (40x40x20 ячеек, чтобы хорошо разрешить 5 слоев)
msh = mesh.create_box(MPI.COMM_WORLD, 
                      [np.array([0, 0, 0]), np.array([L, W, total_thickness])],
                      (40, 40, 20), mesh.CellType.tetrahedron)

# Пространство для перемещений (Lagrange P1)
V = fem.functionspace(msh, ("Lagrange", 1, (3,)))

# --- 2. СВОЙСТВА МАТЕРИАЛОВ (ПО СЛОЯМ) ---
# Создаем пространство DG0 (одно значение на каждую ячейку сетки)
V_prop = fem.functionspace(msh, ("DG", 0))

E_field = fem.Function(V_prop)
nu_field = fem.Function(V_prop)
alpha_field = fem.Function(V_prop)

# Границы слоев по Z
z_bounds = [
    0.0, 
    th_Mo, 
    th_Mo + th_Cer, 
    2*th_Mo + th_Cer, 
    2*th_Mo + 2*th_Cer, 
    total_thickness
]

def map_properties(x):
    # По умолчанию - Молибден (индексы слоев 0, 2, 4)
    # E=320 ГПа, nu=0.31, alpha=5.2e-6
    e_vals = np.full(x.shape[1], 320.0e9)
    n_vals = np.full(x.shape[1], 0.31)
    a_vals = np.full(x.shape[1], 5.2e-6)
    
    # Керамика (индексы слоев 1, 3)
    # E=297 ГПа, nu=0.25, alpha=6.0e-6
    is_cer1 = (x[2] >= z_bounds[1]) & (x[2] <= z_bounds[2])
    is_cer2 = (x[2] >= z_bounds[3]) & (x[2] <= z_bounds[4])
    mask_cer = is_cer1 | is_cer2
    
    e_vals[mask_cer] = 297.0e9
    n_vals[mask_cer] = 0.25
    a_vals[mask_cer] = 6.0e-6
    return e_vals, n_vals, a_vals

# Интерполируем свойства в ячейки
e_data, n_data, a_data = map_properties(msh.geometry.x.T)
# Для DG0 нужно заполнить массив значений напрямую через ячейки или через интерполяцию
# Самый надежный способ для слоев - функция с проверкой координат
def e_expr(x): return map_properties(x)[0]
def n_expr(x): return map_properties(x)[1]
def a_expr(x): return map_properties(x)[2]

E_field.interpolate(e_expr)
nu_field.interpolate(n_expr)
alpha_field.interpolate(a_expr)

# --- 3. ФИЗИКА И УРАВНЕНИЯ ---
# Параметры Ламэ теперь тоже поля
mu = E_field / (2 * (1 + nu_field))
lmbda = E_field * nu_field / ((1 + nu_field) * (1 - 2 * nu_field))

# Тензор КТР (теперь изотропный, но разный в слоях)
alpha_tensor = alpha_field * ufl.Identity(3)

# Нелинейная кинематика (Green-Lagrange)
I = ufl.Identity(3)
def E_GL(u):
    F = I + ufl.grad(u)
    return 0.5 * (F.T * F - I)

u = fem.Function(V)
v = ufl.TestFunction(V)
dT_eff = fem.Constant(msh, PETSc.ScalarType(0.0))

# Энергия с учетом теплового расширения
def psi(u):
    E_strain = E_GL(u) - alpha_tensor * dT_eff
    return mu * ufl.inner(E_strain, E_strain) + 0.5 * lmbda * (ufl.tr(E_strain))**2

dx = ufl.dx
Pi = psi(u) * dx
F_res = ufl.derivative(Pi, u, v)
J = ufl.derivative(F_res, u, ufl.TrialFunction(V))

# --- 4. ГРАНИЧНЫЕ УСЛОВИЯ (Жесткая заделка левого торца) ---
def left_face(x):
    return np.isclose(x[0], 0.0)

f_entities = mesh.locate_entities_boundary(msh, 2, left_face)
dofs = fem.locate_dofs_topological(V, 2, f_entities)
bc = fem.dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), dofs, V)

# --- 5. РЕШАТЕЛЬ ---
problem = NonlinearProblem(F_res, u, bcs=[bc], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.rtol = 1e-6
solver.max_it = 30

# Прямой метод (MUMPS) для стабильности
ksp = solver.krylov_solver
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# --- 6. НАПРЯЖЕНИЯ И ЗАПИСЬ ---
W_stress = fem.functionspace(msh, ("Lagrange", 1))
stress_field = fem.Function(W_stress)
stress_field.name = "von_Mises_Stress"
u.name = "Deformation"

# Формула Мизеса через поля mu и lambda
E_val = E_GL(u) - alpha_tensor * dT_eff
S = lmbda * ufl.tr(E_val) * I + 2.0 * mu * E_val 
dev_S = S - (1./3) * ufl.tr(S) * I
von_Mises_ufl = ufl.sqrt(1.5 * ufl.inner(dev_S, dev_S))
stress_expr = fem.Expression(von_Mises_ufl, W_stress.element.interpolation_points())

vtx = VTXWriter(msh.comm, "MoCeramic_Cooling.bp", [u, stress_field], engine="BP4")

# --- 7. ЦИКЛ ОХЛАЖДЕНИЯ (от 1400 до 20 C) ---
dT_final = -1380.0
nsteps = 50

print(f"Старт охлаждения: {nsteps} шагов...")
for i in range(1, nsteps + 1):
    t = i / nsteps
    dT_eff.value = dT_final * t
    
    try:
        solver.solve(u)
        u.x.scatter_forward()
        stress_field.interpolate(stress_expr)
        vtx.write(t)
        print(f"Шаг {i}/{nsteps}: dT = {dT_eff.value:.1f}")
    except RuntimeError:
        print("Ошибка сходимости! Попробуй увеличить nsteps.")
        break

vtx.close()
print("Готово! Результат в MoCeramic_Cooling.bp")
