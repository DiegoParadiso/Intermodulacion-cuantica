"""
Simular cómo tres modos vibracionales (f₁, f₂, f₃) interactúan no linealmente, 
generando nuevos modos (f₄, f₅, ...) mediante un circuito cuántico de 3qubytes.
"""

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Cargar API Key desde archivo .env
load_dotenv()
API_KEY = os.getenv("IBM_QUANTUM_API_KEY")

if not API_KEY:
    raise ValueError("No se encontró la API Key en el archivo .env")

# Conectar con IBM Quantum
service = QiskitRuntimeService(channel="ibm_quantum", token=API_KEY)

# Crear circuito cuántico base
n = 3
qc = QuantumCircuit(n)

# Registro clásico
cr = ClassicalRegister(n, name='cr')
qc.add_register(cr)

# Modos vibracionales base
qc.h(0)             # f₁
qc.rz(1.0, 0)       # fase de f₁

qc.h(1)             # f₂
qc.rz(2.0, 1)       # fase de f₂

qc.h(2)             # f₃
qc.rz(3.0, 2)       # fase de f₃

# Acoplamientos no lineales
qc.cx(0, 1)         # interacción f₁ ↔ f₂
qc.cx(1, 2)         # interacción f₂ ↔ f₃
qc.rz(1.5, 2)       # intermodulación → f₄ emergente

# Medición
qc.measure([0, 1, 2], cr)

# Visualización previa con simulador ideal (Statevector)
qc_no_measure = qc.remove_final_measurements(inplace=False)
state = Statevector.from_instruction(qc_no_measure)

print("\n🔍 Amplitudes complejas del estado cuántico (3 qubits):")
for i, amp in enumerate(state.data):
    bin_state = format(i, f'0{n}b')
    mag = np.abs(amp)
    phase = np.angle(amp)
    print(f"|{bin_state}⟩: amplitud = {amp:.3f}, |amplitud|² = {mag**2:.3f}, fase = {phase:.3f} rad")

# Mostrar gráfico City del estado
plot_state_city(state, title="Estado Cuántico antes de Medición (3 qubits)")
plt.show()

# Ejecución en backend real de IBM
backend = service.least_busy(operational=True, simulator=False)

print("Gates compatibles con el backend:", backend.configuration().basis_gates)

transpiled_qc = transpile(qc, backend=backend, optimization_level=2)
print("Circuito original:")
print(qc)
print("\nCircuito transpileado:")
print(transpiled_qc)

with Session(backend=backend) as session:
    sampler = Sampler(mode=session)
    job = sampler.run([transpiled_qc])
    result = job.result()
    counts = result[0].data.cr.get_counts()

print("Distribución cuántica medida:", counts)

# Gráfico final
plt.bar(counts.keys(), counts.values(), color='royalblue')
plt.xlabel('Estados de salida')
plt.ylabel('Probabilidad')
plt.title('Intermodulación Vibracional Cuántica (3 Qubits)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()