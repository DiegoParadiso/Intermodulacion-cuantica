"""
Simular cómo tres modos vibracionales base (f₁, f₂, f₃) interactúan no linealmente 
dentro de un espacio-tiempo cuántico, generando nuevos modos emergentes (f₄, f₅) 
mediante un circuito cuántico con 4 qubits.
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

# Crear circuito cuántico con 4 qubits
n = 4
qc = QuantumCircuit(n)
cr = ClassicalRegister(n, name='cr')
qc.add_register(cr)

# Vibraciones base
qc.h(0)             # f₁
qc.rz(1.0, 0)

qc.h(1)             # f₂
qc.rz(2.0, 1)

# Modos emergentes (intermodulación)
qc.h(2)             # pre-f₃
qc.rz(3.0, 2)

qc.h(3)             # pre-f₄
qc.rz(4.0, 3)

# Acoplamientos no lineales (interacciones entre frecuencias)
qc.cx(0, 2)         # f₁ → f₃
qc.cx(1, 2)         # f₂ → f₃
qc.cx(0, 3)         # f₁ → f₄
qc.cx(1, 3)         # f₂ → f₄

# Intermodulación cuántica
qc.rz(1.2, 2)
qc.rz(0.8, 3)

# Medición
qc.measure([0, 1, 2, 3], cr)

# Simulación previa
qc_no_measure = qc.remove_final_measurements(inplace=False)
state = Statevector.from_instruction(qc_no_measure)

print("\n🔍 Amplitudes complejas del estado cuántico (4 qubits):")
for i, amp in enumerate(state.data):
    bin_state = format(i, f'0{n}b')
    mag = np.abs(amp)
    phase = np.angle(amp)
    print(f"|{bin_state}⟩: amplitud = {amp:.3f}, |amplitud|² = {mag**2:.3f}, fase = {phase:.3f} rad")

# Graficar estado
plot_state_city(state, title="Estado Cuántico antes de Medición (4 Qubits)")
plt.show()

# Ejecutar en backend real
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

# Gráfico de resultados
plt.bar(counts.keys(), counts.values(), color='royalblue')
plt.xlabel('Estados de salida')
plt.ylabel('Probabilidad')
plt.title('Intermodulación Vibracional Cuántica (4 Qubits)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()