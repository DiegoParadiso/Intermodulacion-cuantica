'''
Simulación de un sistema cuántico con 20 qubits para analizar intermodulación vibracional mediante:
    -circuitos
    -FFT
    -correlaciones
'''

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector
from dotenv import load_dotenv
import os
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import combinations

# Cargar API Key desde archivo .env
load_dotenv()
API_KEY = os.getenv("IBM_QUANTUM_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró la API Key en el archivo .env")

# Conectar con IBM Quantum
service = QiskitRuntimeService(channel="ibm_quantum", token=API_KEY)

# Crear circuito cuántico con 20 qubits
n = 20  # Aumentamos los qubits a 20
qc = QuantumCircuit(n)
cr = ClassicalRegister(n, name='cr')
qc.add_register(cr)

# Vibraciones base (frecuencias iniciales)
for i in range(n):
    qc.h(i)             # Aplicamos Hadamard a todos los qubits
    qc.rz(1.0 * (i + 1), i)  # Rotación de fase con frecuencias crecientes

# Modos emergentes (intermodulación)
for i in range(n//2, n):
    qc.h(i)             # Aplicamos Hadamard a la mitad superior de los qubits
    qc.rz(1.5 * (i + 1), i)  # Rotación de fase con frecuencias más altas

# Entrelazamiento (más interacciones)
for i in range(n - 1):
    qc.cx(i, i + 1)

# Agregar más puertas CNOT para interacciones no lineales
for i in range(n - 2):
    qc.cx(i, i + 2)

# Aumentar las rotaciones de fase para generar más interferencia
for i in range(n):
    qc.rz(0.5 * (i + 1), i)  # Ajusta las frecuencias

# Agregar más interacciones no lineales
for i in range(n - 1):
    qc.cx(i, (i + 1) % n)

# Medición
qc.measure([i for i in range(n)], cr)

# Simulación previa sin mediciones
qc_no_measure = qc.remove_final_measurements(inplace=False)
state = Statevector.from_instruction(qc_no_measure)

# Ejecutar en backend real
backend = service.least_busy(operational=True, simulator=False)
transpiled_qc = transpile(qc, backend=backend, optimization_level=2)

with Session(backend=backend) as session:
    sampler = Sampler(mode=session)
    job = sampler.run([transpiled_qc])
    result = job.result()
    counts = result[0].data.cr.get_counts()

# Gráfico de resultados
plt.bar(counts.keys(), counts.values(), color='royalblue')
plt.xlabel('Estados de salida')
plt.ylabel('Probabilidad')
plt.title('Intermodulación Vibracional Cuántica (20 Qubits)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# FFT (análisis de frecuencias)
n_states = 2**n
sorted_states = [format(i, f'0{n}b') for i in range(n_states)]
probs = np.array([counts.get(state, 0) for state in sorted_states], dtype=np.float64)
probs /= probs.sum()
fft_result = fft(probs)
frequencies = fftfreq(n_states)
amplitudes = np.abs(fft_result)
half = n_states // 2

plt.figure(figsize=(8, 4))
plt.stem(frequencies[:half], amplitudes[:half])
plt.title("Espectro de Fourier de la Intermodulación Cuántica")
plt.xlabel("Frecuencia (modo armónico)")
plt.ylabel("Amplitud")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Modos armónicos dominantes
threshold = 0.05 * np.max(amplitudes)
dominant_modes = [(frequencies[i], amplitudes[i]) for i in range(half) if amplitudes[i] > threshold]

print("\nModos armónicos dominantes (FFT):")
for f, a in dominant_modes:
    print(f"Frecuencia: {f:.3f}, Amplitud: {a:.3f}")

# Estados más frecuentes tras medición
print("\nEstados más frecuentes tras medición: ")
top_states = sorted(counts.items(), key=lambda x: -x[1])[:10]
for k, v in top_states:
    print(f"{k}: {v}")

# Calcular distancias de Hamming entre los estados más frecuentes
def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

print("\nDistancias de Hamming entre los estados más frecuentes:")
for (s1, _), (s2, _) in combinations(top_states, 2):
    print(f"Hamming({s1}, {s2}) = {hamming_distance(s1, s2)}")

# Crear un heatmap de correlación entre los qubits
measurements = list(counts.keys())
measurement_data = np.array([list(map(int, state)) for state in measurements])

# Crear una matriz de correlación
correlation_matrix = np.corrcoef(measurement_data.T)  # Transponer para correlacionar qubits entre sí

# Crear el heatmap de las correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=[f'Q{i}' for i in range(n)], yticklabels=[f'Q{i}' for i in range(n)], linewidths=0.5)
plt.title("Mapa de Calor de Correlaciones entre Qubits")
plt.xlabel("Qubits")
plt.ylabel("Qubits")
plt.tight_layout()
plt.show()
