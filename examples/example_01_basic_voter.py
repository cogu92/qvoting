"""
Ejemplo 1: Votador de mayoría con error mitigation en simulación
================================================================

Demuestra el flujo completo:
  1. Crear circuito votador
  2. Simular con ruido
  3. Aplicar readout mitigation
  4. Comparar resultados
"""

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

from qvoting.voters import majority_voter
from qvoting.mitigation.readout import confusion_matrix, apply_readout_mitigation
from qvoting.core.execution import execute_circuit

# ── 1. Crear circuito ────────────────────────────────────────────────────
print("=== QVoting — Ejemplo básico ===\n")
qc = majority_voter(num_inputs=3)
print("Circuito votador de mayoría (3 entradas):")
print(qc.draw(output="text"))

# ── 2. Backend ideal ─────────────────────────────────────────────────────
sim_ideal = AerSimulator()
counts_ideal = execute_circuit(qc, sim_ideal, shots=1024)
print("\n📊 Resultado ideal:")
for state, freq in sorted(counts_ideal.items(), key=lambda x: -x[1]):
    print(f"  |{state}⟩ → {freq:4d} ({freq/1024*100:.1f}%)")

# ── 3. Simular con ruido de lectura ──────────────────────────────────────
M = confusion_matrix(prob_0_to_1=0.02, prob_1_to_0=0.05)

noise_model = NoiseModel()
noise_model.add_all_qubit_readout_error(ReadoutError([
    [M[0, 0], M[1, 0]],
    [M[0, 1], M[1, 1]],
]))
sim_noisy = AerSimulator(noise_model=noise_model)

counts_noisy = execute_circuit(qc, sim_noisy, shots=1024)
print("\n📊 Resultado con ruido:")
for state, freq in sorted(counts_noisy.items(), key=lambda x: -x[1]):
    print(f"  |{state}⟩ → {freq:4d} ({freq/1024*100:.1f}%)")

# ── 4. Aplicar mitigación ─────────────────────────────────────────────────
counts_mitigated = apply_readout_mitigation(counts_noisy, M, num_qubits=1)
print("\n📊 Resultado mitigado:")
for state, freq in sorted(counts_mitigated.items(), key=lambda x: -x[1]):
    print(f"  |{state}⟩ → {freq:4d} ({freq/sum(counts_mitigated.values())*100:.1f}%)")

# ── 5. Comparación ────────────────────────────────────────────────────────
total = 1024
p1_ideal = counts_ideal.get("1", 0) / total
p1_noisy = counts_noisy.get("1", 0) / total
p1_mit = counts_mitigated.get("1", 0) / sum(counts_mitigated.values())

print("\n📈 Resumen:")
print(f"  P(maj=1) ideal:    {p1_ideal:.4f}")
print(f"  P(maj=1) ruidoso:  {p1_noisy:.4f}")
print(f"  P(maj=1) mitigado: {p1_mit:.4f}")
print(f"  Mejora: {(p1_mit - p1_noisy) / max(1 - p1_noisy, 1e-8) * 100:+.1f}%")
