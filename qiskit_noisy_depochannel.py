import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator, Aer
from qiskit_aer.library import save_statevector
#from qiskit.providers.aer.library import save_statevector
#import qiskit_aer.noise as noise
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
#from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import SamplerV2


### Create the Bell circuit
def Bell_QC() -> QuantumCircuit:

    qr = QuantumRegister(2, name="Qbit")
    cr = ClassicalRegister(2, name="Cbit")

    qc = QuantumCircuit(qr, cr, name="Bell circuit")

    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    qc.barrier()

    qc.save_statevector()

    qc.measure(range(qr.size), range(cr.size))
    
    #qc.measure_all()

    return qc



def Bell_QC_Identities():

    qr = QuantumRegister(2, name="Qubit")
    cr = ClassicalRegister(2, name="Cbit")

    qc = QuantumCircuit(qr, cr)

    angle = np.pi / 5
    #### ci mettiamo tante identità per realizzare il noise
    qc.barrier()

    for k in range(100):
        for i in range(qr.size): ### ho aggiunto queste controlled ry per creare fake noise e funziona
            qc.cry(angle, qr[0], qr[1])
            qc.cry(angle, qr[1], qr[0])
            qc.cx(qr[0], qr[1])
            qc.cx(qr[0], qr[1])
            qc.cry(-angle, qr[1], qr[0])
            qc.cry(-angle, qr[0], qr[1])
    
    qc.barrier()

    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    qc.barrier()

    #qc.save_statevector()
    #qc.measure(range(qr.size), range(cr.size))
    qc.measure_all()  ### con qc.measure() non funziona il count per questo circuito

    return qc


def DepoChannel(lam1: float, lam2: float, qubits_with_error1: int, qubits_with_error2: int):

    depo_err_chan1 = depolarizing_error(lam1, qubits_with_error1)
    depo_err_chan2 = depolarizing_error(lam2, qubits_with_error2)
    
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depo_err_chan1, ['u2'])  ### you can see which are the basics gates computing 
    # the decomposition
    noise_model.add_all_qubit_quantum_error(depo_err_chan2, ['cx'])

    return noise_model



shots = 8192
qc_id = Bell_QC().decompose()

#qc_id.decompose().draw("mpl")  in this way you can see which is the gate combinations to reproduce the Hadamard for the
# specific simulator you have chosen

l1 = 1e-2
l2 = 1

### DEPOLARIZING ###
noise_depolarizing = DepoChannel(l1, l2, 1, 2)
sim_noise = AerSimulator(noise_model=noise_depolarizing)
circ_noise = transpile(qc_id, backend=sim_noise)  ### on the ideal decomposed circuit
result_depochannel = sim_noise.run(circ_noise, shots=shots).result()
counts_depochannel = result_depochannel.get_counts(0)

sv1 = result_depochannel.data(0)['statevector']

### IDEAL CIRCUIT ###
sim_id = AerSimulator()
circ_id = transpile(qc_id, backend=sim_id)
result_id = sim_id.run(circ_id, shots=8192).result()
counts_ideal = result_id.get_counts(0)

sv2 = result_id.data(0)['statevector']


### FAKE NOISE CIRCUIT ###
backend_fake = FakeManilaV2()

deco_id = Bell_QC_Identities().decompose()


circ_noisy_fake = transpile(deco_id, backend=backend_fake)
sampler = SamplerV2(backend_fake)
job = sampler.run([circ_noisy_fake], shots=shots).result()
#fake_result = job.result()[0]
counts_noisy_fake = job[0].data.meas.get_counts()
#sv3 = job[0].get_statevector()

'''statevector_simulator = Aer.get_backend('statevector_simulator')
job_sv = statevector_simulator.run(transpile(deco_id, statevector_simulator), shots=shots)
result_sv = job_sv.result()
statevector = result_sv.get_statevector()'''


if __name__ == "__main__":

    print("Sate vectors:")
    #print(sv1), print(sv2), print(sv3)

    legend = ['Depolarizing', 'Ideal', 'NoisyFake']
    plot_histogram([counts_depochannel, counts_ideal, counts_noisy_fake], legend=legend)
    plt.show()
