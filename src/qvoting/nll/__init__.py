"""
qvoting.nll
-----------
Neural Linked List (NLL): quantum data structure with trainable node parameters.

Extends QLL (hierarchical majority voter DAG) by replacing fixed MAJ_k gates
with parameterized RY(θ_i) + CNOT node encoders, enabling adaptive quantum memory.

Main components
---------------
NLLNode         : single 2-qubit trainable node
NeuralLinkedList: full NLL circuit builder (n nodes, trainable θ and φ)
SPSATrainer     : SPSA optimizer for NLL parameters
pattern_recovery_rate : evaluate P(target_pattern) from measurement counts
"""
from .node import NLLNode
from .circuit import NeuralLinkedList, pattern_recovery_rate
from .trainer import SPSATrainer

__all__ = [
    "NLLNode",
    "NeuralLinkedList",
    "SPSATrainer",
    "pattern_recovery_rate",
]
