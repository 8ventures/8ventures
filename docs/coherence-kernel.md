# Coherence Kernel Utilities

This module turns the formal "Coherence Kernel" specification into reusable TypeScript primitives. Each operator in the cycle maps to a composable function so you can plug the kernel into physics, biology, cognition, or software systems.

## Mathematical correspondences

| Concept | Symbol | Code entry point |
| --- | --- | --- |
| Difference tensor | \(\Delta_{ij} = x_i - x_j\) | `differenceTensor(graph)` / `pairDifference(states, edge)` |
| Pair tension energy | \(\varepsilon_{ij} = w_{ij} \, \Vert \Delta_{ij} \Vert_{Q_{ij}}^2\) | `pairEnergy(states, edge)` |
| Field energy | \(\mathcal{E}(X)\) | `fieldEnergy(graph, states?)` |
| Resonance | \(r_{ij}\) | `pairResonance(...)`, `resonanceMatrix(graph)` |
| Field coherence | \(\mathcal{C}(X)\) | `fieldCoherence(graph)` |
| Energy gradient | \(-\nabla_X \mathcal{E}\) | `energyGradient(graph)` |
| Form gradient | \(-\nabla_X \mathcal{E}_{\text{form}}\) | provide `LocalPotential.gradient` |
| Kernel half-step | \(X^{k+\frac{1}{2}}\) | `kernelStep(config).halfStepStates` |
| Identity projection | \(\Pi_{\mathcal{M}}\) | `IdentityManifold.project` callback |
| Hebbian feedback | \(\dot w_{ij}\) | `createHebbianUpdater(options)` |
| Mirror loss | \(\mathcal{L}_{\text{mirror}}\) | `mirrorLoss({ world, model })` |
| Multiscale coherence | \(\mathfrak{C}\) | `multiscaleCoherence(config)` |
| Lyapunov functional | \(\mathcal{L}(X,W,Q)\) | `lyapunovFunctional(graph, alpha, beta, lambda)` |

## Quick start

```ts
import {
  kernelStep,
  createHebbianUpdater,
  fieldEnergy,
  fieldCoherence,
  multiscaleCoherence,
  type Graph,
  type LocalPotential,
  type IdentityManifold,
  type CoarseGrainingLevel,
} from "../dist/index.js";

const graph: Graph = {
  nodes: [
    { id: 0, state: [1, 0] },
    { id: 1, state: [0.5, 0.5] },
    { id: 2, state: [0, 1] },
  ],
  edges: [
    { source: 0, target: 1, weight: 1 },
    { source: 1, target: 2, weight: 1 },
    { source: 0, target: 2, weight: 0.5 },
  ],
};

const potentials: LocalPotential[] = [
  { gradient: (x) => [x[0] - 1, x[1]] },
  undefined,
  { gradient: (x) => [x[0], x[1] - 1] },
];

const identityManifold: IdentityManifold = {
  project: (state) => {
    const norm = Math.hypot(...state);
    return norm === 0 ? state : state.map((value) => value / norm);
  },
};

const hebbianFeedback = createHebbianUpdater({
  learningRate: 0.05,
  decay: 0.01,
});

const multiscale: CoarseGrainingLevel[] = [
  {
    label: "base",
    weight: 1,
    graph,
    projector: (states) => states,
  },
];

const result = kernelStep({
  graph,
  stepSize: 0.1,
  potentials,
  identityManifold,
  updateCouplings: hebbianFeedback,
});

console.log("Energy", fieldEnergy(result.updatedGraph, result.nextStates));
console.log("Coherence", fieldCoherence(result.updatedGraph, result.nextStates));
console.log(
  "Multiscale",
  multiscaleCoherence({ levels: multiscale }, result.nextStates),
);
```

## Reflexivity hooks

- **Mirror loss:** Supply empirical and model predictive distributions to `mirrorLoss` to keep the self-model calibrated.
- **Control policy:** Pass a `controlPolicy` callback into `kernelStep` to inject awareness-driven actions \(u(X)\).

## Ethics & guardrails

Define coarse-graining levels (communities, wavelets, spatial bins) and enforce constraints with `enforceEthics`. Each constraint declares a minimum acceptable coherence on that scale.

```ts
import { enforceEthics, type EthicConstraint } from "../dist/index.js";

const constraints: EthicConstraint[] = [
  { levelLabel: "base", minimumCoherence: 0.2 },
];

const ok = enforceEthics(constraints, { levels: multiscale }, result.nextStates);
if (!ok) {
  throw new Error("Intervention would violate coherence guardrail");
}
```

The module stays agnostic about how you generate coarse states or policy gradients, making it easy to integrate with simulators, reinforcement learning, or distributed control stacks.
