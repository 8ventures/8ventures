import {
  add,
  addInPlace,
  ensureMetric,
  matVec,
  norm,
  quadraticForm,
  subtract,
  subtractInPlace,
  zeros,
} from "./math.js";
import type { Matrix } from "./math.js";
import type {
  Edge,
  EthicConstraint,
  Graph,
  KernelDynamicsConfig,
  KernelGradients,
  KernelStepResult,
  MirrorDistributions,
  MultiscaleConfig,
} from "./types.js";

const EPSILON = 1e-12;

function getStates(graph: Graph, override?: number[][]): number[][] {
  if (override) {
    return override;
  }
  return graph.nodes.map((node) => node.state);
}

function getPhases(graph: Graph, override?: Array<number | undefined>): Array<number | undefined> {
  if (override) {
    return override;
  }
  return graph.nodes.map((node) => node.phase);
}

export function pairDifference(states: number[][], edge: Edge): number[] {
  return subtract(states[edge.source], states[edge.target]);
}

export function pairEnergy(states: number[][], edge: Edge): number {
  const dimension = states[edge.source].length;
  const Q = ensureMetric(edge.metric, dimension);
  const delta = pairDifference(states, edge);
  return edge.weight * quadraticForm(delta, Q);
}

export function differenceTensor(
  graph: Graph,
  overrideStates?: number[][],
): number[][] {
  const states = getStates(graph, overrideStates);
  return graph.edges.map((edge) => pairDifference(states, edge));
}

export function fieldEnergy(graph: Graph, overrideStates?: number[][]): number {
  const states = getStates(graph, overrideStates);
  let energy = 0;
  for (const edge of graph.edges) {
    energy += pairEnergy(states, edge);
  }
  return 0.5 * energy;
}

export function pairResonance(
  graph: Graph,
  states: number[][],
  phases: Array<number | undefined>,
  edge: Edge,
): number {
  const phaseSource = phases[edge.source];
  const phaseTarget = phases[edge.target];
  if (phaseSource !== undefined && phaseTarget !== undefined) {
    return Math.cos(phaseSource - phaseTarget);
  }
  const xi = states[edge.source];
  const xj = states[edge.target];
  const norms = norm(xi) * norm(xj);
  if (norms < EPSILON) {
    return 0;
  }
  let dot = 0;
  for (let d = 0; d < xi.length; d += 1) {
    dot += xi[d] * xj[d];
  }
  return dot / norms;
}

export function resonanceMatrix(
  graph: Graph,
  overrideStates?: number[][],
  overridePhases?: Array<number | undefined>,
): number[][] {
  const states = getStates(graph, overrideStates);
  const phases = getPhases(graph, overridePhases);
  const N = states.length;
  const matrix = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  for (const edge of graph.edges) {
    const value = pairResonance(graph, states, phases, edge);
    matrix[edge.source][edge.target] = value;
    matrix[edge.target][edge.source] = value;
  }
  return matrix;
}

export function fieldCoherence(
  graph: Graph,
  overrideStates?: number[][],
  overridePhases?: Array<number | undefined>,
): number {
  const states = getStates(graph, overrideStates);
  const phases = getPhases(graph, overridePhases);
  let weightedSum = 0;
  let weightTotal = 0;
  for (const edge of graph.edges) {
    const res = pairResonance(graph, states, phases, edge);
    weightedSum += edge.weight * res;
    weightTotal += edge.weight;
  }
  if (weightTotal === 0) {
    return 0;
  }
  return weightedSum / weightTotal;
}

export function energyGradient(graph: Graph, overrideStates?: number[][]): number[][] {
  const states = getStates(graph, overrideStates);
  if (states.length === 0) {
    return [];
  }
  const dimension = states[0].length;
  const gradients = states.map((state) => zeros(state.length));
  for (const edge of graph.edges) {
    const delta = pairDifference(states, edge);
    const Q = ensureMetric(edge.metric, dimension);
    const tension = matVec(Q, delta);
    const scaled = tension.map((value) => value * edge.weight);
    addInPlace(gradients[edge.source], scaled);
    subtractInPlace(gradients[edge.target], scaled);
  }
  return gradients;
}

function formGradients(
  graph: Graph,
  potentials: KernelDynamicsConfig["potentials"],
  overrideStates?: number[][],
): number[][] {
  const states = getStates(graph, overrideStates);
  if (!potentials) {
    return states.map((state) => zeros(state.length));
  }
  return states.map((state, idx) => {
    const potential = potentials[idx];
    if (!potential) {
      return zeros(state.length);
    }
    return potential.gradient(state);
  });
}

function combineGradients(energyGrad: number[][], formGrad: number[][]): number[][] {
  return energyGrad.map((grad, idx) => {
    const combined = grad.slice();
    const form = formGrad[idx];
    for (let d = 0; d < combined.length; d += 1) {
      combined[d] += form[d];
    }
    return combined;
  });
}

function applyProjection(
  config: KernelDynamicsConfig,
  candidateStates: number[][],
): number[][] {
  if (!config.identityManifold) {
    return candidateStates;
  }
  return candidateStates.map((state, idx) =>
    config.identityManifold!.project(state, config.graph.nodes[idx]),
  );
}

function collectKernelGradients(
  energyGrad: number[][],
  formGrad: number[][],
  totalGrad: number[][],
): KernelGradients {
  return {
    energy: energyGrad,
    form: formGrad,
    total: totalGrad,
  };
}

export function kernelStep(config: KernelDynamicsConfig): KernelStepResult {
  const states = getStates(config.graph);
  if (states.length === 0) {
    return {
      halfStepStates: [],
      nextStates: [],
      noise: undefined,
      control: undefined,
      energy: 0,
      coherence: 0,
      resonanceMatrix: [],
      gradients: { energy: [], form: [], total: [] },
      updatedGraph: config.graph,
    };
  }

  const energyGrad = energyGradient(config.graph, states);
  const formGrad = formGradients(config.graph, config.potentials, states);
  const totalGrad = combineGradients(energyGrad, formGrad);

  const dimension = states[0].length;
  const control = config.controlPolicy
    ? config.controlPolicy(states, config.time ?? 0)
    : undefined;
  const noise = config.explorationNoise
    ? states.map((_, idx) => config.explorationNoise!(idx, dimension))
    : undefined;

  const halfStepStates = states.map((state, idx) => {
    let updated = state.map(
      (value, d) => value - config.stepSize * totalGrad[idx][d],
    );
    if (control) {
      updated = add(updated, control[idx]);
    }
    if (noise) {
      updated = add(updated, noise[idx]);
    }
    return updated;
  });

  const updatedGraph = config.updateCouplings
    ? config.updateCouplings(halfStepStates, config.graph)
    : config.graph;

  const projectedStates = applyProjection(config, halfStepStates);

  const energy = fieldEnergy(updatedGraph, projectedStates);
  const coherence = fieldCoherence(updatedGraph, projectedStates);
  const resonance = resonanceMatrix(updatedGraph, projectedStates);

  return {
    halfStepStates,
    nextStates: projectedStates,
    noise,
    control,
    energy,
    coherence,
    resonanceMatrix: resonance,
    gradients: collectKernelGradients(energyGrad, formGrad, totalGrad),
    updatedGraph,
  };
}

export function adaptiveWeightUpdate(
  weight: number,
  resonance: number,
  averageResonance: number,
  {
    learningRate,
    decay,
    benefit,
  }: {
    learningRate: number;
    decay: number;
    benefit: boolean;
  },
): number {
  const delta = learningRate * (resonance - averageResonance) * (benefit ? 1 : 0);
  return Math.max(0, weight + delta - decay * weight);
}

export function averageResonanceByNode(
  graph: Graph,
  overrideStates?: number[][],
  overridePhases?: Array<number | undefined>,
): number[] {
  const states = getStates(graph, overrideStates);
  const phases = getPhases(graph, overridePhases);
  const sums = new Array<number>(states.length).fill(0);
  const weights = new Array<number>(states.length).fill(0);
  for (const edge of graph.edges) {
    const resonance = pairResonance(graph, states, phases, edge);
    const weight = edge.weight;
    sums[edge.source] += weight * resonance;
    sums[edge.target] += weight * resonance;
    weights[edge.source] += weight;
    weights[edge.target] += weight;
  }
  return sums.map((sum, idx) => (weights[idx] > 0 ? sum / weights[idx] : 0));
}

export function createHebbianUpdater({
  learningRate,
  decay,
  benefitPredicate,
}: {
  learningRate: number;
  decay: number;
  benefitPredicate?: (context: {
    edge: Edge;
    resonance: number;
    averageSource: number;
    averageTarget: number;
  }) => boolean;
}): (states: number[][], graph: Graph) => Graph {
  return (states, graph) => {
    const phases = getPhases(graph);
    const averages = averageResonanceByNode(graph, states, phases);
    const edges = graph.edges.map((edge) => {
      const resonance = pairResonance(graph, states, phases, edge);
      const avgSource = averages[edge.source];
      const avgTarget = averages[edge.target];
      const baseline = (avgSource + avgTarget) / 2;
      const benefit = benefitPredicate
        ? benefitPredicate({ edge, resonance, averageSource: avgSource, averageTarget: avgTarget })
        : resonance >= baseline;
      const updatedWeight = adaptiveWeightUpdate(edge.weight, resonance, baseline, {
        learningRate,
        decay,
        benefit,
      });
      return { ...edge, weight: updatedWeight };
    });
    return { ...graph, edges };
  };
}

export function gradientMetricUpdate(
  metric: Matrix,
  gradient: Matrix,
  stepSize: number,
): Matrix {
  const updated = metric.map((row, i) =>
    row.map((value, j) => value - stepSize * (gradient[i]?.[j] ?? 0)),
  );
  for (let i = 0; i < updated.length; i += 1) {
    for (let j = i + 1; j < updated[i].length; j += 1) {
      const symValue = 0.5 * (updated[i][j] + updated[j][i]);
      updated[i][j] = symValue;
      updated[j][i] = symValue;
    }
  }
  return updated;
}

export function mirrorLoss({
  world,
  model,
  epsilon = 1e-9,
}: MirrorDistributions): number {
  if (world.length !== model.length) {
    throw new Error("World and model distributions must have equal length");
  }
  let loss = 0;
  for (let i = 0; i < world.length; i += 1) {
    const pw = Math.max(world[i], epsilon);
    const pm = Math.max(model[i], epsilon);
    loss += pw * Math.log(pw / pm);
  }
  return loss;
}

export function enforceEthics(
  constraints: EthicConstraint[],
  config: MultiscaleConfig,
  overrideStates?: number[][],
): boolean {
  const stateLevels = config.levels;
  for (const constraint of constraints) {
    const level = stateLevels.find(
      (candidate) => candidate.label === constraint.levelLabel,
    );
    if (!level) {
      continue;
    }
    const coherence = fieldCoherence(
      level.graph,
      overrideStates ? level.projector(overrideStates) : undefined,
    );
    if (coherence < constraint.minimumCoherence) {
      return false;
    }
  }
  return true;
}

export function multiscaleCoherence(
  config: MultiscaleConfig,
  overrideStates?: number[][],
): number {
  let totalWeight = 0;
  let weightedSum = 0;
  for (const level of config.levels) {
    const states = overrideStates
      ? level.projector(overrideStates)
      : level.projector(level.graph.nodes.map((node) => node.state));
    const coherence = fieldCoherence(level.graph, states);
    const weight = level.weight ?? 1;
    weightedSum += weight * coherence;
    totalWeight += weight;
  }
  if (totalWeight === 0) {
    return 0;
  }
  return weightedSum / totalWeight;
}

export function lyapunovFunctional(
  graph: Graph,
  alpha: number,
  beta: number,
  lambda: number,
  overrideStates?: number[][],
): number {
  const energy = fieldEnergy(graph, overrideStates);
  const coherence = fieldCoherence(graph, overrideStates);
  const weightNorm = Math.sqrt(
    graph.edges.reduce((acc, edge) => acc + edge.weight * edge.weight, 0),
  );
  return beta * energy - alpha * coherence + (lambda / 2) * weightNorm * weightNorm;
}

export function analyzeField(
  graph: Graph,
  overrideStates?: number[][],
  overridePhases?: Array<number | undefined>,
): {
  energy: number;
  coherence: number;
  resonanceMatrix: number[][];
} {
  const energy = fieldEnergy(graph, overrideStates);
  const coherence = fieldCoherence(graph, overrideStates, overridePhases);
  const resonance = resonanceMatrix(graph, overrideStates, overridePhases);
  return { energy, coherence, resonanceMatrix: resonance };
}
