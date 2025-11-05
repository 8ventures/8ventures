import type { Matrix, Vector } from "./math.js";

export interface Agent {
  id: string | number;
  state: Vector;
  amplitude?: number;
  phase?: number;
  layer?: number;
  metadata?: Record<string, unknown>;
}

export interface Edge {
  source: number;
  target: number;
  weight: number;
  metric?: Matrix;
  layer?: number;
}

export interface Graph {
  nodes: Agent[];
  edges: Edge[];
}

export type PotentialEnergy = (x: Vector) => number;
export type PotentialGradient = (x: Vector) => Vector;

export interface LocalPotential {
  energy?: PotentialEnergy;
  gradient: PotentialGradient;
}

export interface IdentityManifold {
  project: (state: Vector, agent: Agent) => Vector;
}

export type NoiseSampler = (agentIndex: number, dimension: number) => Vector;

export type ControlPolicy = (X: Vector[], time: number) => Vector[];

export interface KernelDynamicsConfig {
  graph: Graph;
  potentials?: (LocalPotential | undefined)[];
  identityManifold?: IdentityManifold;
  explorationNoise?: NoiseSampler;
  controlPolicy?: ControlPolicy;
  updateCouplings?: (X: Vector[], graph: Graph) => Graph;
  stepSize: number;
  time?: number;
}

export interface KernelGradients {
  energy: Vector[];
  form: Vector[];
  total: Vector[];
}

export interface KernelStepResult {
  halfStepStates: Vector[];
  nextStates: Vector[];
  noise?: Vector[];
  control?: Vector[];
  energy: number;
  coherence: number;
  resonanceMatrix: Matrix;
  gradients: KernelGradients;
  updatedGraph: Graph;
}

export interface CoarseGrainingLevel {
  projector: (X: Vector[]) => Vector[];
  graph: Graph;
  weight?: number;
  label?: string;
}

export interface MultiscaleConfig {
  levels: CoarseGrainingLevel[];
}

export interface MirrorDistributions {
  world: number[];
  model: number[];
  epsilon?: number;
}

export interface EthicConstraint {
  levelLabel: string;
  minimumCoherence: number;
}
