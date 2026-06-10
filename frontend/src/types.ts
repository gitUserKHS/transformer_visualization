export type Tab =
  | "factory"
  | "microscope"
  | "alignment"
  | "arena"
  | "inference"
  | "evaluation";

export interface ModelConfig {
  vocab_size: number;
  context_length: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  d_ff: number;
  dropout: number;
}

export interface TrainingConfig {
  name: string;
  preset: "inspect" | "balanced" | "deep";
  steps: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  grad_clip: number;
  seed: number;
  model: ModelConfig;
}

export interface Metrics {
  step: number;
  loss: number;
  perplexity: number;
  gradient_norm: number;
  parameter_delta_norm: number;
  learning_rate: number;
  steps_per_second: number;
}

export interface AttentionTrace {
  layer: number;
  head: number;
  token_index: number;
  attention: number[][];
  q: number[];
  keys: number[][];
  values: number[][];
  raw_scores: number[];
  scaled_scores: Array<number | null>;
  weights: number[];
  weighted_value: number[];
}

export interface Trace {
  tokens: string[];
  token_ids: number[];
  target_tokens?: string[];
  top_logits: Array<{
    token: string;
    token_id: number;
    logit: number;
    probability: number;
  }>;
  attention: AttentionTrace;
  embedding: {
    token_vector: number[];
    position_vector: number[];
    sum_vector: number[];
  };
  layer_stats: Array<{ layer: number; mean: number; std: number; norm: number }>;
}

export interface RunSummary {
  id: string;
  name: string;
  status: string;
  step: number;
  total_steps: number;
  config: TrainingConfig;
  losses: Array<{ step: number; loss: number; perplexity: number }>;
  last_metrics: Partial<Metrics>;
  sample: string;
  parameter_count: number;
  created_at: number;
  updated_at?: number;
  stage?: string;
  stage_progress?: number;
  overall_progress?: number;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
  steps_per_second?: number;
  metric_series?: Array<{ step: number; loss: number; perplexity?: number }>;
  error_message?: string | null;
}

export interface Bootstrap {
  dataset: {
    name: string;
    train_count: number;
    validation_count: number;
    bundled: boolean;
    samples: Array<{ index: number; text: string; tokens: string[] }>;
    license: string;
    source: string;
  };
  presets: Record<
    string,
    {
      label: string;
      description: string;
      steps: number;
      batch_size: number;
      learning_rate: number;
    }
  >;
  model: ModelConfig & {
    parameter_count: number;
    checkpoint_ready: boolean;
  };
  history: RunSummary[];
  system: SystemInfo;
  factory: FactoryBootstrap;
}

export interface Generation {
  text: string;
  tokens: string[];
  steps: Array<{
    token: string;
    token_id: number;
    candidates: Array<{ token: string; probability: number }>;
  }>;
  trace: Trace;
}

export interface SystemInfo {
  python: string;
  torch: string;
  torch_cuda: string | null;
  cuda_available: boolean;
  device: string;
  precision: string;
  production_ready: boolean;
  microscope_available: boolean;
  setup_command: string;
  message: string;
  gpu: null | {
    name: string;
    driver_version?: string;
    total_vram_mb?: number;
    free_vram_mb?: number;
    total_vram_bytes?: number;
    free_vram_bytes?: number;
    bf16_supported?: boolean;
  };
}

export interface FactoryBootstrap {
  data: {
    cleaning_trace: Array<{ stage: string; count: number; retention: number }>;
    sft_count: number;
    preference_count: number;
    user_vote_count: number;
    vocab_size: number;
  };
  model: {
    vocab_size: number;
    context_length: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    n_kv_heads: number;
    d_ff: number;
    parameter_count: number;
    kv_cache_128_bytes: number;
    lora: {
      rank: number;
      trainable_parameters: number;
      total_parameters: number;
      trainable_percent: number;
      modules: string[];
    };
  };
  stages: string[];
  history: FactoryRunSummary[];
}

export interface FactoryRunConfig {
  name: string;
  stages: string[];
  pretrain_steps: number;
  sft_steps: number;
  reward_steps: number;
  dpo_steps: number;
  ppo_steps: number;
  grpo_steps: number;
  micro_batch_size: number;
  gradient_accumulation: number;
  learning_rate: number;
  lora_rank: number;
  response_tokens: number;
  group_size: number;
  seed: number;
}

export interface FactoryRunSummary {
  id: string;
  name: string;
  status: string;
  stage: string;
  stage_step: number;
  stage_total: number;
  config: FactoryRunConfig;
  metrics: Record<string, Record<string, unknown>>;
  checkpoints: Record<string, { path: string; bytes: number }>;
  comparisons: Record<string, unknown>;
  peak_vram_bytes: number;
  created_at: number;
  updated_at?: number;
  stage_states?: Record<
    string,
    "pending" | "running" | "completed" | "skipped"
  >;
  stage_progress?: number;
  overall_progress?: number;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
  steps_per_second?: number;
  metric_series?: Array<{
    stage: string;
    step: number;
    loss: number;
    timestamp: number;
  }>;
  error_message?: string | null;
}

export interface DeviceTelemetry {
  available?: boolean;
  source?: string;
  utilization_percent?: number;
  memory_used_bytes?: number;
  memory_total_bytes?: number;
  temperature_c?: number;
  power_w?: number;
  torch_allocated_bytes?: number;
  torch_reserved_bytes?: number;
  device?: string;
}

export interface ActivityResponse {
  server_time: number;
  factory: FactoryRunSummary | null;
  microscope: RunSummary | null;
}

export type ConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "stale";

export interface NeuralTensorStats {
  name: string;
  shape: number[];
  sampled_values: number;
  mean: number;
  std: number;
  rms: number;
  max_abs: number;
}

export interface NeuralBackwardStats {
  name: string;
  gradient_norm: number;
  mean_abs_gradient: number;
  max_abs_gradient: number;
  parameter_tensors: number;
  elements: number;
  sampled_parameters?: number;
  update_rms?: number;
  update_max_abs?: number;
}

export interface LossSurfaceTrace {
  axis: number[];
  losses: number[][];
  center_loss: number;
  minimum_loss: number;
  minimum_at: [number, number];
  direction_scale: number;
  method: string;
}

export interface NeuralTrace {
  stage: string;
  step: number;
  loss: number;
  batch_shape: number[];
  tokens: string[];
  forward: NeuralTensorStats[];
  attention?: {
    q_shape?: number[];
    kv_shape?: number[];
    preview: number[][];
  } | null;
  backward: NeuralBackwardStats[];
  optimizer: {
    name: string;
    learning_rate: number;
    pre_clip_gradient_norm: number;
    clip_limit: number;
    clip_scale: number;
  };
  surface?: LossSurfaceTrace | null;
}

export interface PreferencePair {
  pair_id: string;
  prompt: string;
  response_a: string;
  response_b: string;
  revealed: boolean;
  remaining: number;
}

export interface KVBenchmark {
  cached: {
    text: string;
    elapsed_ms: number;
    tokens_per_second: number;
    steps: Array<{
      index: number;
      token: string;
      input_tokens_computed: number;
      cache_length: number;
      latency_ms: number;
      kv_shape?: number[];
    }>;
  };
  uncached: {
    text: string;
    elapsed_ms: number;
    tokens_per_second: number;
    steps: Array<{
      index: number;
      token: string;
      input_tokens_computed: number;
      cache_length: number;
      latency_ms: number;
    }>;
  };
  tokens_match: boolean;
  speedup: number;
  cache_modes: Array<{ mode: string; kv_heads: number; bytes: number }>;
  continuous_batching: Array<{
    tick: number;
    prefill: string[];
    decode: string[];
    batch_size: number;
  }>;
  note: string;
}
