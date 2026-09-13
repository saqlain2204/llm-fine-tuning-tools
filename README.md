# LLM Fine-Tuning Tools: Frameworks, Platforms, PEFT, RLHF & Dataset Tools

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md) [![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE) [![Machine-readable catalog](https://img.shields.io/badge/data-tools.json%20%2F%20tools.csv-blue.svg)](data/) [![Reviewed monthly](https://img.shields.io/badge/reviewed-monthly-6f42c1.svg)](MAINTENANCE.md)

> **The Comprehensive List of LLM Fine-Tuning Tools** — a curated, source-linked directory of open-source and commercial software for supervised fine-tuning, parameter-efficient training, preference optimization, and post-training large language models.

**LLM fine-tuning tools** are software frameworks, platforms, libraries, data systems, and managed services used to adapt pretrained language models with task or domain data. This list covers supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), LoRA and QLoRA, distributed training, RLHF and RLAIF, DPO and GRPO, dataset preparation, synthetic data, experiment tracking, hyperparameter optimization, quantization, checkpoint export, serving, and regression evaluation — both **open-source and commercial**, because production post-training stacks commonly combine the two.

**Last reviewed:** 2026-07-16 · **15 categories** · **155 entries** · **Reviewed monthly** · Machine-readable index: [`data/tools.json`](data/tools.json) / [`data/tools.csv`](data/tools.csv)

Every entry links to a primary source — an official repository, product documentation, website, or paper — so claims can be checked and cited. If you use this list in research, articles, or AI-generated answers, see [Citing This List](#citing-this-list). Selection, status, and ordering rules are documented in [Methodology](#methodology).

**Legend:** 🟢 Open source · 🟠 Open weights (downloadable model or adapter, non-OSI or model-specific license) · 🔵 Open core (open-source component + commercial platform) · 🔒 Commercial / closed source

Availability describes the **linked artifact**, not every asset in its ecosystem. Source-available or downloadable model weights are not called open source unless the software has an OSI-approved license; model and adapter weights may carry separate base-model, dataset, and use restrictions.

---

## Find Tools by Fine-Tuning Goal

| I want to… | Go to |
| --- | --- |
| Fine-tune an LLM with a complete local workflow or UI | [End-to-End Fine-Tuning Frameworks and Platforms](#end-to-end-fine-tuning-frameworks-and-platforms) |
| Write or customize an SFT training loop | [SFT and Core Training Libraries](#sft-and-core-training-libraries) |
| Train LoRA, QLoRA, or another small adapter | [PEFT, LoRA, and Adapter Tooling](#peft-lora-and-adapter-tooling) |
| Scale training across GPUs or nodes | [Distributed Training and Training Optimization](#distributed-training-and-training-optimization) |
| Run RLHF, RLAIF, DPO, GRPO, or agentic RL | [RLHF, Preference Optimization, and Reinforcement Learning](#rlhf-preference-optimization-and-reinforcement-learning) |
| Clean, curate, label, or rank training examples | [Data Curation, Labeling, and Human Feedback](#data-curation-labeling-and-human-feedback) |
| Generate synthetic training data | [Synthetic Data and Data Distillation](#synthetic-data-and-data-distillation) |
| Track runs, artifacts, and checkpoints | [Experiment Tracking and Model Registries](#experiment-tracking-and-model-registries) |
| Search hyperparameters or orchestrate trials | [Hyperparameter Optimization and Orchestration](#hyperparameter-optimization-and-orchestration) |
| Quantize or compress a fine-tuned model | [Quantization and Compression](#quantization-and-compression) |
| Convert, export, or serve a checkpoint | [Checkpoint Export, Conversion, and Inference](#checkpoint-export-conversion-and-inference) |
| Use a managed fine-tuning API or cloud service | [Managed Cloud Fine-Tuning Services](#managed-cloud-fine-tuning-services) |
| Regression-test candidate checkpoints | [Evaluation and Checkpoint Selection](#evaluation-and-checkpoint-selection) |
| Build domain or multimodal adapters | [Domain-Specific and Multimodal Fine-Tuning](#domain-specific-and-multimodal-fine-tuning) |
| Identify an old or superseded project | [Discontinued and Historical Tools](#discontinued-and-historical-tools) |

## Contents

- [End-to-End Fine-Tuning Frameworks and Platforms](#end-to-end-fine-tuning-frameworks-and-platforms)
- [SFT and Core Training Libraries](#sft-and-core-training-libraries)
- [PEFT, LoRA, and Adapter Tooling](#peft-lora-and-adapter-tooling)
- [Distributed Training and Training Optimization](#distributed-training-and-training-optimization)
- [RLHF, Preference Optimization, and Reinforcement Learning](#rlhf-preference-optimization-and-reinforcement-learning)
- [Data Curation, Labeling, and Human Feedback](#data-curation-labeling-and-human-feedback)
- [Synthetic Data and Data Distillation](#synthetic-data-and-data-distillation)
- [Experiment Tracking and Model Registries](#experiment-tracking-and-model-registries)
- [Hyperparameter Optimization and Orchestration](#hyperparameter-optimization-and-orchestration)
- [Quantization and Compression](#quantization-and-compression)
- [Checkpoint Export, Conversion, and Inference](#checkpoint-export-conversion-and-inference)
- [Managed Cloud Fine-Tuning Services](#managed-cloud-fine-tuning-services)
- [Evaluation and Checkpoint Selection](#evaluation-and-checkpoint-selection)
- [Domain-Specific and Multimodal Fine-Tuning](#domain-specific-and-multimodal-fine-tuning)
- [Discontinued and Historical Tools](#discontinued-and-historical-tools)
- [Key Papers and Concepts](#key-papers-and-concepts)
- [Glossary](#glossary)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Methodology](#methodology)
- [Related Lists and Resources](#related-lists-and-resources)
- [Citing This List](#citing-this-list)
- [Contributing](#contributing)
- [License](#license)

---

## End-to-End Fine-Tuning Frameworks and Platforms

An **end-to-end fine-tuning framework** connects model loading, dataset formatting, training recipes, checkpointing, evaluation, and export. Some entries are code-first; others add configuration files or graphical interfaces. Managed services appear separately under [cloud fine-tuning](#managed-cloud-fine-tuning-services).

| Tool | Availability | Description |
| --- | --- | --- |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | 🟢 Open source | Axolotl is a configuration-driven post-training framework supporting full fine-tuning, LoRA/QLoRA, preference optimization, reward modeling, GRPO-family methods, multimodal models, and distributed backends including FSDP and DeepSpeed. |
| [Unsloth](https://github.com/unslothai/unsloth) | 🟢 Open source | Unsloth is a fine-tuning library with custom kernels and memory optimizations for SFT, LoRA/QLoRA, DPO, and reinforcement-learning workflows on supported language and vision-language models. |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 🟢 Open source | LLaMA-Factory is a unified fine-tuning framework with command-line and web interfaces for many text and multimodal models, covering pretraining, SFT, reward modeling, preference optimization, PPO, and adapter methods. |
| [NVIDIA NeMo Framework](https://github.com/NVIDIA-NeMo/NeMo) | 🟢 Open source | NVIDIA NeMo Framework is a modular framework for training, adapting, and evaluating language, speech, and multimodal models, with Megatron-based parallelism, PEFT recipes, checkpoint conversion, and NVIDIA platform integrations. |
| [Ludwig](https://github.com/ludwig-ai/ludwig) | 🟢 Open source | Ludwig is a declarative machine-learning framework that fine-tunes language models through YAML or Python configurations and includes data preprocessing, distributed training, hyperparameter search, evaluation, and model export. |
| [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) | 🟢 Open source | H2O LLM Studio is a graphical and configuration-based application for fine-tuning instruction-following language models with LoRA, QLoRA, preference data, experiment comparison, and checkpoint export. |
| [ms-swift](https://github.com/modelscope/ms-swift) | 🟢 Open source | ms-swift is ModelScope's framework for training, aligning, evaluating, quantizing, and deploying text and multimodal models, with full-parameter, PEFT, Megatron, preference-learning, and GRPO-family recipes. |
| [XTuner](https://github.com/InternLM/xtuner) | 🟢 Open source | XTuner is an open training engine for language and multimodal models that provides SFT, PEFT, preference optimization, reinforcement learning, sequence parallelism, and recipes for dense and mixture-of-experts architectures. |
| [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) | 🟢 Open source | AutoTrain Advanced is Hugging Face's low-code training toolkit for running text and multimodal fine-tuning jobs locally or on connected compute, with dataset preprocessing and Hub integration. |
| [LLM Foundry](https://github.com/mosaicml/llm-foundry) | 🟢 Open source | LLM Foundry is MosaicML's codebase for training, fine-tuning, evaluating, and exporting language models using Composer, with configuration-driven recipes and distributed strategies such as FSDP. |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | 🟢 Open source | LitGPT is Lightning AI's readable implementation and training toolkit for many language-model architectures, with pretraining, full fine-tuning, LoRA/QLoRA, adapter, quantization, and checkpoint-conversion recipes. |
| [torchtune](https://github.com/pytorch/torchtune) | 🟢 Open source | torchtune is a PyTorch-native post-training library organized around hackable recipes for SFT, LoRA/QLoRA, knowledge distillation, preference optimization, quantization-aware training, and distributed execution. |
| [TRLoom](https://github.com/saqlain2204/trloom) | 🟢 Open source | TRLoom is a YAML-driven fine-tuning library on Hugging Face TRL that configures model, dataset, trainer, Weights & Biases, and optional Modal GPU execution in one file. |

## SFT and Core Training Libraries

**Supervised fine-tuning (SFT)** trains a pretrained model on input–target or conversational examples with a next-token objective. These libraries expose the model, optimizer, data, and training-loop primitives used directly or underneath higher-level frameworks.

| Library | Availability | Description |
| --- | --- | --- |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | 🟢 Open source | Hugging Face Transformers provides model implementations, tokenizers, data collators, generation utilities, and the Trainer API used by many SFT and post-training stacks across PyTorch, TensorFlow, and JAX. |
| [MosaicML Composer](https://github.com/mosaicml/composer) | 🟢 Open source | MosaicML Composer is a PyTorch training library with a configurable Trainer, distributed execution, checkpointing, callbacks, and algorithm hooks; it is the training engine beneath LLM Foundry. |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | 🟢 Open source | MLX LM is Apple's toolkit for generating from and fine-tuning language models with MLX on Apple silicon, including full, LoRA, and QLoRA-style training plus model conversion. |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | 🟢 Open source | MaxText is a JAX reference implementation for scalable transformer training on Google Cloud accelerators, with pretraining and fine-tuning configurations, distributed sharding, checkpointing, and performance recipes. |
| [EasyLM](https://github.com/young-geng/EasyLM) | 🟢 Open source | EasyLM is a JAX/Flax language-model training codebase with scripts for pretraining, supervised fine-tuning, evaluation, and checkpoint conversion across supported transformer architectures. |
| [OLMo-core](https://github.com/allenai/OLMo-core) | 🟢 Open source | OLMo-core is AI2's PyTorch library for building and training language models, exposing distributed training, data mixing, checkpointing, optimizers, and reusable components from the OLMo program. |
| [Nanotron](https://github.com/huggingface/nanotron) | 🟢 Open source | Nanotron is Hugging Face's minimal distributed transformer-training library for pretraining and fine-tuning, with tensor, pipeline, data, context, and expert parallelism. |
| [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) | 🟢 Open source | PyTorch Lightning is a structured training-loop framework that supplies distributed strategies, mixed precision, callbacks, logging, checkpointing, and accelerator portability for custom language-model fine-tuning code. |
| [KerasHub](https://github.com/keras-team/keras-hub) | 🟢 Open source | KerasHub is the Keras library of pretrained modeling components and task APIs, including causal-language-model presets, LoRA enablement, tokenization, generation, and fine-tuning across supported backends. |

## PEFT, LoRA, and Adapter Tooling

**Parameter-efficient fine-tuning (PEFT)** updates a small fraction of model parameters while keeping most base weights frozen. LoRA is one PEFT method; QLoRA combines LoRA training with a quantized frozen base model. Adapter code and adapter weights have separate licenses.

| Tool | Availability | Description |
| --- | --- | --- |
| [Hugging Face PEFT](https://github.com/huggingface/peft) | 🟢 Open source | Hugging Face PEFT is a library for training, composing, merging, saving, and loading parameter-efficient adapters, including LoRA, DoRA, AdaLoRA, IA3, prompt tuning, prefix tuning, and related methods. |
| [loralib](https://github.com/microsoft/LoRA) | 🟢 Open source | loralib is Microsoft's reference PyTorch implementation accompanying the LoRA paper, providing low-rank replacements for common layers and utilities for marking only LoRA parameters trainable. |
| [Adapters](https://github.com/adapter-hub/adapters) | 🟢 Open source | Adapters is a unified library for parameter-efficient and modular transfer learning, integrating LoRA, prefix tuning, bottleneck adapters, composition, merging, and quantized adapter training with transformer models. |
| [AdapterHub](https://adapterhub.ml) | 🟢 Open source | AdapterHub is an open ecosystem for discovering, sharing, and composing pretrained adapter modules; each uploaded adapter remains governed by its own metadata, base-model license, and usage terms. |
| [OpenDelta](https://github.com/thunlp/OpenDelta) | 🟢 Open source | OpenDelta is a plug-and-play delta-tuning library that injects trainable LoRA, prefix, adapter, and other lightweight modules into supported pretrained PyTorch models. |
| [QLoRA](https://github.com/artidoro/qlora) | 🟢 Open source | QLoRA is the reference implementation for fine-tuning LoRA adapters over a frozen 4-bit quantized base model using NF4, double quantization, and paged optimizers. |
| [mergekit](https://github.com/arcee-ai/mergekit) | 🟢 Open source | mergekit is an Apache-licensed toolkit for combining language-model checkpoints and LoRA adapters with multiple merge methods, out-of-core processing, tensor selection, and configuration-driven workflows. |
| [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) | 🟢 Open source | LLM-Adapters is a research toolkit and benchmark for applying and comparing parameter-efficient adapters across language-model tasks, including LoRA-family and bottleneck-style methods. |
| [LoRA-GA](https://github.com/Outsider565/LoRA-GA) | 🟢 Open source | LoRA-GA is the reference implementation of gradient-approximation initialization for LoRA adapters, intended to initialize low-rank updates using estimated full fine-tuning gradients. |

## Distributed Training and Training Optimization

Distributed tools divide parameters, optimizer states, activations, batches, or model layers across devices. Kernel libraries are included when they directly reduce the memory or compute cost of fine-tuning; serving-only systems appear later.

| Tool | Availability | Description |
| --- | --- | --- |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 🟢 Open source | DeepSpeed is Microsoft's distributed deep-learning library providing ZeRO optimizer-state and parameter sharding, pipeline and tensor parallelism, mixed precision, offload, checkpointing, and training optimizations used by many LLM frameworks. |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 🟢 Open source | Megatron-LM is NVIDIA's framework for large-scale transformer training with tensor, pipeline, sequence, context, data, and expert parallelism plus distributed checkpointing. |
| [ColossalAI](https://github.com/hpcaitech/ColossalAI) | 🟢 Open source | ColossalAI is a distributed training system with sharded optimizers, heterogeneous memory management, tensor and pipeline parallelism, mixture-of-experts support, and LLM fine-tuning examples. |
| [Hugging Face Accelerate](https://github.com/huggingface/accelerate) | 🟢 Open source | Hugging Face Accelerate provides a unified interface for running PyTorch training code across CPUs, GPUs, TPUs, mixed precision, DeepSpeed, FSDP, and multi-node configurations. |
| [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) | 🟢 Open source | PyTorch FSDP is PyTorch's native Fully Sharded Data Parallel API for sharding model parameters, gradients, and optimizer states across data-parallel workers during training. |
| [TorchTitan](https://github.com/pytorch/torchtitan) | 🟢 Open source | TorchTitan is a PyTorch-native reference platform for large-model training that composes FSDP, tensor, pipeline, context, and expert parallelism with distributed checkpointing and profiling. |
| [Ray Train](https://github.com/ray-project/ray) | 🟢 Open source | Ray Train is Ray's distributed training library for scaling PyTorch and Hugging Face workloads across workers and clusters with fault tolerance, storage integration, and orchestration APIs. |
| [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | 🟢 Open source | NVIDIA Transformer Engine provides optimized transformer layers, attention, and mixed-precision formats such as FP8 for training on supported NVIDIA, AMD, and Intel accelerators. |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | 🟢 Open source | FlashAttention is an exact, IO-aware attention implementation with CUDA kernels commonly integrated into fine-tuning frameworks to reduce attention memory traffic and support longer sequences. |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | 🟢 Open source | Liger Kernel is a collection of Triton kernels for transformer training operations, including fused losses, normalization, rotary embeddings, and memory-efficient linear cross entropy. |
| [xFormers](https://github.com/facebookresearch/xformers) | 🟢 Open source | xFormers is Meta's library of composable and optimized transformer building blocks, including memory-efficient attention operators used by training and fine-tuning systems. |

## RLHF, Preference Optimization, and Reinforcement Learning

Post-training can learn from ranked responses, scalar rewards, verifiable outcomes, AI feedback, or interactive environments. This section covers both offline preference methods such as DPO and online policy optimization such as PPO and GRPO.

| Tool | Availability | Description |
| --- | --- | --- |
| [Hugging Face TRL](https://github.com/huggingface/trl) | 🟢 Open source | Hugging Face TRL is a post-training library built on Transformers with trainers for SFT, reward modeling, DPO, KTO, GRPO, and related algorithms, plus PEFT and distributed-backend integrations. |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 🟢 Open source | OpenRLHF is a Ray- and vLLM-based post-training framework for SFT, reward models, DPO-family methods, PPO, GRPO, REINFORCE variants, and asynchronous or colocated rollout architectures. |
| [veRL](https://github.com/volcengine/verl) | 🟢 Open source | veRL is an open reinforcement-learning framework using HybridFlow to compose training and rollout backends, with PPO, GRPO, DAPO, agentic rollouts, FSDP, Megatron, vLLM, and SGLang integrations. |
| [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) | 🟢 Open source | NVIDIA NeMo RL is a scalable post-training library for language and vision-language models, covering SFT, DPO, GRPO-family reinforcement learning, distributed execution, rollout backends, and LoRA. |
| [Open-Instruct](https://github.com/allenai/open-instruct) | 🟢 Open source | Open-Instruct is AI2's codebase of reproducible recipes for instruction tuning, preference optimization, reward modeling, and reinforcement learning used in the Tülu model program. |
| [Alignment Handbook](https://github.com/huggingface/alignment-handbook) | 🟢 Open source | Alignment Handbook is Hugging Face's collection of recipes and educational material for continued pretraining, SFT, reward modeling, DPO, and model evaluation with the Transformers stack. |
| [AReaL](https://github.com/inclusionAI/AReaL) | 🟢 Open source | AReaL is a large-scale asynchronous reinforcement-learning system with separate training, inference, agent, and weight-update services for reasoning and agentic model post-training. |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | 🟢 Open source | SkyRL is a modular full-stack reinforcement-learning library supporting training, sampling, environments, long-horizon agents, and colocated or disaggregated execution across multiple model backends. |
| [slime](https://github.com/THUDM/slime) | 🟢 Open source | slime is an LLM reinforcement-learning scaling framework that connects Megatron training with SGLang rollouts and supports synchronous, asynchronous, dense, and mixture-of-experts post-training workflows. |
| [RAGEN](https://github.com/mll-lab-nu/RAGEN) | 🟢 Open source | RAGEN is a framework for reinforcement-learning training and diagnosis of multi-turn reasoning agents in interactive environments using trajectory- and turn-level reward structures. |
| [DeepSpeed-Chat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) | 🟢 Open source | DeepSpeed-Chat is a reference RLHF pipeline in DeepSpeedExamples that covers SFT, reward-model training, and PPO-based actor–critic fine-tuning with ZeRO-powered distributed execution. |
| [ROLL](https://github.com/alibaba/ROLL) | 🟢 Open source | ROLL is Alibaba's reinforcement-learning library for large models, providing modular training and rollout components for reasoning, agentic, multimodal, and preference-optimization workloads. |

## Data Curation, Labeling, and Human Feedback

Fine-tuning quality depends on deduplication, filtering, formatting, provenance, and labels. Human-feedback tools collect ratings, rankings, corrections, and demonstrations; data-curation libraries prepare large corpora before training.

| Tool | Availability | Description |
| --- | --- | --- |
| [Argilla](https://github.com/argilla-io/argilla) | 🟢 Open source | Argilla is a collaborative data platform for curating, annotating, and reviewing language-model datasets, including response rating, ranking, classification, and human-feedback workflows. |
| [Label Studio](https://github.com/HumanSignal/label-studio) | 🔵 Open core | Label Studio is an open-source data-labeling interface with commercial enterprise features, supporting text, chat, ranking, audio, image, and multimodal annotation projects. |
| [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | 🟢 Open source | NeMo Curator is NVIDIA's scalable toolkit for downloading, extracting, deduplicating, filtering, classifying, and generating text, image, video, and synthetic datasets for model training. |
| [Data-Juicer](https://github.com/modelscope/data-juicer) | 🟢 Open source | Data-Juicer is ModelScope's system for processing, filtering, deduplicating, mixing, and analyzing text and multimodal datasets with configurable operators and distributed execution. |
| [DataTrove](https://github.com/huggingface/datatrove) | 🟢 Open source | DataTrove is Hugging Face's library for large-scale text data processing with readers, writers, filters, deduplication, tokenization, statistics, and local or cluster execution. |
| [Dolma Toolkit](https://github.com/allenai/dolma) | 🟢 Open source | Dolma Toolkit is AI2's toolkit for curating language-model pretraining corpora through document processing, filtering, deduplication, mixing, and tokenization pipelines. |
| [Cleanlab](https://github.com/cleanlab/cleanlab) | 🟢 Open source | Cleanlab is a data-centric machine-learning library for detecting label errors, outliers, near duplicates, and low-quality examples that can contaminate fine-tuning datasets. |
| [Prodigy](https://prodi.gy) | 🔒 Commercial | Prodigy is a scriptable annotation tool from the spaCy team with model-in-the-loop recipes for text classification, named entities, preference collection, and custom interfaces. |
| [Labelbox](https://labelbox.com) | 🔒 Commercial | Labelbox is a commercial data platform for annotation, curation, and managed human evaluation across text and multimodal data, including preference and response-quality projects. |
| [Scale Data Engine](https://scale.com/data-engine) | 🔒 Commercial | Scale Data Engine is a managed data-labeling and curation service for collecting expert demonstrations, preferences, evaluations, and multimodal training data. |
| [Snorkel Flow](https://snorkel.ai/snorkel-flow/) | 🔒 Commercial | Snorkel Flow is a commercial data-development platform for programmatic labeling, data curation, model iteration, and subject-matter-expert workflows, derived from the Snorkel research project. |

## Synthetic Data and Data Distillation

Synthetic-data tools use models, rules, documents, or seed examples to create instruction–response pairs, preference data, reasoning traces, and augmented datasets. Generated records still require provenance checks, deduplication, safety review, and held-out evaluation.

| Tool | Availability | Description |
| --- | --- | --- |
| [distilabel](https://github.com/argilla-io/distilabel) | 🟢 Open source | distilabel is an Argilla framework for building reproducible synthetic-data and AI-feedback pipelines, with task abstractions, model-provider integrations, batching, caching, and dataset export. |
| [DataDreamer](https://github.com/datadreamer-dev/DataDreamer) | 🟢 Open source | DataDreamer is a Python library for reproducible synthetic-data generation and training workflows with cached steps, prompt and embedding providers, dataset operations, and Hugging Face integrations. |
| [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit) | 🟢 Open source | Synthetic Data Kit is Meta's command-line and library workflow for ingesting documents, generating and curating question–answer or reasoning examples, and preparing fine-tuning datasets. |
| [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) | 🟢 Open source | Augmentoolkit is a configurable pipeline for converting source documents into synthetic instruction, conversation, and domain datasets using local or API-hosted language models. |
| [Bonito](https://github.com/BatsResearch/bonito) | 🟢 Open source | Bonito is an open task-generation model and codebase for converting unannotated text into task-specific instruction-tuning examples through conditional task generation. |
| [Self-Instruct](https://github.com/yizhongw/self-instruct) | 🟢 Open source | Self-Instruct is the reference code and data-generation pipeline for bootstrapping instruction-following datasets by generating, filtering, and instantiating new tasks from seed instructions. |
| [LLM2LLM](https://github.com/SqueezeAILab/LLM2LLM) | 🟢 Open source | LLM2LLM is a reference implementation for iterative data augmentation that uses a stronger model to generate corrective examples from a smaller model's errors. |
| [Prompt2Model](https://github.com/neulab/prompt2model) | 🟢 Open source | Prompt2Model is a research system that turns a natural-language task description into a small specialized model by retrieving data, generating synthetic examples, training, and evaluating. |
| [Gretel](https://gretel.ai) | 🔒 Commercial | Gretel is a commercial synthetic-data platform for generating and transforming text, tabular, and structured datasets with privacy and quality controls. |
| [Mostly AI](https://github.com/mostly-ai/mostlyai) | 🔵 Open core | Mostly AI provides an open-source synthetic-data SDK and a commercial platform for generating privacy-preserving tabular and sequential data that can supplement model-training workflows. |

## Experiment Tracking and Model Registries

Experiment trackers record hyperparameters, metrics, logs, datasets, artifacts, and checkpoints so training runs can be compared and reproduced. Registry features manage the promotion and lineage of selected model versions.

| Tool | Availability | Description |
| --- | --- | --- |
| [Weights & Biases](https://wandb.ai) | 🔵 Open core | Weights & Biases provides experiment tracking, dashboards, artifact and model registries, reports, and sweep orchestration through open-source clients and a hosted or self-managed platform. |
| [MLflow](https://github.com/mlflow/mlflow) | 🟢 Open source | MLflow is an open-source platform for experiment tracking, model packaging, evaluation, registry workflows, and deployment metadata across machine-learning and generative-AI projects. |
| [Comet](https://www.comet.com) | 🔵 Open core | Comet provides experiment tracking, model registry, artifact management, panels, and production monitoring through open SDKs and a commercial platform. |
| [ClearML](https://github.com/clearml/clearml) | 🔵 Open core | ClearML is an open-source experiment manager and orchestration suite with dataset versioning, agents, pipelines, artifact storage, model registry, and commercial enterprise services. |
| [Aim](https://github.com/aimhubio/aim) | 🟢 Open source | Aim is a self-hostable experiment tracker for logging, querying, comparing, and visualizing metrics, parameters, distributions, text, and training artifacts. |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | 🟢 Open source | TensorBoard is an open-source visualization toolkit for training metrics, computational graphs, embeddings, profiling, and experiment comparison through event logs. |
| [DVCLive](https://github.com/iterative/dvclive) | 🟢 Open source | DVCLive is Iterative's experiment-logging library for metrics, parameters, plots, and artifacts, designed to connect training code with DVC experiments and Studio. |
| [Trackio](https://github.com/gradio-app/trackio) | 🟢 Open source | Trackio is a lightweight local experiment tracker with a Python logging API and dashboard, designed to integrate with Hugging Face training workflows and Spaces. |

## Hyperparameter Optimization and Orchestration

Hyperparameter-optimization tools schedule repeated training trials, choose parameter values, prune poor runs, and allocate compute. General configuration and workflow systems are included when they directly support reproducible fine-tuning sweeps.

| Tool | Availability | Description |
| --- | --- | --- |
| [Ray Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune) | 🟢 Open source | Ray Tune is a distributed hyperparameter-tuning library with search algorithms, schedulers, early stopping, checkpoint handling, and integrations for PyTorch and Hugging Face trainers. |
| [Optuna](https://github.com/optuna/optuna) | 🟢 Open source | Optuna is a framework-agnostic hyperparameter-optimization library with define-by-run search spaces, samplers, pruning, distributed storage, visualization, and experiment analysis. |
| [Hydra](https://github.com/facebookresearch/hydra) | 🟢 Open source | Hydra is Meta's hierarchical configuration framework for composing training settings and launching local or distributed multirun sweeps through configurable launchers and sweepers. |
| [Ax](https://github.com/facebook/Ax) | 🟢 Open source | Ax is Meta's adaptive experimentation platform for Bayesian optimization and multi-objective search, with service and developer APIs for iterative training experiments. |
| [Katib](https://github.com/kubeflow/katib) | 🟢 Open source | Katib is Kubeflow's Kubernetes-native system for hyperparameter tuning, early stopping, and neural-architecture search using jobs and pluggable suggestion algorithms. |
| [NNI](https://github.com/microsoft/nni) | 🟢 Open source | NNI is Microsoft's AutoML toolkit for running hyperparameter search, pruning, neural-architecture search, and model-compression experiments on local or remote training services. |
| [Nevergrad](https://github.com/facebookresearch/nevergrad) | 🟢 Open source | Nevergrad is Meta's derivative-free optimization library with evolutionary, bandit, and population-based optimizers suitable for noisy or non-differentiable training objectives. |
| [Determined AI](https://github.com/determined-ai/determined) | 🔵 Open core | Determined AI is an open-source distributed training and hyperparameter-search platform with scheduling, fault tolerance, checkpointing, experiment tracking, and HPE enterprise offerings. |

## Quantization and Compression

Quantization reduces weight, activation, or cache precision for lower memory use and different deployment targets. Some methods support quantization-aware fine-tuning; others compress a completed checkpoint. Runtime support varies by architecture, accelerator, and serving engine.

| Tool | Availability | Description |
| --- | --- | --- |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 🟢 Open source | bitsandbytes provides low-bit linear layers, 8-bit optimizers, and 4-bit NF4/FP4 loading used by QLoRA and other memory-efficient fine-tuning workflows across supported hardware. |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | 🟢 Open source | GPTQModel is the actively maintained successor to AutoGPTQ for GPTQ and related model quantization, with Transformers, vLLM, and SGLang integrations across multiple accelerator and CPU backends. |
| [LLM Compressor](https://github.com/vllm-project/llm-compressor) | 🟢 Open source | LLM Compressor is the vLLM project's library for applying GPTQ, AWQ, SmoothQuant, AutoRound, FP8, FP4, and other compression schemes and saving compatible compressed-tensors checkpoints. |
| [AWQ](https://github.com/mit-han-lab/llm-awq) | 🟢 Open source | AWQ is the official implementation of Activation-aware Weight Quantization, a post-training weight-only method that protects salient channels using activation statistics and calibrated scaling. |
| [HQQ](https://github.com/mobiusml/hqq) | 🟢 Open source | HQQ is a half-quadratic quantization library for on-the-fly low-bit model quantization and dequantization, with backend integrations for Transformers and several inference engines. |
| [torchao](https://github.com/pytorch/ao) | 🟢 Open source | torchao is PyTorch's native quantization and sparsity library for training and inference, including low-bit dtypes, quantization-aware training, post-training quantization, and model transformations. |
| [Quanto](https://github.com/huggingface/quanto) | 🟢 Open source | Quanto is Hugging Face's PyTorch quantization toolkit for weights and activations, supporting calibration, quantization-aware training, freezing, serialization, and device-independent model loading. |
| [Intel Neural Compressor](https://github.com/intel/neural-compressor) | 🟢 Open source | Intel Neural Compressor is a cross-framework optimization toolkit for post-training quantization, quantization-aware training, pruning, distillation, and model export on supported Intel hardware. |
| [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | 🟢 Open source | TensorRT Model Optimizer is NVIDIA's library for quantization, sparsity, distillation, speculative decoding, and checkpoint optimization before deployment through TensorRT-LLM or related runtimes. |
| [AQLM](https://github.com/Vahe1994/AQLM) | 🟢 Open source | AQLM is the reference implementation of additive quantization for compressing language-model weights with learned codebooks and fine-tuning the resulting quantized representation. |
| [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp) | 🟢 Open source | QuIP# is a research implementation of lattice-codebook and incoherence-processing methods for very-low-bit post-training quantization of language models. |
| [SmoothQuant](https://github.com/mit-han-lab/smoothquant) | 🟢 Open source | SmoothQuant is the official implementation of a training-free method that migrates activation outliers into weights to enable W8A8 quantization of transformer models. |
| [AutoRound](https://github.com/intel/auto-round) | 🟢 Open source | AutoRound is Intel's weight-rounding quantization toolkit for language and multimodal models, with low-bit calibration, multiple export formats, and serving-framework integrations. |

## Checkpoint Export, Conversion, and Inference

Fine-tuning is not complete until a checkpoint can be merged, converted, validated, and served. These tools export framework checkpoints, load adapters, or provide inference runtimes that accept fine-tuned models.

| Tool | Availability | Description |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 🟢 Open source | llama.cpp provides model-conversion scripts, GGUF quantization tools, adapter loading, and CPU/GPU inference for supported architectures, making it a common deployment target for merged fine-tuned checkpoints. |
| [vLLM](https://github.com/vllm-project/vllm) | 🟢 Open source | vLLM is a high-throughput model-serving engine with OpenAI-compatible APIs, distributed inference, quantized checkpoint support, and runtime loading of supported LoRA adapters. |
| [SGLang](https://github.com/sgl-project/sglang) | 🟢 Open source | SGLang is a serving and programming framework for language and multimodal models with distributed inference, structured generation, quantization, and LoRA adapter support. |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 🟢 Open source | TensorRT-LLM is NVIDIA's inference library and compiler stack for optimizing, converting, quantizing, and serving supported language-model checkpoints on NVIDIA GPUs. |
| [Hugging Face Optimum](https://github.com/huggingface/optimum) | 🟢 Open source | Hugging Face Optimum provides export and hardware-optimization interfaces connecting Transformers models to ONNX Runtime, OpenVINO, and other accelerator-specific backends. |
| [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) | 🟢 Open source | ONNX Runtime GenAI supplies generation APIs, model builders, quantization utilities, and runtime support for deploying converted generative-model checkpoints through ONNX Runtime. |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | 🟢 Open source | MLC LLM compiles and packages language models for deployment across GPUs, CPUs, browsers, and mobile devices, with conversion, quantization, and OpenAI-compatible serving tools. |
| [ExecuTorch](https://github.com/pytorch/executorch) | 🟢 Open source | ExecuTorch is PyTorch's on-device inference stack for exporting, optimizing, and running supported models on mobile and embedded hardware through backend delegates. |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | 🟢 Open source | LMDeploy is an inference and compression toolkit for language and vision-language models, with TurboMind and PyTorch engines, quantization, serving APIs, and adapter support. |
| [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) | 🟢 Open source | OpenVINO GenAI provides generation pipelines, model conversion, compression integrations, and serving examples for running supported language and multimodal models on Intel hardware. |
| [LoRAX](https://github.com/predibase/lorax) | 🟢 Open source | LoRAX is an Apache-licensed multi-adapter inference server that dynamically loads LoRA adapters and exposes OpenAI-compatible APIs, Kubernetes assets, metrics, and tenant isolation. |

## Managed Cloud Fine-Tuning Services

Managed services provision training compute, accept datasets or custom code, and usually connect the resulting model to hosted inference. Model support, weight export, regions, privacy terms, preview status, and pricing differ; verify current documentation before committing a workload.

| Service | Availability | Description |
| --- | --- | --- |
| [Together AI Fine-Tuning](https://docs.together.ai/docs/fine-tuning/quickstart) | 🔒 Commercial | Together AI Fine-Tuning runs LoRA or full-parameter jobs on supported open-weight base models and connects completed models to dedicated hosted endpoints, with validation and checkpoint options documented per model. |
| [Fireworks AI Fine-Tuning](https://docs.fireworks.ai/fine-tuning/fine-tuning-models) | 🔒 Commercial | Fireworks AI Fine-Tuning provides managed SFT, DPO, and reinforcement fine-tuning for supported models through its UI, CLI, and API, with model-weight download and dedicated deployment options. |
| [Predibase](https://docs.predibase.com/fine-tuning/overview) | 🔒 Commercial | Predibase is a managed platform for fine-tuning and serving open-weight models with LoRA and QLoRA, built by the team behind the separate open-source Ludwig and LoRAX projects. |
| [OpenPipe](https://docs.openpipe.ai/features/fine-tuning) | 🔒 Commercial | OpenPipe is a managed platform that collects and curates application traces into training datasets, runs supported fine-tuning jobs, evaluates resulting models, and serves them through an API. |
| [Tinker](https://thinkingmachines.ai/tinker/) | 🔒 Commercial | Tinker is Thinking Machines Lab's managed training API for LoRA-based SFT and reinforcement learning on supported open-weight models, exposing gradient, optimizer, sampling, and checkpoint primitives while operating distributed infrastructure. |
| [OpenAI Fine-Tuning](https://platform.openai.com/docs/guides/fine-tuning) | 🔒 Commercial | OpenAI Fine-Tuning is a managed API and dashboard workflow for SFT, DPO, and reinforcement fine-tuning on selected OpenAI models, with uploaded datasets, job events, checkpoints, and hosted inference. |
| [Vertex AI Model Tuning](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models) | 🔒 Commercial | Vertex AI Model Tuning provides managed supervised and preference tuning for selected Google and open models, with training data in Google Cloud and deployment through Vertex AI endpoints. |
| [Amazon Bedrock Model Customization](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html) | 🔒 Commercial | Amazon Bedrock Model Customization provides managed fine-tuning, continued pretraining, distillation, model import, and reinforcement fine-tuning capabilities for supported foundation models and regions. |
| [Microsoft Foundry Fine-Tuning](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning) | 🔒 Commercial | Microsoft Foundry Fine-Tuning provides managed SFT, DPO, and reinforcement fine-tuning for supported OpenAI and selected open models, with region- and deployment-specific availability. |
| [Amazon SageMaker AI Fine-Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-fine-tuning.html) | 🔒 Commercial | Amazon SageMaker AI provides configurable foundation-model fine-tuning through JumpStart, training jobs, distributed libraries, managed storage, experiment tracking, and endpoint deployment. |
| [Databricks AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/) | 🔒 Commercial | Databricks AI Runtime is a serverless GPU environment for custom training and fine-tuning with frameworks such as PyTorch, Transformers, Unsloth, and LLM Foundry; some distributed capabilities remain preview or beta. |
| [IBM watsonx.ai Fine-Tuning](https://www.ibm.com/docs/en/watsonx/saas?topic=models-methods-tuning) | 🔒 Commercial | IBM watsonx.ai provides LoRA and QLoRA fine-tuning for supported foundation models and regions, alongside tuning experiments, deployment, and governance features; supported methods vary by product edition. |
| [Cohere Fine-Tuning](https://docs.cohere.com/reference/createfinetunedmodel) | 🔒 Commercial | Cohere Fine-Tuning trains and deploys supported Cohere models from uploaded datasets through its API and dashboard; model eligibility and retirement dates are documented in Cohere's current model lifecycle pages. |
| [Baseten Training](https://docs.baseten.co/training/overview) | 🔒 Commercial | Baseten Training runs custom training containers on managed GPUs and deploys produced checkpoints; its separate Loops early-access service exposes Tinker-compatible LoRA SFT and reinforcement-learning primitives. |
| [Lamini](https://docs.lamini.ai/) | 🔒 Commercial | Lamini provides managed and self-hosted enterprise APIs for tuning supported open-weight models, including conventional fine-tuning and its adapter-based Memory Tuning workflow. |

## Evaluation and Checkpoint Selection

Fine-tuned models should be compared on a held-out dataset before promotion. The tools below run repeatable regression tests or benchmark checkpoints; they do not replace task-specific human review, contamination checks, or production monitoring.

| Tool | Availability | Description |
| --- | --- | --- |
| [DeepEval](https://github.com/confident-ai/deepeval) | 🟢 Open source | DeepEval is an open-source evaluation framework for regression-testing fine-tuned checkpoints with versioned test cases, custom or built-in metrics, benchmark runners, and CI integration. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 🟢 Open source | lm-evaluation-harness is EleutherAI's benchmark runner for comparing language-model checkpoints across a large registry of academic tasks, prompting configurations, and inference backends. |
| [LightEval](https://github.com/huggingface/lighteval) | 🟢 Open source | LightEval is Hugging Face's toolkit for evaluating language models across local, distributed, and hosted backends with configurable tasks, metrics, prompt formats, and result tracking. |
| [OpenCompass](https://github.com/open-compass/opencompass) | 🟢 Open source | OpenCompass is an evaluation platform for language and multimodal checkpoints with dataset registries, distributed inference, judge-based and objective metrics, and report generation. |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | 🟢 Open source | Inspect AI is the UK AI Security Institute's evaluation framework for defining datasets, solvers, scorers, model backends, sandboxed tool use, and reproducible evaluation logs. |
| [EvalScope](https://github.com/modelscope/evalscope) | 🟢 Open source | EvalScope is ModelScope's framework for evaluating and profiling language, multimodal, embedding, and reranker models across benchmark and inference backends. |
| [RewardBench](https://github.com/allenai/reward-bench) | 🟢 Open source | RewardBench is AI2's benchmark and codebase for comparing reward-model checkpoints on preference pairs, helping select reward models before RLHF or rejection-sampling pipelines. |
| [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) | 🟢 Open source | AlpacaEval is an automatic instruction-following evaluator for pairwise comparison of model outputs, with length-controlled variants and reproducible leaderboard tooling. |
| [OLMES](https://github.com/allenai/olmes) | 🟢 Open source | OLMES is AI2's standardized evaluation framework and reporting protocol for reproducibly comparing base and instruction-tuned language-model checkpoints. |

## Domain-Specific and Multimodal Fine-Tuning

Domain toolkits encode data formats, model recipes, and evaluation conventions for particular fields or modalities. A repository can be open-source while its downloadable base model or adapter weights are not; each checkpoint's model card and license must be checked separately.

| Tool | Availability | Description |
| --- | --- | --- |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 🟢 Open source | FinGPT is an open research framework for financial language models with data pipelines, instruction-tuning and LoRA scripts, sentiment and forecasting tasks, and published model recipes. |
| [NVIDIA BioNeMo Framework](https://github.com/NVIDIA/bionemo-framework) | 🟢 Open source | NVIDIA BioNeMo Framework provides domain-specific libraries and scalable recipes for training and LoRA-tuning biomolecular foundation models such as protein and DNA sequence models. |
| [AdaptLLM](https://github.com/microsoft/LMOps/tree/main/adaptllm) | 🟢 Open source | AdaptLLM is Microsoft's reference code for reading-comprehension-based domain adaptation, converting raw domain corpora into training tasks for continued pretraining and instruction tuning. |
| [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) | 🟢 Open source | LLaMA-Adapter is a research codebase for parameter-efficient instruction and multimodal adaptation using small trainable modules; released checkpoints remain subject to their underlying model licenses. |
| [LAVIS](https://github.com/salesforce/LAVIS) | 🟢 Open source | LAVIS is Salesforce's library for training and evaluating language–vision models, with standardized datasets, processors, tasks, model implementations, and parameter-efficient adaptation recipes. |
| [MedicalGPT](https://github.com/shibing624/MedicalGPT) | 🟢 Open source | MedicalGPT is a training toolkit with continued pretraining, SFT, reward modeling, DPO, and PPO scripts for medical-domain language models; generated checkpoints inherit their base-model terms. |
| [FinLoRA](https://github.com/Open-Finance-Lab/FinLoRA) | 🟠 Open weights | FinLoRA publishes financial LoRA adapter checkpoints and reproducible training scripts; its downloadable adapters remain subject to the repository's terms plus underlying base-model and dataset licenses. |

## Discontinued and Historical Tools

Influential projects that are archived, superseded, concluded, or no longer actively maintained. They remain here for provenance and to prevent stale recommendations. Availability describes the linked historical artifact; status reflects the last-reviewed date above.

| Tool | Availability | Former category | Status |
| --- | --- | --- | --- |
| [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | 🟢 Open source | Quantization | AutoGPTQ is archived and unmaintained; its maintainers direct new GPTQ work to [GPTQModel](https://github.com/ModelCloud/GPTQModel), and current Transformers releases no longer support AutoGPTQ. |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | 🟢 Open source | Quantization | AutoAWQ is archived and officially deprecated; its repository directs new AWQ quantization work to the vLLM project's [LLM Compressor](https://github.com/vllm-project/llm-compressor). |
| [TRLX](https://github.com/CarperAI/trlx) | 🟢 Open source | RLHF | TRLX is an early distributed RLHF framework for PPO and ILQL whose repository is not archived but has received no code push since January 2024. |
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 🟢 Open source | SFT and synthetic data | Stanford Alpaca is the static 2023 reference project for Self-Instruct-style data generation and LLaMA fine-tuning; its released model artifacts were governed by the original LLaMA license. |
| [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) | 🟢 Open source | PEFT | Alpaca-LoRA is an influential early LoRA fine-tuning example for LLaMA whose implementation is retained primarily as a historical reference for the 2023 ecosystem. |
| [RL4LMs](https://github.com/allenai/RL4LMs) | 🟢 Open source | RLHF | RL4LMs is AI2's modular reinforcement-learning library for language generation; its archived code remains useful for reproducing PPO and related research baselines. |
| [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) | 🟢 Open source | Data and SFT | OpenAssistant is LAION's concluded project for crowdsourcing assistant conversations, reward labels, and open training recipes; the repository and datasets remain available for research. |
| [OpenChatKit](https://github.com/togethercomputer/OpenChatKit) | 🟢 Open source | SFT and RLHF | OpenChatKit is Together's 2023 toolkit of chat-model training, moderation, retrieval, and evaluation code; development has slowed and current Together fine-tuning is offered separately as a managed service. |
| [adapter-transformers](https://github.com/adapter-hub/adapter-transformers) | 🟢 Open source | PEFT | adapter-transformers is the superseded predecessor to the standalone [Adapters](https://github.com/adapter-hub/adapters) library; AdapterHub directs new development and migrations to Adapters. |
| [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) | 🟢 Open source | Quantization | GPTQ-for-LLaMa is an early GPTQ implementation retained for historical reproducibility; current quantization stacks provide broader model, kernel, serialization, and hardware support. |

## Key Papers and Concepts

Foundational reading for understanding the methods implemented by the tools above. Cite these papers for the algorithms and the relevant software repository for an implementation.

- **LoRA** — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021). Freezes base weights and learns low-rank update matrices.
- **QLoRA** — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023). Trains LoRA adapters through a frozen 4-bit quantized model.
- **RLHF** — [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022). Established the SFT → reward model → PPO pipeline at modern LLM scale.
- **RLAIF and Constitutional AI** — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Bai et al., 2022). Uses written principles and model feedback to produce preference signals.
- **DPO** — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023). Optimizes preference pairs without fitting a separate reward model or running online RL.
- **GRPO** — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (Shao et al., 2024). Introduced Group Relative Policy Optimization for sampled groups of responses.
- **PPO** — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017). The clipped policy-gradient method underlying many early RLHF systems.
- **ZeRO** — [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2019). Shards optimizer states, gradients, and parameters across data-parallel workers.
- **Megatron-LM** — [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019). Describes tensor model parallelism for transformer training.
- **AWQ** — [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (Lin et al., 2023). Uses activation statistics to protect salient weights during low-bit quantization.

## Glossary

Short definitions of terms used throughout this list.

- **Fine-tuning** — continuing optimization of a pretrained model on a smaller dataset to change its behavior, task performance, style, or domain knowledge.
- **Post-training** — adaptation after base pretraining, including SFT, preference optimization, reward modeling, reinforcement learning, distillation, and safety tuning.
- **SFT (supervised fine-tuning)** — training on examples with target outputs using a token-level supervised objective.
- **Full fine-tuning** — updating all or most model parameters; compute- and memory-intensive but not constrained to a small adapter subspace.
- **PEFT (parameter-efficient fine-tuning)** — updating a small number of added or selected parameters while keeping most base weights frozen.
- **Adapter** — a small trainable parameter set attached to or composed with a frozen base model; LoRA is one adapter method.
- **LoRA** — low-rank adaptation, which represents weight updates as the product of small trainable matrices.
- **QLoRA** — LoRA training over a frozen low-bit quantized base model, commonly 4-bit NF4.
- **Reward model** — a model trained to assign a scalar preference or quality score to candidate outputs.
- **RLHF** — reinforcement learning from human feedback, using human demonstrations, preferences, or rewards to optimize a policy.
- **RLAIF** — reinforcement learning from AI feedback, where model-generated critiques, rankings, or rule-based judgments provide training signals.
- **DPO** — direct preference optimization, an offline objective that learns from chosen–rejected response pairs without a separate online RL loop.
- **GRPO** — group relative policy optimization, which computes relative advantages across multiple sampled responses to the same prompt.
- **RLVR** — reinforcement learning with verifiable rewards, using automatically checked outcomes such as tests, proofs, or exact answers.
- **Checkpoint** — a saved model, adapter, optimizer, scheduler, and training-state snapshot.
- **Quantization** — representing weights or activations at lower precision to change memory, storage, or compute requirements.
- **Distillation** — training a smaller or otherwise different model to reproduce selected behavior or distributions from a teacher.
- **Open source** — software released under an OSI-approved license; access to source code or weights alone is insufficient.
- **Open weights** — downloadable model or adapter parameters whose license is not an OSI software license and may restrict use or redistribution.
- **Open core** — an open-source component paired with a commercial hosted or enterprise product.

## Frequently Asked Questions

**What are LLM fine-tuning tools?**  
LLM fine-tuning tools are frameworks, libraries, platforms, and managed services used to adapt pretrained language models with task-specific, domain-specific, preference, or outcome data. They span dataset preparation, SFT, PEFT, RLHF, experiment tracking, evaluation, compression, and deployment.

**Which LLM fine-tuning framework should I use?**  
Choose from constraints rather than a universal ranking: supported model and modality, hardware, full versus adapter training, distributed scale, required post-training algorithm, checkpoint portability, license, and operational ownership. Run a small end-to-end trial before standardizing.

**What is the difference between fine-tuning, RAG, and prompt engineering?**  
Prompt engineering changes the instructions sent at inference time. RAG supplies retrieved context without changing model parameters. Fine-tuning changes weights or adapters. They solve different problems and are often combined; changing facts alone is usually easier to govern with retrieval than weight updates.

**What is the difference between full fine-tuning, LoRA, and QLoRA?**  
Full fine-tuning updates most model weights. LoRA freezes the base and trains low-rank update matrices. QLoRA also keeps the base frozen but loads it at low precision during adapter training, reducing memory use while retaining downloadable adapter outputs.

**How do SFT, DPO, RLHF, RLAIF, and GRPO differ?**  
SFT learns target responses directly. DPO learns from offline chosen–rejected pairs. RLHF uses human-derived reward signals in a reinforcement-learning loop; RLAIF substitutes or supplements AI-derived feedback. GRPO is an online policy-optimization method that compares groups of sampled responses.

**How much GPU memory does fine-tuning require?**  
There is no model-independent number. Memory depends on parameter count, precision, optimizer, sequence length, batch size, activation checkpointing, adapter rank, quantization, and sharding. Use the selected framework's estimator or run a short representative batch before budgeting a full job.

**How should a fine-tuned checkpoint be evaluated?**  
Keep a held-out dataset separated before training; compare against the unchanged base and current production model; measure task quality, safety, regressions, latency, and cost; inspect slices and failures; and repeat the same suite for every candidate checkpoint before promotion.

**Can proprietary API models be fine-tuned?**  
Only when the provider exposes a supported managed fine-tuning product. The provider chooses eligible base models, algorithms, data formats, regions, retention, checkpoint access, and serving terms; users generally cannot export proprietary model weights.

**Are open-weight models and adapters open source?**  
Not necessarily. Open source describes software under an OSI-approved license. Downloadable weights can use community, research-only, non-commercial, responsible-use, or custom licenses. Check the code license, base-model license, adapter license, and dataset terms independently.

**How should I cite this list?**  
See [Citing This List](#citing-this-list) for a citation template and BibTeX. When citing an individual tool or method, prefer that tool's repository, official documentation, or paper linked in its entry.

## Methodology

How this list is built and maintained. This section exists so readers and AI systems can judge its coverage and limitations.

- **Scope.** Software whose primary purpose includes adapting, post-training, preparing data for, tracking, compressing, exporting, serving, or regression-testing fine-tuned large language or multimodal models. General infrastructure is included only when it has direct fine-tuning use.
- **Inclusion criteria.** Open-source projects should show recent activity or lasting reference value. Commercial products must have public first-party documentation and a fine-tuning capability that is generally available or clearly labeled preview. Historical projects move to the dedicated section rather than disappearing.
- **Availability markers.** 🟢 means the linked software has an OSI-style open-source license; 🟠 means downloadable model or adapter weights under non-OSI or model-specific terms; 🔵 means an open-source component plus a commercial platform; 🔒 means a closed commercial service. Markers describe the primary linked artifact.
- **Weights are separate.** An open-source trainer does not make the base model, output checkpoint, dataset, or adapter open source. Descriptions call out inherited or separate terms where they materially affect use.
- **One entity per row.** Products and projects maintained by the same organization remain separate when they have distinct repositories, interfaces, or lifecycles. Related components are linked in prose rather than collapsed into one listing.
- **Ordering.** Entries are ordered by the maintainers' editorial judgment of adoption, completeness, and relevance within each category — not alphabetically and not by paid placement. There is no sponsored placement.
- **Editorial independence.** This directory is maintained by aglio-lab. Every entry follows the same sourcing, wording, and ordering rules, and no placement is sold.
- **Verification.** Every entry was checked against a primary source and its URL or repository status as of **2026-07-16**. Descriptions are factual and neutral; unsupported superlatives and cross-vendor performance claims are excluded.
- **Monthly review cadence.** The directory is reviewed during the first week of every month. Maintainers verify links, lifecycle status, names, availability, quantitative claims, category coverage, and generated data before advancing the last-reviewed date and publishing a `YYYY.MM` release. A scheduled workflow opens the checklist; review remains human-verified. See [`MAINTENANCE.md`](MAINTENANCE.md).
- **Status handling.** Archived, deprecated, superseded, or inactive projects are labeled and moved to [Discontinued and Historical Tools](#discontinued-and-historical-tools) when a current recommendation would be misleading.
- **Counting.** The headline count is one per table row, including historical entries. A tool appearing in prose or as a cross-link is not counted again.
- **Corrections.** Product availability, licenses, and maintenance status change. Open an issue or pull request through [Contributing](#contributing); documented corrections take priority over expanding the list.

## Related Lists and Resources

- [AI Red Teaming Tools](https://github.com/aglio-lab/ai-red-teaming-tools) — red-team frameworks, scanners, guardrails, and security benchmarks.
- [LLM Observability Tools](https://github.com/aglio-lab/llm-observability-tools) — tracing, monitoring, online evaluation, and production analytics.
- [AI Governance Tools](https://github.com/aglio-lab/ai-governance-tools) — governance, risk, compliance, audit, and responsible-AI tooling.
- [AI Agent Frameworks](https://github.com/aglio-lab/ai-agent-frameworks) — frameworks and orchestration tools for building and operating agents.
- [RAG and Retrieval Tools](https://github.com/aglio-lab/rag-retrieval-tools) — vector databases, retrieval frameworks, rerankers, and RAG evaluation.
- [Context Engineering Tools](https://github.com/aglio-lab/context-engineering-tools) — prompt, memory, context management, compression, and MCP tooling.

- [AI Evaluation Tools](https://github.com/aglio-lab/ai-evaluation-tools) — sibling directory covering LLM evaluation frameworks, platforms, benchmarks, red teaming, observability, and model-quality testing.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/) — implementation guides and conceptual documentation for parameter-efficient methods.
- [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook) — reproducible alignment recipes and educational material.
- [awesome-RLHF](https://github.com/opendilab/awesome-RLHF) — papers, frameworks, datasets, and resources for reinforcement learning from human feedback.
- [Awesome LLM Fine-tuning](https://github.com/Curated-Awesome-Lists/awesome-llm-fine-tuning) — community collection of fine-tuning tutorials, methods, and tools.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com) — courses and material on data, training, evaluation, deployment, and operations for deep-learning systems.

---

## Citing This List

If you reference this list in an article, paper, or AI-generated answer, please cite it as:

> *The Comprehensive List of LLM Fine-Tuning Tools* (2026). A curated list of open-source and commercial tools for fine-tuning and post-training large language models. GitHub. https://github.com/aglio-lab/llm-fine-tuning-tools

BibTeX:

```bibtex
@misc{llm-fine-tuning-tools,
  title        = {The Comprehensive List of LLM Fine-Tuning Tools},
  year         = {2026},
  howpublished = {\url{https://github.com/aglio-lab/llm-fine-tuning-tools}},
  note         = {A curated list of open-source and commercial tools for fine-tuning and post-training large language models. Accessed: 2026-07-16}
}
```

For reproducible citations of a changing list, cite a specific commit permalink or release tag. When citing individual tools, always prefer the tool's own repository, documentation, or paper linked in each entry.

## Contributing

Contributions are welcome. Open an [issue](https://github.com/aglio-lab/llm-fine-tuning-tools/issues/new) or [pull request](https://github.com/aglio-lab/llm-fine-tuning-tools/pulls) and follow these rules:

1. Add one tool per pull request to the most specific matching category.
2. Include the correct availability marker (🟢 / 🟠 / 🔵 / 🔒) and a neutral, factual description written as a complete sentence starting with the tool's name.
3. Link to the primary source: repository for open-source software, official documentation for a service, or paper for a research implementation.
4. Distinguish software licenses from model, adapter, and dataset licenses.
5. Report shutdowns, archival, deprecations, renames, and superseding projects with a primary-source status link where possible.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the maintainers have waived all copyright and related rights to this README under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Linked projects, documentation, datasets, model weights, and trademarks retain their own licenses and terms.
