# HybridSBD — Ablation Study Implementation Prompt

I will provide two inputs:

1. **The HybridSBD source-code repository** — the existing implementation that must be inspected and modified.
2. **The Ablation Study Specification (ABLATION_SPEC_REQUIRED_protocolA.md)** — the authoritative document describing the required experiments, configurations, training procedures, evaluation protocol, analyses, and deliverables. It defines 13 training runs. No other specification files exist; ignore any reference to other spec versions.

Your task is to implement this specification in the existing repository while preserving the intended model architecture, experimental methodology, and evaluation consistency.

## General instructions

- Read the complete specification carefully before taking any action.
- Treat the specification as the single source of truth for all technical and experimental requirements, together with the additional instructions at the end of this prompt.
- Do not reproduce, redefine, or independently modify requirements already specified in that document.
- Do not introduce additional experiments, model components, evaluation protocols, or methodological changes.
- Do not assume that a requirement is already implemented correctly simply because a corresponding function or configuration exists.
- Verify requirements against the actual source code and execution paths.
- Preserve existing functionality unless a change is explicitly required by the specification.
- Prioritize correctness, reproducibility, and traceability over implementation shortcuts.

## Stage 1 — Repository inspection and implementation planning

**Do not modify any source code during this stage.**

First, inspect the complete repository and map the specification requirements to the existing implementation.

Your inspection must cover:

1. **Model architecture:** Examine the model implementation and its forward pass. Identify the relevant modules, feature streams, dependencies, dimensions, prediction heads, and configuration mechanisms.
2. **Training pipeline:** Examine the training scripts, loss functions, data loaders, augmentations, optimization, validation, early stopping, checkpoint management, and experiment logging.
3. **Evaluation pipeline:** Examine all relevant evaluation scripts and shared functions. Trace the complete inference and evaluation process, including prediction aggregation, threshold selection, post-processing, and metric computation.
4. **Data handling:** Inspect dataset construction, training/validation/test splits, sampling procedures, caching, and potential data overlap or leakage.
5. **Reproducibility:** Examine random seed handling, DataLoader workers, sampling determinism, environment configuration, and experiment tracking.
6. **Benchmarking:** Examine the existing model benchmarking and resource-measurement functionality.
7. **Implementation compatibility:** Identify any differences between the current repository and the specification, including architectural, dimensional, procedural, and evaluation-related differences.

### Stage 1 report

Provide a structured report containing:

- Repository structure and relevant files.
- Mapping between specification requirements and existing implementation, including a table mapping each configuration flag (specification Section 4) to the exact code locations it controls and every place that consumes its output.
- Identified implementation gaps and inconsistencies.
- Potential dependencies and compatibility issues.
- Any risks that could affect experimental validity or reproducibility.
- Confirmation (or rejection, with reasons) of the defaults listed under "Additional instructions" below.
- A file-by-file implementation plan.
- A proposed validation and testing plan.

Clearly distinguish between:
- Requirements already satisfied.
- Requirements partially implemented.
- Requirements not implemented.
- Requirements that cannot be verified without additional information or execution.

If you identify ambiguities or conflicts not covered by this prompt, explain them and request clarification rather than making assumptions.

**Stop after completing Stage 1. Wait for my explicit approval before modifying any files.**

## Stage 2 — Implementation

Begin this stage only after I approve the Stage 1 report and implementation plan.

Implement all requirements in the specification and the additional instructions below.

During implementation:

- Follow the specification exactly.
- Use the existing architecture and codebase as the starting point.
- Implement configuration changes in a clean and maintainable manner.
- Ensure that configuration options control the actual computations in the forward pass.
- Ensure that disabled components are genuinely bypassed where required.
- Make dimensional changes consistent throughout the architecture.
- Integrate the required training and loss variations without unintentionally changing other training behavior.
- Preserve the specified data splits, preprocessing, augmentation, and optimization procedures.
- Implement the specified evaluation procedures using the existing evaluation pipeline wherever possible.
- Keep all experimental variants consistent with the required reference implementation.
- Save checkpoints, configurations, evaluation results, and other required experimental outputs.
- Ensure that the implementation is reproducible and that all relevant settings are recorded.

Do not silently change existing behavior. If an implementation decision requires deviating from the specification, stop and explain the issue before proceeding.

## Stage 3 — Testing and validation

Before considering the implementation complete, validate it systematically.

Use CPU-based tests and dummy inputs wherever possible. Do not start full training or require GPU execution merely to validate the implementation.

Test the following:

1. **Default configuration equals the current model (first acceptance test):** with all ablation flags at their default values, the model has the same parameter names, shapes, and exact parameter count as the current model (≈ 10.9M), and an existing checkpoint loads with no missing or unexpected keys.
2. Model construction for every one of the 13 configurations.
3. Forward-pass execution and output dimensions.
4. Feature dimensions (including the fusion input size) and exact parameter counts.
5. Correct activation or bypassing of model components.
6. Correct prediction-head configurations (two heads, or one head for A2).
7. Loss computation and backward propagation, including that for A1 the z-head receives no gradient.
8. Optimizer compatibility.
9. Checkpoint saving and loading.
10. Reproducibility-related changes.
11. Evaluation-script compatibility, including single-output variants.
12. Threshold-search and post-processing functionality.
13. Benchmark-script execution where hardware permits.
14. Generation and consistency of the required experimental records (CSV).

For every test, report:

- What was tested.
- How it was tested.
- Whether it passed or failed.
- Any limitations or untested conditions.

Do not claim that an experiment or test passed unless it was actually executed successfully. Clearly distinguish CPU validation from full GPU training and experimental reproduction.

## Stage 4 — Final deliverables and report

Provide all deliverables required by the specification and by the additional instructions below.

Additionally, provide a concise implementation report containing:

- Files added, modified, or removed.
- Summary of the implemented changes.
- Mapping between implemented functionality and specification requirements.
- Validation results.
- Any unresolved issues or deviations.
- Instructions for running the experiments.
- Any requirements that still need execution on the user's GPU or access to external datasets.

Include an `IMPLEMENTATION_NOTES.md` file documenting the implementation decisions and validation results.

## Additional instructions for this project

- **Explicitly required deliverables**, in addition to those in the specification:
  - One launcher (shell script or config files) with all 13 runs predefined, one command per run, named A0, A1, A2, A3, A4, C1, C2a, C2b, C3, C4, C5a, C5b, C6.
  - The CSV export script producing the format of specification Section 7.
- **Defaults for points the specification leaves open.** Use these and confirm them in the Stage 1 report, or explain why the code makes one impossible:
  1. Early stopping: keep the current validation procedure; apply each variant's own inference rule.
  2. A2 head bias: RetinaNet-style prior equal to the measured positive rate of the unified target u on the training data.
  3. C4: apply the existing spatial dropout (0.15) to whichever feature feeds FrameSimilarity.
  4. ClipShots cut vs. gradual: use the annotation's type labels if present; otherwise length ≤ 1 frame = cut.

## Important execution rules

1. Work incrementally and keep modifications reviewable.
2. Do not overwrite or remove original functionality unnecessarily.
3. Do not fabricate results, measurements, parameter counts, or test outcomes.
4. Do not perform full experimental training unless explicitly requested.
5. Do not implement anything beyond the specification and the additional instructions above.
6. If a technical issue prevents faithful implementation, report it instead of silently introducing an alternative.
7. Maintain consistency across all experimental variants and the reference model.
8. Keep the implementation, configuration, and reported results traceable to the same code version.

**Immediate task: Start with Stage 1 only. Inspect the specification and the source repository, then return the inspection report and proposed implementation plan. Do not modify any code until I explicitly approve proceeding to Stage 2.**
