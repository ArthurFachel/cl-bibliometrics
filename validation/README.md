# Validation set (F2)

`to_label.csv` holds 150 works drawn at random (seed 20261019) from the in-scope corpus, stratified by first-appearance year: 50 from 2015-2021, 50 from 2022-2024 and 50 from 2025-2026. Rule predictions are deliberately left out so the labeller is not anchored by them.

Label each row from the title and abstract only, write `1` when the label applies and leave it blank otherwise, then save the file as `labels.csv` and run `make validate`.

| column | 1 when the paper... |
|---|---|
| is_cl | is actually about continual / lifelong / incremental learning or catastrophic forgetting (0 = corpus false positive) |
| replay | stores, generates or replays past samples or features |
| regularization | constrains updates by importance weights (EWC, SI, MAS), distillation or other loss penalties |
| architecture | grows the network, isolates parameters, uses masks, experts or modules per task |
| optimization | projects or constrains gradients (orthogonal, null space, GEM) or targets the loss landscape |
| peft_llm | works with LLMs, VLMs or pre-trained foundation models, or uses LoRA, prompts or adapters |
| plasticity | studies loss of plasticity, dormant neurons, resets or trainability |
| theory | gives a formal or theoretical analysis (bounds, proofs, analytic models) |
| survey_benchmark | is a survey or review, or contributes a benchmark or dataset |
| online | uses online, task-free or streaming settings, or concept drift |
| federated | uses federated learning |
| rl | uses reinforcement learning or agents |
| application | applies CL to a named domain outside ML itself (medical, fault diagnosis, remote sensing, speech...) |

The labels are multi-label: a paper can be `replay` + `peft_llm` + `application` at once. Use `notes` for doubtful cases.
