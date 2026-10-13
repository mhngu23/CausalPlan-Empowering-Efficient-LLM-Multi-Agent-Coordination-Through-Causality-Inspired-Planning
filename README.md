# CausalPlan: Empowering Efficient LLM Multi-Agent Coordination Through Causality-Inspired Planning

[![arXiv](https://img.shields.io/badge/arXiv-2508.13721-b31b1b.svg)](https://arxiv.org/abs/2508.13721)

CausalPlan learns a causal action model from agent trajectories. It then uses that model to guide the action selection of LLM agents in multi-agent coordination, without fine-tuning.

<p align="center"><img src="assets/framework.png" width="100%" alt="CausalPlan framework"/></p>

## Installation

```bash
conda env create -f environment.yaml
conda activate causalplan
pip install -e lib/overcooked_ai
```

- **Partner agents:** download the pretrained partners (SP, PBT, FCP, MEP, COLE, BC) from [COLE-Platform](https://drive.google.com/drive/folders/1s88a_muyG6pVlfcKDKop6R1Fhxr8dcGH). Put them in `./models`.
- **BC partners:** these also need the TensorFlow 1 environment, `environment_bc.yaml`. Install it, then run `pip install -e lib/stable-baselines`.
- **LLM access:** log in to Hugging Face for the planner LLMs. Set `COHERE_API_KEY` for the analysis and explainer model (Command R).

## Usage

Run all commands from `src/`.

```bash
# CausalPlan (Player 1) with a COLE partner
python main_causal.py -l cramped_room --p0 COLE --p1 ProAgent \
    --gpt_planner_model meta-llama/Meta-Llama-3-8B-Instruct --gamma 0.5 --seed 0

# Backbone only (no causal knowledge)
python main_causal.py -l cramped_room --p0 COLE --p1 ProAgent \
    --gpt_planner_model meta-llama/Meta-Llama-3-8B-Instruct -cg False -fh False --seed 0
```

- **Layouts:** `cramped_room`, `asymmetric_advantages`, `coordination_ring`, `forced_coordination`, `counter_circuit`.
- **LLMs:** `google/gemma-1.1-7b-it`, `meta-llama/Meta-Llama-3-8B-Instruct`, `Qwen/Qwen2.5-14B-Instruct-1M`, `meta-llama/Llama-3.3-70B-Instruct`.
- **Causal action matrices:** pretrained matrices are in `data/`. To learn a new one:

```bash
# 1. Collect trajectories with MEP (50 episodes x 400 steps)
python main_causal.py -l counter_circuit --p0 MEP --p1 MEP --save_buffer True --record_action_only True \
    --collect_data p1 --horizon 400 --episode 50 -bp buffer_cc_p1.pt

# 2. Train the SCA model
python main_causal.py -l counter_circuit --train_SCM True -bp buffer_cc_p1.pt \
    -cgp edge_params_with_action_after_200k_train_50k_test_w_op_final_cc_act_only_p1.pt
```

## Acknowledgements

Built on [ProAgent](https://github.com/PKU-Alignment/ProAgent), [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai), [COLE-Platform](https://github.com/liyang619/COLE-Platform) and [Crafter](https://github.com/danijar/crafter).

## Citation

```bibtex
@article{nguyen2025causalplan,
  title   = {CausalPlan: Empowering Efficient LLM Multi-Agent Coordination Through Causality-Inspired Planning},
  author  = {Nguyen, Minh Hoang and Do, Van Dai and Nguyen, Dung and Nguyen, Thin and Le, Hung},
  journal = {arXiv preprint arXiv:2508.13721},
  year    = {2025}
}
```
