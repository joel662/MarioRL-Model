# 🕹️ Beating the World Record with AI: PPO Experiments in Super Mario Bros

This project applies **Reinforcement Learning (RL)**, using **Proximal Policy Optimization (PPO)**, to train AI agents to play *Super Mario Bros*. We implement and compare two distinct training approaches:

- 🎯 **Approach 1**: Train a separate PPO agent per level (1-1 to 1-4)
- 🌐 **Approach 2**: Train a single generalized PPO agent using dynamic level sampling

The comparison focuses on performance, generalization ability, and scalability.

---

## 📁 Project Structure

- `train_multilevel.py` — Trains 4 separate PPO agents (one for each level)  
- `train_generalized_mario.py` — Trains a single PPO agent across all levels with dynamic sampling  
- `continue_training.py` — Resumes training for the generalized PPO agent  
- `continue_train.py` — Resumes training for the per-level PPO agents  
- `mul_level.py` — Contains logic for level selection and sampling during multi-agent training  
- `level.py` — Helper for evaluating agent performance across levels  
- `test.py` — Runs a trained model and evaluates it on a selected level  
- `WRvsAI rec.txt` — Manual timing comparison between world record and AI agent completions  
- `Mario_PPO_Presentation.pptx` — Final project slides used for presentation  
- `Mario_PPO_Report.pdf` — ACM-format report detailing methodology, results, and discussion  
- `requirements.txt` — List of Python libraries required for the project  
- `README.md` — This documentation file

---

## 🧠 Methods

### ✅ Approach 1: Per-Level PPO Agents
- Trains one PPO model per level (1-1 to 1-4)
- Early stopping if completion rate exceeds 95%
- Fast convergence
- High success rate, but no knowledge transfer between levels

### 🌐 Approach 2: Generalized PPO with Dynamic Sampling
- Trains a single PPO agent across all levels
- Uses dynamic sampling weights based on level completion rates:
  ```
  weight = max(0.2, 1 - (completion_rate)^2)
  ```
- Promotes generalization across unseen or less-trained levels

---

## 🔧 Setup Instructions

### ✅ Requirements
- Python 3.8+
- `gym-super-mario-bros`
- `nes-py`
- `stable-baselines3`
- `opencv-python`
- `matplotlib`

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

### Train Per-Level Agents (Approach 1)

```bash
python train_multilevel.py
```

### Train Generalized Agent (Approach 2)

```bash
python train_generalized_mario.py
```

### Resume Training

```bash
python continue_training.py   # Generalized model
python continue_train.py      # Per-level model
```

---

## 🎯 Evaluate Trained Model

```bash
python test.py --model_path ./models/W1-1.zip --level 1-1
```

---

## 📊 Results

| Level | World Record | Per-Level PPO | Generalized PPO |
|-------|--------------|----------------|------------------|
| 1-1   | 370          | 357            | **366**          |
| 1-2   | 350          | 333            | **338**          |
| 1-3   | 260          | 247            | —                |
| 1-4   | 263          | 259            | **261**          |

---

## 📄 Report & Presentation

- 📄 [ACM Format Report](./Mario_PPO_Report.pdf)
- 🎞️ [Project Presentation](./Mario_PPO_Presentation.pptx)

---

## 🙏 Acknowledgments

- [OpenAI Gym](https://gym.openai.com/)
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- The Mario speedrunning community for benchmark times

---

## 📌 License

This project is for academic use under the [MIT License](./LICENSE).
