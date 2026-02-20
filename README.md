# NestIQ: Multi-Task Property Intelligence

NestIQ is a PyTorch + Streamlit project that predicts:
- `Sale Price` (regression)
- `Monthly Rent` (regression)
- `Demand Class` (`High Demand` or `Low Demand`, classification)
- `Demand Confidence` (probability percentage)

The model is trained with multi-task learning using a shared representation and two heads.

## Project Structure

- `data.py`: Synthetic multi-task dataset generation (`5` inputs, `2` regression targets, `1` binary class)
- `model.py`: `MultiTaskPropertyModel` (shared base + regression head + classification head)
- `train.py`: Joint training loop with combined loss and checkpoint saving
- `evaluate.py`: Model evaluation script (regression error + classification accuracy)
- `inference.py`: Streamlit app UI + model loading + inference logic
- `app.py`: App entrypoint (`run_app()`)
- `main.py`: Same as app entrypoint for convenience
- `requirements.txt`: Runtime dependencies
- `saved_model.pt`: Trained checkpoint (created/updated by `train.py`)

## Model Definition

### Inputs
- `area`
- `bedrooms`
- `bathrooms`
- `distance_from_city_center`
- `property_age`

### Outputs
- Regression head: `2` values
  - `sale_price`
  - `rent_price`
- Classification head: `1` logit
  - `high_demand` probability via sigmoid

### Architecture
- Shared base (MLP)
- Regression head: linear layer with `2` outputs
- Classification head: linear layer with `1` output

## Training

Training is multi-task and uses both losses per batch:

- Regression: `MSELoss`
- Classification: `BCEWithLogitsLoss`
- Total: `total_loss = regression_loss + classification_loss`

Regression targets are normalized inside the dataset for stable optimization. During inference/evaluation, predictions are de-normalized using stored mean/std.

### Run Training

```bash
python train.py
```

This saves:
- `model_state_dict`
- `input_dim`, `hidden_dim`
- `reg_mean`, `reg_std`

to `saved_model.pt`.

## Evaluation

Run:

```bash
python evaluate.py
```

Printed metrics:
- Regression MSE
- Regression MAE
- Classification Accuracy

## Inference and UI

The Streamlit UI keeps the existing design language and includes:
- Property feature controls
- Prediction cards for Sale and Rent
- Demand prediction section with:
  - Label badge (`High Demand` in green, `Low Demand` in red)
  - Confidence percentage

### Run App

```bash
streamlit run app.py
```

or

```bash
streamlit run inference.py
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- If `saved_model.pt` does not exist or has old format, app falls back to heuristic predictions so UI remains usable.
- Re-run `train.py` to generate a fresh multitask checkpoint and use full model inference.
- Input defaults in the UI are set to `0` so you can select values manually after app launch.
