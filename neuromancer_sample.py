from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import extract_features_from_query, load_scenario
from optimization.utils import infer_dt_seconds

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from neuromancer.modules.blocks import MLP as NMMLP 


@dataclass
class CostWeights:
    y: float = 1.0
    u: float = 1e-3
    du: float = 1e-2


@dataclass
class MPCSolution:
    X: torch.Tensor
    U: torch.Tensor
    loss: float
    loss_history: Sequence[float]
    solver: str


def compute_scaler(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.full_like(std, 1e-6), std)
    return mean, std


def build_core_network(input_dim: int, hidden_sizes: Sequence[int], device: torch.device) -> nn.Module:
    core = NMMLP(insize=input_dim, outsize=1, hsizes=list(hidden_sizes))
    return core.to(device)


class NeuralSurrogate(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        mv_dim: int,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        predict_delta: bool = True,
    ) -> None:
        super().__init__()
        self.core = core
        self.mv_dim = mv_dim
        self.state_dim = 1
        self.predict_delta = predict_delta
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", torch.clamp(input_std, min=1e-6))
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", torch.clamp(target_std, min=1e-6))

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        z = torch.cat([x[:, :1], u], dim=-1)
        z_norm = (z - self.input_mean) / self.input_std
        y_norm = self.core(z_norm)
        y = y_norm * self.target_std + self.target_mean
        if self.predict_delta:
            return x[:, :1] + y
        return y

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.step(x, u)


def train_surrogate_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    mv_dim: int,
    hidden_sizes: Sequence[int],
    device: torch.device,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[NeuralSurrogate, float]:
    dataset_size = features.shape[0]
    if dataset_size < 2:
        raise ValueError("Not enough samples to train the surrogate model.")
    input_mean, input_std = compute_scaler(features)
    target_mean, target_std = compute_scaler(targets)
    features_norm = (features - input_mean) / input_std
    targets_norm = (targets - target_mean) / target_std
    core = build_core_network(features.shape[1], hidden_sizes, device)
    loader = DataLoader(
        TensorDataset(features_norm, targets_norm),
        batch_size=min(batch_size, dataset_size),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(core.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = core(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= dataset_size
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
    if best_state is not None:
        core.load_state_dict(best_state)
    surrogate = NeuralSurrogate(
        core=core,
        mv_dim=mv_dim,
        input_mean=input_mean.to(device),
        input_std=input_std.to(device),
        target_mean=target_mean.to(device),
        target_std=target_std.to(device),
    ).to(device)
    surrogate.eval()
    return surrogate, best_loss


def prepare_training_tensors(
    df: pd.DataFrame,
    target_col: str,
    mv_cols: Sequence[str],
    window: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = min(window, len(df) - 1)
    if window < 1:
        raise ValueError("Window is too short to build training sequences.")
    history = df.tail(window + 1).copy().reset_index(drop=True)
    y = torch.tensor(history[target_col].astype(np.float32).values, device=device)
    U = torch.tensor(history[mv_cols].astype(np.float32).values, device=device)
    features = torch.cat([y[:-1].unsqueeze(-1), U[:-1]], dim=1)
    targets = (y[1:] - y[:-1]).unsqueeze(-1)
    return features, targets


def try_neuromancer_solver(
    surrogate: NeuralSurrogate,
    x0: torch.Tensor,
    u_prev: torch.Tensor,
    setpoint: float,
    u_min: np.ndarray,
    u_max: np.ndarray,
    du_max: np.ndarray,
    Np: int,
    Nu: int,
    weights: CostWeights,
    device: torch.device,
) -> Optional[MPCSolution]:
    try:
        from neuromancer.constraint import equality, variable  # type: ignore
        from neuromancer.loss import PenaltyLoss, objective  # type: ignore
        from neuromancer.optimizers.ocp import OCPSolver  # type: ignore
        from neuromancer.problem import Problem  # type: ignore
    except Exception as err:  # pylint: disable=broad-except
        print(f"[Neuromancer] unable to import OCPSolver components ({err}); using differentiable fallback.")
        return None
    print("[Neuromancer] Native OCPSolver wiring is not yet configured in this script; using differentiable fallback.")
    return None


class ManualDifferentiableNMPC:
    def __init__(
        self,
        surrogate: NeuralSurrogate,
        Np: int,
        Nu: int,
        sp: float,
        u_min: np.ndarray,
        u_max: np.ndarray,
        du_max: np.ndarray,
        weights: CostWeights,
        device: torch.device,
        lr: float = 5e-2,
        max_iters: int = 400,
        patience: int = 120,
    ) -> None:
        self.sur = surrogate
        self.Np = int(Np)
        self.Nu = int(max(1, Nu))
        self.mv_dim = surrogate.mv_dim
        self.device = device
        self.lr = lr
        self.max_iters = max_iters
        self.patience = patience
        self.weights = weights
        self.sp = torch.tensor([sp], dtype=torch.float32, device=device)
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=device)
        self.du_max = torch.tensor(du_max, dtype=torch.float32, device=device)

    def solve(self, x0: torch.Tensor, u_prev: torch.Tensor) -> MPCSolution:
        decision = torch.nn.Parameter(torch.zeros(self.Nu, self.mv_dim, device=self.device))
        optimizer = torch.optim.Adam([decision], lr=self.lr)
        best: Optional[dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        best_iter = -1
        loss_history: list[float] = []
        for iteration in range(self.max_iters):
            optimizer.zero_grad(set_to_none=True)
            loss, X, U = self._loss_and_traj(decision, x0, u_prev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([decision], 10.0)
            optimizer.step()
            loss_value = float(loss.item())
            loss_history.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best = {"X": X.detach().clone(), "U": U.detach().clone()}
                best_iter = iteration
            if self.patience and (iteration - best_iter) >= self.patience:
                break
        if best is None:
            raise RuntimeError("Differentiable NMPC failed to produce a feasible plan.")
        return MPCSolution(
            X=best["X"],
            U=best["U"],
            loss=best_loss,
            loss_history=loss_history,
            solver="manual-differentiable (Neuromancer fallback)",
        )

    def _loss_and_traj(
        self,
        decision: torch.Tensor,
        x0: torch.Tensor,
        u_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        du_scale = self.du_max.view(1, -1)
        dU = du_scale * torch.tanh(decision)
        u_sequence = []
        u_k = u_prev.clone()
        for k in range(self.Nu):
            u_k = torch.clamp(u_k + dU[k], self.u_min, self.u_max)
            u_sequence.append(u_k)
        if u_sequence:
            U = torch.stack(u_sequence, dim=0)
        else:
            U = torch.zeros(0, self.mv_dim, device=self.device)
        if self.Np > self.Nu:
            hold_source = U[-1] if U.shape[0] > 0 else u_prev
            hold = hold_source.unsqueeze(0).repeat(self.Np - self.Nu, 1)
            U_full = torch.cat([U, hold], dim=0) if U.shape[0] > 0 else hold
        else:
            U_full = U[: self.Np]
        if U_full.shape[0] < self.Np:
            tail = U_full[-1].unsqueeze(0).repeat(self.Np - U_full.shape[0], 1)
            U_full = torch.cat([U_full, tail], dim=0)
        x_k = x0.clone()
        X_list = [x_k]
        for k in range(self.Np):
            u_k = U_full[k].unsqueeze(0)
            x_k = self.sur.step(x_k, u_k)
            X_list.append(x_k)
        X = torch.cat(X_list, dim=0)
        tracking = self.weights.y * torch.sum((X[1:, 0] - self.sp.squeeze()) ** 2)
        if U_full.numel() > 0:
            effort = self.weights.u * torch.sum(U_full[: self.Nu] ** 2)
            u_stack = torch.cat([u_prev.unsqueeze(0), U_full[: self.Nu]], dim=0)
            du = u_stack[1:] - u_stack[:-1]
            slew = self.weights.du * torch.sum(du ** 2)
        else:
            effort = torch.tensor(0.0, device=self.device)
            slew = torch.tensor(0.0, device=self.device)
        loss = tracking + effort + slew
        return loss, X, U_full


def run_pipeline() -> None:
    base_dir = Path(__file__).resolve().parent
    scenario_path = base_dir / "scenarios" / "automotive" / "SCENARIO_11.json"
    data_path = base_dir / "test_data" / "automotive" / "automotive_press_003.csv"
    scenario = load_scenario(str(scenario_path))
    input_request = (
        scenario.get("agent_workflow", {})
        .get("step_6_orchestration_to_autocontrol", {})
        .get("request", {})
    )
    if not input_request:
        raise KeyError("Scenario is missing the orchestration-to-autocontrol request block.")
    feature_query = input_request.get("feature_names", [])
    target_query = input_request.get("target_col")
    target_setpoint = input_request.get("control_setpoint")
    control_horizon_minutes = input_request.get("control_horizon_minutes", 1.0)
    prediction_multiplier = 2.0
    history_multiplier = 4.0
    df_raw = pd.read_csv(data_path)
    dataset_columns = [c for c in df_raw.columns if c != "TIMESTAMP"]
    extracted_X, extracted_y, _ = extract_features_from_query(feature_query, target_query, dataset_columns)
    mv_cols = [mv for mv in extracted_X if mv is not None]
    if not mv_cols:
        raise ValueError("No manipulated variables were matched from the scenario query.")
    if extracted_y is None:
        raise ValueError("Failed to map target column from the scenario query.")
    target_col = extracted_y
    if target_setpoint is None:
        target_setpoint = float(df_raw[target_col].median())
    else:
        target_setpoint = float(target_setpoint)
    control_horizon_minutes = float(control_horizon_minutes)
    use_columns = ["TIMESTAMP", target_col] + mv_cols
    missing_cols = [col for col in use_columns if col not in df_raw.columns]
    if missing_cols:
        raise KeyError(f"Dataset is missing required columns: {missing_cols}")
    df_model = df_raw[use_columns].dropna().reset_index(drop=True)
    if len(df_model) < 20:
        raise ValueError("Not enough clean samples in the dataset to build the controller.")
    dt = infer_dt_seconds(df_model)
    n_rows = len(df_model)
    total_minutes = max(n_rows - 1, 1) * dt / 60.0
    prediction_horizon_minutes = control_horizon_minutes * prediction_multiplier
    history_horizon_minutes = min(control_horizon_minutes * history_multiplier, total_minutes)
    pred_steps = max(int(round(prediction_horizon_minutes * 60.0 / dt)), 1)
    ctrl_steps = max(int(round(control_horizon_minutes * 60.0 / dt)), 1)
    window_samples = int(round(history_horizon_minutes * 60.0 / dt))
    window_samples = min(max(window_samples, pred_steps + 2, 48), n_rows - 2)
    window_samples = max(window_samples, 10)
    features, targets = prepare_training_tensors(df_model, target_col, mv_cols, window_samples, DEVICE)
    surrogate, train_loss = train_surrogate_model(
        features,
        targets,
        mv_dim=len(mv_cols),
        hidden_sizes=(128, 128),
        device=DEVICE,
        epochs=450,
        lr=5e-3,
    )
    u_min = np.array([df_model[mv].min() for mv in mv_cols], dtype=np.float32)
    u_max = np.array([df_model[mv].max() for mv in mv_cols], dtype=np.float32)
    du_max = 0.05 * (u_max - u_min + 1e-6)
    x0_value = float(df_model[target_col].iloc[-1])
    x0_tensor = torch.tensor([[x0_value]], dtype=torch.float32, device=DEVICE)
    u_prev_tensor = torch.tensor(
        df_model[mv_cols].iloc[-1].astype(np.float32).values,
        dtype=torch.float32,
        device=DEVICE,
    )
    weights = CostWeights()
    Nu = max(1, min(pred_steps, ctrl_steps))
    solution = try_neuromancer_solver(
        surrogate,
        x0_tensor,
        u_prev_tensor,
        target_setpoint,
        u_min,
        u_max,
        du_max,
        pred_steps,
        Nu,
        weights,
        DEVICE,
    )
    if solution is None:
        solver = ManualDifferentiableNMPC(
            surrogate=surrogate,
            Np=pred_steps,
            Nu=Nu,
            sp=target_setpoint,
            u_min=u_min,
            u_max=u_max,
            du_max=du_max,
            weights=weights,
            device=DEVICE,
        )
        solution = solver.solve(x0_tensor, u_prev_tensor)
    time_idx = np.arange(pred_steps, dtype=float) * dt
    df_future = pd.DataFrame({"t_sec": time_idx, "t_min": time_idx / 60.0})
    x_traj = solution.X.detach().cpu().numpy()
    u_traj = solution.U.detach().cpu().numpy()
    df_future[target_col] = x_traj[1:, 0]
    for j, mv in enumerate(mv_cols):
        df_future[mv] = u_traj[:, j]
    print(
        f"[HORIZON] control={control_horizon_minutes:.1f} min ({ctrl_steps} steps) "
        f"prediction={prediction_horizon_minutes:.1f} min ({pred_steps} steps) "
        f"history={history_horizon_minutes:.1f} min ({window_samples} samples)"
    )
    print(
        f"[Neuromancer] surrogate training loss (normalized MSE): {train_loss:.4e} "
        f"on {features.shape[0]} samples"
    )
    print(f"[Neuromancer] solver used: {solution.solver}, objective={solution.loss:.4e}")
    if u_traj.size > 0:
        print(
            f"[Neuromancer] first optimal move: {u_traj[0]}  "
            f"setpoint: {target_setpoint:.2f}  x0: {x0_value:.2f}"
        )
    preview_steps = min(Nu, len(df_future))
    print(f"\n[PREDICTION TRAJECTORY - CONTROL HORIZON (first {preview_steps} steps)]")
    if preview_steps > 0:
        print(df_future.iloc[:preview_steps].to_string(index=False))
    else:
        print("No control horizon steps to display.")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
