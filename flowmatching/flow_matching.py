import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from utils import Bunch


class FlowMatching:
    def __init__(
        self,
        sourceloader,
        targetloader,
        model,
        mode="velocity",
        t_dist="uniform",
        device=None,
        normalize_pred=False,
        geometric=False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric

        self.best_loss = float('inf')
        self.best_model_state = None
        self.input_dim = None

    def sample_from_loader(self, loader):
        """Sample a batch from dataloader"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            return batch[0].to(self.device)
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            if hasattr(loader.dataset, '__getitem__'):
                dummy = loader.dataset[0][0]
                return torch.zeros(loader.batch_size, *dummy.shape, device=self.device)
            return torch.zeros(loader.batch_size, 1, device=self.device)

    def sample_time_and_flow(self):
        """Sample time t and flow (for velocity or target mode)"""
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]

        if self.t_dist == "beta":
            alpha, beta_param = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,)).to(self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        t_pad = t.view(-1, *([1] * (x0.dim() - 1)))
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon
        ut = x1 - x0

        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, eps=epsilon, batch_size=batch_size)

    def forward(self, flow):
        flow_pred = self.model(flow.xt, flow.t)
        return None, flow_pred

    def loss_fn(self, flow_pred, flow):
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def vector_field(self, xt, t):
        """Compute vector field at point xt and time t"""
        _, pred = self.forward(Bunch(xt=xt, t=t, batch_size=xt.size(0)))
        return pred if self.mode == "velocity" else pred - xt

    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, 
          log_freq=5, accum_steps=None):
        """Train with optional gradient accumulation"""
        self.sigma = sigma
        last_loss = 1e99
        patience_count = 0
        pbar = tqdm(range(n_iters), desc="Training steps")
        
        accum_count = 0
        accumulated_loss = 0
        
        use_grad_accum = accum_steps is not None and accum_steps > 1
        effective_accum_steps = accum_steps if use_grad_accum else 1
        
        for i in pbar:
            flow = self.sample_time_and_flow()
            _, flow_pred = self.forward(flow)
            _, loss = self.loss_fn(flow_pred, flow)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                continue
            
            loss_scaled = loss / effective_accum_steps if use_grad_accum else loss
            
            loss_scaled.backward()
            accum_count += 1
            accumulated_loss += loss.item()
            
            should_update = (not use_grad_accum) or (accum_count == effective_accum_steps) or (i == n_iters - 1)
            
            if should_update:
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
                
                avg_loss = accumulated_loss / accum_count
                
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                if avg_loss > last_loss:
                    patience_count += 1
                    if patience_count >= patience:
                        logging.info(f"Early stopping at iteration {i}")
                        break
                else:
                    patience_count = 0
                
                last_loss = avg_loss
                accum_count = 0
                accumulated_loss = 0
            
            if i % log_freq == 0:
                display_loss = loss.item()
                desc = f"Iters [loss {display_loss:.6f}"
                if use_grad_accum:
                    desc += f", accum {accum_count}/{effective_accum_steps}"
                desc += "]"
                pbar.set_description(desc)
        
        if use_grad_accum and accum_count > 0:
            optimizer.step()
            optimizer.zero_grad()

    def map(self, x0, n_steps=50, return_traj=False, method="euler"):
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size, flat_dim = x0.size()
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        for i, t in enumerate(times[:-1]):
            with torch.no_grad():
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                pred = self.model(xt, t_tensor)
                if pred.dim() > 2: pred = pred.squeeze(-1)
                vt = pred if self.mode == "velocity" else pred - xt
                
                if method == "euler":
                    xt = xt + vt * dt
                elif method == "rk4":
                    k1 = vt
                    k2 = self.model(xt + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                    if k2.dim() > 2: k2 = k2.squeeze(-1)
                    k2 = k2 if self.mode == "velocity" else k2 - (xt + 0.5 * dt * k1)
                    k3 = self.model(xt + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                    if k3.dim() > 2: k3 = k3.squeeze(-1)
                    k3 = k3 if self.mode == "velocity" else k3 - (xt + 0.5 * dt * k2)
                    k4 = self.model(xt + dt * k3, t_tensor + dt)
                    if k4.dim() > 2: k4 = k4.squeeze(-1)
                    k4 = k4 if self.mode == "velocity" else k4 - (xt + dt * k3)
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()
        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, source_noise_std=0.001, **map_kwargs):
        assert self.input_dim is not None, "Set `self.input_dim` before generating weights."
        source_samples = torch.randn(n_samples, self.input_dim, device=self.device) * source_noise_std
        return self.map(source_samples, **map_kwargs)


class WeightSpaceFlowModel(nn.Module):
    """Standard flow model for weight space"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1, time_embed_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2), 
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)


class VisionTransformerFlowModel(nn.Module):
    """Flow model specifically designed for Vision Transformers and BERT"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1, time_embed_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Deeper network for complex transformer weight spaces
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Zero initialization for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
        logging.info(f"VisionTransformerFlowModel: input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)