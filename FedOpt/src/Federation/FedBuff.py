# FedOpt/src/Federation/FedBuff.py

import random
import time
from copy import deepcopy
from typing import Optional, Dict, Any
import threading

import torch
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Communication.communication import state_dict_to_json, json_to_state_dict
import logging

logger = logging.getLogger("FedBuff")


class FedBuff(AbstractAggregation):
    """
    FedBuff (Buffered Asynchronous Aggregation) for FedOpt.

    Principle:
      - Server buffers client deltas.
      - When buffer reaches K updates, server flushes:
            w <- w + eta * mean(delta)

    + Timeout extension:
      - If buffer stays non-empty for too long (timeout), flush anyway.
    """

    def __init__(self, config: Dict[str, Any]):
        self.device = config["device"]
        self.server_model = FedBuffModel(config)

        # --- FedBuff server hyperparams (put these in server_config) ---
        server_cfg = config.get("server_config", {})
        self.K = int(server_cfg.get("fedBuff_buffer_size", 10))          # buffer size
        self.eta = float(server_cfg.get("fedBuff_server_lr", 1.0))       # server lr
        self.weight_by_samples = bool(server_cfg.get("fedBuff_weight_by_samples", True))

        # --- Timeout flush (seconds) ---
        # If > 0: flush even if buffer < K when it stays too long.
        self.flush_timeout_s = float(server_cfg.get("fedBuff_flush_timeout_s", 0))
        self.timeout_check_interval_s = float(server_cfg.get("fedBuff_timeout_check_interval_s", 0.2))
        self.timeout_min_updates = int(server_cfg.get("fedBuff_timeout_min_updates", 1))

        # --- State ---
        self.global_epoch = 0  # increments on flush only
        self._buffer_count = 0
        self._buffer_weight = 0.0
        self._buffer_acc = None  # accumulated weighted delta state_dict (tensors)
        self._buffer_first_ts = None  # monotonic timestamp when buffer became non-empty

        # For logging/debug
        self._total_updates_received = 0

        # Concurrency guard (watchdog thread may flush while apply() is running)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if self.flush_timeout_s and self.flush_timeout_s > 0:
            self._watchdog_thread = threading.Thread(
                target=self._timeout_watchdog_loop,
                name="FedBuffTimeoutWatchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

        logger.info(
            f"FedBuff initialized with K={self.K}, eta={self.eta}, "
            f"weight_by_samples={self.weight_by_samples}"
        )
        if self.flush_timeout_s and self.flush_timeout_s > 0:
            logger.info(
                f"FedBuff timeout enabled: timeout={self.flush_timeout_s}s "
                f"(check_interval={self.timeout_check_interval_s}s, min_updates={self.timeout_min_updates})"
            )

    def _init_buffer_acc_if_needed(self, reference_state_dict: Dict[str, torch.Tensor]) -> None:
        if self._buffer_acc is None:
            self._buffer_acc = {k: torch.zeros_like(v) for k, v in reference_state_dict.items()}

    def _add_delta_to_buffer(self, delta_sd: Dict[str, torch.Tensor], weight: float) -> None:
        # Safety
        if weight <= 0:
            weight = 1.0

        server_sd = self.server_model.model.state_dict()
        self._init_buffer_acc_if_needed(server_sd)

        # Accumulate: buffer_acc += weight * delta
        with torch.no_grad():
            for k in server_sd.keys():
                self._buffer_acc[k] += (delta_sd[k] * weight)

        self._buffer_count += 1
        self._buffer_weight += weight

        # Start the timer when the buffer becomes non-empty
        if self._buffer_first_ts is None:
            self._buffer_first_ts = time.monotonic()

        # LOG : buffer_len=x/K at every update (no eval here)
        logger.info(
            f"FedBuff BUFFER -> global_epoch={self.global_epoch} "
            f"buffer_len={self._buffer_count}/{self.K}"
        )

    def _flush_locked(self, reason: str) -> bool:
        """
        Flush buffer into the global model.
        Must be called with self._lock held.
        Returns True if a flush happened, else False.
        """
        if self._buffer_count <= 0 or self._buffer_acc is None:
            return False

        server_sd = self.server_model.model.state_dict()

        # Mean delta
        denom = self._buffer_weight if self.weight_by_samples else float(self._buffer_count)
        if denom <= 0:
            denom = float(self._buffer_count) if self._buffer_count > 0 else 1.0

        with torch.no_grad():
            new_sd = {}
            for k in server_sd.keys():
                delta_mean = self._buffer_acc[k] / denom
                new_sd[k] = server_sd[k] + (self.eta * delta_mean)

            self.server_model.model.load_state_dict(new_sd)

        # Reset buffer
        self._buffer_count = 0
        self._buffer_weight = 0.0
        self._buffer_acc = None
        self._buffer_first_ts = None

        # Flush = a real global update
        self.global_epoch += 1
        logger.info(
            f"FedBuff FLUSH[{reason}] -> global_epoch={self.global_epoch} "
            f"buffer_len={self._buffer_count}/{self.K}"
        )
        return True

    def _maybe_flush_locked(self) -> bool:
        """
        Decide whether to flush (K reached OR timeout reached).
        Must be called with self._lock held.
        """
        # Normal FedBuff condition: buffer full
        if self._buffer_count >= self.K:
            return self._flush_locked(reason="K")

        # Timeout condition: buffer non-empty for too long
        if self.flush_timeout_s and self.flush_timeout_s > 0 and self._buffer_first_ts is not None:
            if self._buffer_count >= self.timeout_min_updates:
                elapsed = time.monotonic() - self._buffer_first_ts
                if elapsed >= self.flush_timeout_s:
                    return self._flush_locked(reason="timeout")

        return False

    def apply(self, aggregated_dict: Optional[Dict[str, Any]] = None, num_clients: Optional[int] = None):
        """
        Called by Server.
        In FedOpt async mode, aggregated_dict is usually a dict of 1 message.
        """
        if aggregated_dict is None:
            raise ValueError("aggregated_dict cannot be None.")

        try:
            flushed_any = False

            for _, msg in aggregated_dict.items():
                self._total_updates_received += 1

                # Expecting ASOFed-like delta payload: "delta_model" + "client_samples" + "client_id"
                delta_sd = json_to_state_dict(msg["delta_model"], self.device)

                client_samples = float(msg.get("client_samples", 1.0))
                weight = client_samples if self.weight_by_samples else 1.0

                with self._lock:
                    self._add_delta_to_buffer(delta_sd, weight)
                    if self._maybe_flush_locked():
                        flushed_any = True

            if not flushed_any:
                logger.debug(
                    f"FedBuff buffered updates: {self._buffer_count}/{self.K} "
                    f"(total_received={self._total_updates_received})"
                )

        except Exception as e:
            logger.error(f"Error in FedBuff aggregation: {e}")
            raise

    def _timeout_watchdog_loop(self):
        """
        Background loop: flush on timeout even if no new updates arrive.
        """
        while not self._stop_event.is_set():
            time.sleep(self.timeout_check_interval_s)

            if not (self.flush_timeout_s and self.flush_timeout_s > 0):
                continue

            try:
                with self._lock:
                    if self._buffer_count <= 0 or self._buffer_first_ts is None:
                        continue
                    if self._buffer_count < self.timeout_min_updates:
                        continue

                    elapsed = time.monotonic() - self._buffer_first_ts
                    if elapsed >= self.flush_timeout_s:
                        self._flush_locked(reason="timeout")
            except Exception:
                logger.exception("FedBuff timeout watchdog crashed")

    def shutdown(self):
        """
        Optional: call this if your framework has a server shutdown hook.
        """
        self._stop_event.set()

    def get_server_model(self):
        # Keep same keys as other algos: server_model + global_epoch
        return {
            "server_model": state_dict_to_json(self.server_model.model.state_dict()),
            "global_epoch": self.global_epoch,
            # Optional debug fields (harmless for clients):
            "fedbuff_buffer_fill": self._buffer_count,
            "fedbuff_buffer_size": self.K,
        }

class FedBuffModel(AbstractModel):
    """
    Client-side logic:
      - receive server model
      - train locally
      - compute delta = (w_local - w_server_start)
      - send delta_model + client_samples + client_id (FedOpt-friendly)
    """

    def __init__(self, config: Dict[str, Any]):
        self.name = "FedBuff"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)

        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Keep same metadata pattern as ASOFed
        self.client_id = random.randint(0, 10000)
        self.model_version = 0

        # For delta computation
        self.server_model_snapshot = deepcopy(self.model).to(self.device)
        self.delta_state_dicts = None
        self.num_samples = 0

    @normal_timer
    def train(self, rounds, index):
        """
        In FedOpt, 'rounds' here is actually epochs (BaseClient passes self.epochs).
        """
        logger.info("Training...")
        start_time = time.perf_counter()

        self.model.train()
        if self.dataset is None:
            raise Exception("Dataset is None")

        # Snapshot the server model at the start of local training
        self.server_model_snapshot = deepcopy(self.model).to(self.device)

        # Determine client samples for weighting (as in ASOFed)
        client_loader = self.dataset.train_loader[index % self.dataset.num_parts]
        self.num_samples = len(client_loader.dataset)

        for epoch in range(rounds):
            losses = []
            for inputs, targets in client_loader:
                # MedMNIST handling pattern used across algos
                from FedOpt.src.Federation.dataset import MedMNIST
                if isinstance(self.dataset, MedMNIST):
                    if targets.shape != torch.Size([1, 1]):
                        targets = targets.squeeze().long()
                    else:
                        targets = torch.tensor(targets[:, 0])

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                losses.append(loss.detach())

            logger.info(f"EPOCH: {epoch + 1}/{rounds}, LOSS: {torch.tensor(losses).mean():.4f}")

        # Compute delta: w_local - w_server_start
        local_sd = self.model.state_dict()
        base_sd = self.server_model_snapshot.state_dict()
        self.delta_state_dicts = {k: (local_sd[k] - base_sd[k]) for k in local_sd.keys()}

        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time

    def get_client_model(self):
        if self.delta_state_dicts is None:
            # Fallback: send zero delta if somehow called before training
            base_sd = self.model.state_dict()
            zero_delta = {k: torch.zeros_like(v) for k, v in base_sd.items()}
            delta_json = state_dict_to_json(zero_delta)
        else:
            delta_json = state_dict_to_json(self.delta_state_dicts)

        return {
            "delta_model": delta_json,
            "client_samples": self.num_samples,
            "client_id": self.client_id,
            "model_version": self.model_version,  # kept for consistency with FedAsync style
        }

    def set_client_model(self, msg):
        # Server sends server_model + global_epoch
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
        self.model_version = msg.get("global_epoch", 0)
