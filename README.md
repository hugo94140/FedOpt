# FedOpt: A Communication-Flexible Federated Learning Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
</p>

## Overview

**FedOpt** is an open-source Federated Learning (FL) framework designed to study the interplay between FL algorithms and communication protocols. Unlike existing frameworks that hardwire specific transport mechanisms, FedOpt natively supports multiple communication protocols (TCP, HTTP, MQTT, gRPC), enabling systematic investigation of protocol-specific effects on FL performance.

FedOpt is the companion framework for **FederNet**, an emulation platform for FL benchmarking under realistic network and device conditions. Together, they provide a comprehensive environment for developing, testing, and evaluating FL algorithms.

### Key Features

- **Multi-Protocol Support**: Native support for TCP, HTTP/REST, MQTT, and gRPC communication protocols
- **Synchronous & Asynchronous FL**: Supports both synchronous (FedAvg, SCAFFOLD, FedDyn) and asynchronous (FedAsync, AsyncFedED, ASOFed) algorithms
- **Optimization Techniques**: Built-in model pruning, dynamic client selection, and client sampling strategies
- **Extensible Architecture**: Plugin-based design for easy addition of new protocols, algorithms, and datasets
- **Docker Support**: Containerized deployment for integration with FederNet emulation platform

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker for containerized deployment
- (Optional) Mosquitto MQTT broker for MQTT protocol

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/antonio-boiano/FedOpt.git
cd FedOpt

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The main dependencies are:
- **PyTorch** & **torchvision**: Deep learning framework
- **grpcio** & **grpcio-tools**: gRPC communication support
- **paho-mqtt**: MQTT protocol support
- **aiohttp**: HTTP/REST protocol support
- **fedlab**: Dataset partitioning utilities
- **medmnist**: Medical imaging dataset
- **matplotlib** & **pandas**: Visualization and data handling
- **PyYAML**: Configuration management

### Docker Installation

```bash
# Build the Docker image
cd Docker
./build.sh

# Or manually:
docker build -t fedopt -f Docker/Dockerfile . --platform linux/amd64
```

## Quick Start

### Running with Default Configuration

```bash
# Start the server
cd FedOpt/src
python3 run.py --mode server --protocol grpc --port 8082

# In separate terminals, start clients
python3 run.py --mode client --protocol grpc --port 8082 --ip localhost
```

### Using Docker Compose

```bash
cd Docker
docker-compose up
```

### Configuration via YAML

All parameters can be configured through `FedOpt/resources/config.yaml`:

```yaml
# General settings
ip: 'localhost'
port: 1883
protocol: "mqtt"  # Options: tcp, mqtt, grpc, http
mode: "Server"
fl_method: "FedAvg"  # Options: FedAvg, FedDyn, SCAFFOLD, FedAsync, etc.
model: "LeNetModified"
device: "cpu"

# Dataset settings
dataset: "MedMNIST"
num_parts_dataset: 10
alpha: 10  # Dirichlet distribution parameter

# Client configuration
client_config:
  epochs: 5
  local_step_size: 0.001

# Server configuration
server_config:
  rounds: 100
  client_round: 5
  min_client_to_start: 5
```

## Component Documentation

### Communication Package (`src/Communication/`)

The communication layer provides a protocol-agnostic abstraction for FL message passing.

| File | Description |
|------|-------------|
| `base.py` | Abstract `BaseClient` and `BaseServer` classes defining the interface for all protocols |
| `registry.py` | Protocol registry pattern and factory for dynamic protocol instantiation |
| `communication_manager.py` | High-level manager for creating and running communication instances |
| `protocols/` | Concrete protocol implementations (gRPC, MQTT, TCP, HTTP) |

**Supported Protocols:**

| Protocol | Description | Use Case |
|----------|-------------|----------|
| gRPC | High-performance RPC with Protocol Buffers | Production deployments, low latency |
| gRPC-stream | Bidirectional streaming gRPC | Real-time updates |
| MQTT | Publish-subscribe messaging | IoT environments, unreliable networks |
| TCP | Raw socket communication | Simple deployments, debugging |
| HTTP | RESTful communication | Web-based integrations |

### Federation Package (`src/Federation/`)

Contains FL algorithm implementations and model definitions.

| Component | Description |
|-----------|-------------|
| `abstract.py` | `AbstractAggregation` (server-side) and `AbstractModel` (client-side) base classes |
| `manager.py` | Factory functions `federation_manager()` and `model_manager()` |
| `dataset.py` | Dataset classes (MNIST, CIFAR10, MedMNIST) with IID/non-IID partitioning |
| `nn.py` | Neural network architectures (AlexNet, LeNet, VGG, MLP variants) |

**Supported Algorithms:**

| Algorithm | Type | Description |
|-----------|------|-------------|
| FedAvg | Synchronous | Baseline federated averaging |
| FedAvgN | Synchronous | FedAvg with Adam optimizer |
| FedDyn | Synchronous | Dynamic regularization |
| SCAFFOLD | Synchronous | Variance reduction with control variates |
| FedAsync | Asynchronous | Staleness-weighted aggregation |
| AsyncFedED | Asynchronous | Euclidean distance-based weighting |
| ASOFed | Asynchronous | Adaptive staleness optimization |
| Unweighted | Asynchronous | Simple async without staleness weighting |

### Optimizations Package (`src/Optimizations/`)

| Module | Description |
|--------|-------------|
| `ClientSelection/` | Score-based client selection using data entropy, communication time, and training efficiency |
| `DynamicSampling/` | Exponential decay sampling: `K_t = K_0 / exp(λt)` |
| `ModelPruning/` | Filter pruning for CNNs based on L1 norm importance |

### Utilities Package (`src/Utils/`)

| Module | Description |
|--------|-------------|
| `logging_fedopt.py` | Configurable logging with stdout handler |
| `Decorators/timer.py` | `@normal_timer` and `@thread_timer` decorators for performance measurement |
| `Extractor/` | Scripts for parsing logs and generating accuracy/time plots |

## Supported Datasets

| Dataset | Classes | Image Size | Description |
|---------|---------|------------|-------------|
| MNIST | 10 | 28×28×1 | Handwritten digits |
| CIFAR-10 | 10 | 32×32×3 | Natural images |
| MedMNIST (PathMNIST) | 9 | 28×28×3 | Colorectal cancer histology |

### Data Partitioning

- **IID**: Uniform random distribution across clients
- **Non-IID (Dirichlet)**: `p_k,w ~ Dir(α)` where `α` controls heterogeneity
  - `α → ∞`: IID-like distribution
  - `α → 0`: Highly skewed distribution
- **Non-IID (Class Skew)**: Each client has dominant classes

## Neural Network Models

| Model | Parameters | Dataset Compatibility |
|-------|------------|----------------------|
| MLP | ~12K | MNIST |
| LeNet-5 | ~62K | MNIST, MedMNIST |
| AlexNet variants | ~150K-850K | All datasets |
| VGG variants | ~1.5M+ | CIFAR-10, MedMNIST |

## Command Line Arguments

```bash
python3 run.py [OPTIONS]

General:
  --ip              Server IP address
  --port            Communication port
  --protocol        Protocol: tcp, mqtt, grpc, http
  --mode            Mode: server, client
  --fl_method       Algorithm: FedAvg, FedDyn, SCAFFOLD, FedAsync, etc.
  --model           Model: AlexNet, LeNet, MLP, VGG
  --device          Device: cpu, cuda

Dataset:
  --dataset         Dataset: MNIST, CIFAR10, MedMNIST
  --parts_dataset   Number of data partitions
  --alpha           Dirichlet concentration parameter
  --iid             IID ratio (1.0 = fully IID)

Training:
  --epochs          Local epochs per round
  --lr              Learning rate
  --rounds          Total FL rounds
  --num_clients     Clients per round
  --min_clients     Minimum clients to start

Optimizations:
  --dyn_sampling    Enable dynamic sampling (true/false)
  --client_selection Enable client selection (true/false)
  --model_pruning   Enable model pruning (true/false)
  --prune_interval  Rounds between pruning
  --prune_ratio     Fraction of filters to prune

Output:
  --accuracy_file   Path for accuracy plot
  --verbose         Logging level: debug, info, warning, error
```

## Adding New Protocols

FedOpt uses a decorator-based registration system:

```python
from FedOpt.src.Communication.base import BaseClient, BaseServer
from FedOpt.src.Communication.registry import register_protocol

@register_protocol("myprotocol", description="Custom protocol")
class MyProtocolClient(BaseClient):
    async def connect(self):
        # Connection logic
        pass
    
    async def disconnect(self):
        # Cleanup logic
        pass
    
    async def send_message(self, message):
        # Send implementation
        pass
    
    async def start_listening(self):
        # Receive loop
        pass

@register_protocol("myprotocol")
class MyProtocolServer(BaseServer):
    async def start(self):
        # Start server
        pass
    
    async def stop(self):
        # Stop server
        pass
    
    async def send_to_client(self, client_id, message):
        # Send to specific client
        pass
    
    async def broadcast(self, message, client_ids=None):
        # Broadcast to clients
        pass
```

## Adding New Algorithms

Extend the abstract classes in `Federation/abstract.py`:

```python
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel

class MyAlgorithm(AbstractAggregation):
    def __init__(self, config):
        self.server_model = MyAlgorithmModel(config)
    
    def apply(self, aggregated_dict, num_clients):
        # Server-side aggregation logic
        pass
    
    def get_server_model(self):
        return {"server_model": state_dict_to_json(self.server_model.model.state_dict())}

class MyAlgorithmModel(AbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["client_config"]["local_step_size"])
    
    def train(self, epochs, index):
        # Local training logic
        pass
    
    def get_client_model(self):
        return {"client_model": state_dict_to_json(self.model.state_dict())}
    
    def set_client_model(self, msg):
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
```

Register in `Federation/manager.py`:
```python
def federation_manager(config):
    method = config["fl_method"]
    if method == "MyAlgorithm":
        return MyAlgorithm(config)
    # ...
```

## Running Experiments

### Batch Experiments

Use the provided shell script:

```bash
./SIMULATIONS.sh
```

This script:
1. Starts the server with specified configuration
2. Launches multiple clients
3. Collects logs for analysis

### Example: Comparing Protocols

```bash
# gRPC experiment
python3 run.py --mode server --protocol grpc --fl_method FedAvg --rounds 100 &
for i in {1..5}; do
    python3 run.py --mode client --protocol grpc --index $i &
done

# MQTT experiment (requires Mosquitto broker)
mosquitto -p 1883 -d
python3 run.py --mode server --protocol mqtt --fl_method FedAvg --rounds 100 &
for i in {1..5}; do
    python3 run.py --mode client --protocol mqtt --index $i &
done
```

## Citation

If you use FedOpt in your research, please cite:

```bibtex
@article{boiano2025federnet,
  title={FederNet: A Network and Device-Aware Emulation Platform for Federated Learning Benchmarking},
  author={Boiano, Antonio and Avanzini, Marta and Brambilla, Mattia and Nicoli, Monica and Redondi, Alessandro E. C.},
  journal={Elsevier},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was funded by the European Union under Grant Agreement No 101080564 (Project TRUSTroke). Views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union or The European Health and Digital Agency (HaDEA).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

- **Antonio Boiano** - antonio.boiano@polimi.it Department of Electronics, Information and Bioengineering, Politecnico di Milano
- GitHub: [@antonio-boiano](https://github.com/antonio-boiano)
