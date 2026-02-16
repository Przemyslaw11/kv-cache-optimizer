# KV CACHE OPTIMIZATION FOR PROMPT PROCESSING
## Comprehensive Execution Plan (8-12 Weeks)

**Project ID:** C1  
**Target Venue:** ICML 2026 (Submission: January 31, 2026)  
**Computational Resources:** 8× NVIDIA A100-40GB, Athena Supercomputer  
**Timeline:** 8-12 weeks (flexible based on results)  
**Risk Level:** Low  
**Expected Outcome:** High-quality publication at ICML or NeurIPS 2026

---

## EXECUTIVE SUMMARY

### Research Objective
Extend KVQuant's per-channel quantization technique to the **prefill phase** of LLM inference, implementing blocked memory allocation to eliminate reallocation overhead, and provide the **first energy-validated analysis** of KV cache compression during prompt processing.

### Key Innovation
1. **Batched quantization algorithm** for variable-length prompts during prefill
2. **Blocked CSR/CSC allocation** reducing memory overhead by 40-60%
3. **Fused attention-quantization kernels** for efficient prefill
4. **Energy analysis** showing prefill quantization enables 2-3× throughput improvements

### Success Criteria
- ✅ <5% latency overhead vs. fp16 prefill
- ✅ 3× larger batch size enabled by quantization
- ✅ <0.1 perplexity degradation on LongBench
- ✅ Energy per token reduced proportionally to throughput improvement
- ✅ ICML 2026 acceptance (or NeurIPS 2026 backup)

### Why This Matters
- **Scientific Impact:** First systematic study of KV quantization for prefill phase with hardware-validated metrics
- **Practical Impact:** Enables 2-3× larger batch sizes in production, reducing cost-per-token by 40-60%
- **Alignment:** Perfect fit with your energy efficiency + hardware profiling expertise

---

## PHASE-BY-PHASE ROADMAP

### **PHASE 1: SETUP & BASELINE (WEEKS 1-2)**
**Goal:** Reproduce KVQuant results, establish baseline measurements, set up infrastructure

### **PHASE 2: BATCHED QUANTIZATION ALGORITHM (WEEK 3)**
**Goal:** Implement variable-length batch quantization, validate accuracy

### **PHASE 3: BLOCKED MEMORY ALLOCATION (WEEKS 4-5)**
**Goal:** Design and implement zero-copy block-based allocation

### **PHASE 4: FUSED KERNELS (WEEK 6)**
**Goal:** Optimize prefill-quantization pipeline with Triton

### **PHASE 5: ACCURACY VALIDATION (WEEK 7)**
**Goal:** Comprehensive evaluation on LongBench, RULER, Passkey Retrieval

### **PHASE 6: THROUGHPUT BENCHMARKING (WEEK 8)**
**Goal:** Systematic throughput measurement across batch sizes and prompt lengths

### **PHASE 7: ENERGY PROFILING (WEEK 9)**
**Goal:** Hardware-validated energy measurements with NVML

### **PHASE 8: ABLATIONS & ANALYSIS (WEEK 10)**
**Goal:** Ablation studies, statistical analysis, Pareto frontiers

### **PHASE 9: PAPER WRITING (WEEKS 11-12)**
**Goal:** Draft, revise, and submit to ICML 2026

---

## DETAILED WEEK-BY-WEEK EXECUTION PLAN

---

## WEEK 1: Environment Setup & Repository Preparation

### Objectives
- [ ] Set up development environment on Athena
- [ ] Clone and configure KVQuant repository
- [ ] Verify baseline KVQuant results
- [ ] Install all dependencies
- [ ] Configure Slurm job submission

### Tasks

#### Day 1: Environment Setup
```bash
# SSH into Athena
ssh your_username@athena.cyfronet.pl

# Load required modules
module load cuda/12.4.0
module load python/3.10
module load pytorch/2.4.0

# Create project directory
cd $PLG_GROUPS_STORAGE
mkdir -p kvquant-prefill
cd kvquant-prefill

# Clone repositories
git clone https://github.com/SqueezeAILab/KVQuant.git
cd KVQuant

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --break-system-packages -r requirements.txt
pip install --break-system-packages triton==2.1.0
pip install --break-system-packages pynvml
pip install --break-system-packages codecarbon
pip install --break-system-packages wandb
pip install --break-system-packages optuna
```

#### Day 2: Download Models & Datasets
```bash
# Download Llama-2-7B-32K model
cd $PLG_GROUPS_STORAGE/kvquant-prefill
mkdir -p models
cd models

# Using HuggingFace CLI
huggingface-cli login  # Use your token
huggingface-cli download togethercomputer/Llama-2-7B-32K-Instruct --local-dir ./Llama-2-7B-32K

# Download datasets
cd $PLG_GROUPS_STORAGE/kvquant-prefill
mkdir -p datasets

# LongBench (download script)
git clone https://github.com/THUDM/LongBench.git
cd LongBench
python download_data.py  # Downloads to data/
```

**Storage Breakdown:**
- Llama-2-7B-32K: ~14GB
- Llama-2-13B-32K: ~26GB (optional for later)
- LongBench: ~5GB
- RULER (generated): ~2GB
- Total: ~25GB (well within limits)

#### Day 3: Reproduce KVQuant Baseline
**Goal:** Verify you can reproduce the NeurIPS paper results

```bash
# Test KVQuant generation (autoregressive decoding)
cd $PLG_GROUPS_STORAGE/kvquant-prefill/KVQuant

# Run inference with quantized KV cache (generation phase)
python main.py \
  --model_path ../models/Llama-2-7B-32K \
  --dataset wikitext2 \
  --quantization nuq3 \
  --sparsity 0.01 \
  --max_seq_len 2048
```

**Expected Result:** Should match Table 1 from KVQuant paper:
- fp16 baseline: 5.12 PPL
- nuq3-1%: 5.17 PPL (~0.05 degradation)

**If results don't match:**
- Check CUDA version compatibility
- Verify calibration samples are identical
- Check random seed settings

#### Day 4: Baseline Prefill Measurements
**Create baseline measurement script:**

```python
# File: scripts/measure_prefill_baseline.py
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml

def measure_prefill_latency(model, tokenizer, prompt_lengths, batch_sizes):
    """Measure fp16 prefill latency baseline"""
    results = {}
    
    for prompt_len in prompt_lengths:
        for batch_size in batch_sizes:
            # Generate dummy prompts
            prompt = "Hello " * prompt_len
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt", 
                             padding=True).to("cuda")
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(**inputs)
            
            # Measure
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(100):
                with torch.no_grad():
                    _ = model(**inputs)
            
            torch.cuda.synchronize()
            end = time.time()
            
            latency = (end - start) / 100
            tokens_per_sec = (prompt_len * batch_size) / latency
            
            results[(prompt_len, batch_size)] = {
                'latency_sec': latency,
                'tokens_per_sec': tokens_per_sec
            }
            
            print(f"Prompt {prompt_len}, Batch {batch_size}: "
                  f"{latency:.3f}s, {tokens_per_sec:.1f} tok/s")
    
    return results

# Run baseline
model = AutoModelForCausalLM.from_pretrained(
    "../models/Llama-2-7B-32K",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-7B-32K")

prompt_lengths = [512, 2048, 8192, 32768]
batch_sizes = [1, 4, 16, 32]

baseline_results = measure_prefill_latency(model, tokenizer, 
                                          prompt_lengths, batch_sizes)

# Save results
import json
with open('baseline_prefill_results.json', 'w') as f:
    json.dump(baseline_results, f, indent=2)
```

**Run baseline measurement:**
```bash
# Create Slurm job
sbatch scripts/run_baseline_measurement.sh
```

**Expected outputs:**
- Baseline prefill latency for various (prompt_len, batch_size) pairs
- Memory usage statistics
- Tokens/sec throughput

#### Day 5: Set Up Monitoring Infrastructure

**Create WandB project:**
```python
# File: scripts/setup_wandb.py
import wandb

wandb.login()  # Use your API key

# Initialize project
wandb.init(
    project="kvquant-prefill",
    name="baseline-measurements",
    config={
        "model": "Llama-2-7B-32K",
        "phase": "baseline",
        "quantization": "fp16"
    }
)
```

**Create energy monitoring wrapper:**
```python
# File: utils/energy_monitor.py
import pynvml
import time
from codecarbon import EmissionsTracker

class EnergyMonitor:
    """Monitor GPU energy consumption using NVML"""
    
    def __init__(self, gpu_id=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.tracker = EmissionsTracker()
        
    def start(self):
        self.start_time = time.time()
        self.start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self.tracker.start()
        
    def stop(self):
        self.end_time = time.time()
        self.end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self.tracker.stop()
        
        # Energy in Joules (NVML returns millijoules)
        energy_joules = (self.end_energy - self.start_energy) / 1000.0
        duration = self.end_time - self.start_time
        
        return {
            'energy_joules': energy_joules,
            'duration_sec': duration,
            'avg_power_watts': energy_joules / duration,
            'carbon_emissions_kg': self.tracker.final_emissions
        }

# Test energy monitoring
if __name__ == "__main__":
    monitor = EnergyMonitor(gpu_id=0)
    
    monitor.start()
    time.sleep(5)  # Simulate work
    results = monitor.stop()
    
    print(f"Energy: {results['energy_joules']:.2f} J")
    print(f"Avg Power: {results['avg_power_watts']:.2f} W")
```

**Deliverables Week 1:**
- ✅ Athena environment configured
- ✅ KVQuant repository cloned and dependencies installed
- ✅ Models and datasets downloaded
- ✅ KVQuant baseline results reproduced
- ✅ fp16 prefill baseline measurements completed
- ✅ Monitoring infrastructure (WandB, NVML) set up

---

## WEEK 2: Deep Dive into KVQuant Codebase

### Objectives
- [ ] Understand KVQuant's quantization pipeline
- [ ] Identify where prefill vs. generation logic differs
- [ ] Profile memory allocation patterns
- [ ] Design batched quantization algorithm

### Tasks

#### Day 1: Code Reading & Documentation

**Key files to understand:**
```
KVQuant/
├── kvquant/
│   ├── __init__.py
│   ├── modelutils.py          # Model loading utilities
│   ├── datautils.py           # Dataset handling
│   ├── quant.py               # Core quantization logic ⭐
│   ├── nuq.py                 # Non-uniform quantization
│   ├── sparse.py              # Dense-and-sparse quantization
│   └── kernels/
│       ├── triton_kernels.py  # Triton CUDA kernels ⭐
│       └── utils.py
├── main.py                    # Entry point
└── experiments/
    └── generation.py          # Generation-only experiments ⭐
```

**Read and annotate:**
1. `quant.py`: How does per-channel quantization work?
2. `nuq.py`: How are non-uniform datatypes computed offline?
3. `triton_kernels.py`: How are quantized KV caches loaded during generation?
4. `generation.py`: Where is the generation vs. prefill split?

**Create documentation:**
```markdown
# File: docs/kvquant_architecture.md

## KVQuant Pipeline (Generation Phase)

1. **Calibration (Offline):**
   - Load 16 samples from Wikitext-2
   - Compute Fisher information per layer
   - Derive nuqX datatype using sensitivity-weighted K-means
   - Compute per-channel scaling factors for Keys

2. **Inference (Online - Generation Phase):**
   - For each new token generated:
     - Compute Key/Value for new token
     - Quantize new Key/Value using pre-computed datatype
     - Append to KV cache (CSC for Keys, CSR for Values)
     - Use quantized cache for attention computation

3. **Missing: Prefill Phase**
   - Currently, prefill uses fp16
   - Keys/Values are quantized AFTER prefill
   - Opportunity: Quantize DURING prefill for batched prompts
```

#### Day 2: Profile Memory Allocation

**Create memory profiling script:**
```python
# File: scripts/profile_memory.py
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "../models/Llama-2-7B-32K",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-7B-32K")

# Profile prefill
prompt = "Hello " * 2048
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with torch.no_grad():
        outputs = model(**inputs)

# Print memory usage
print(prof.key_averages().table(
    sort_by="cuda_memory_usage", row_limit=20
))

# Export trace
prof.export_chrome_trace("prefill_memory_trace.json")
```

**Analyze memory bottlenecks:**
- Where is memory allocated for KV cache?
- How much memory for 32K tokens × batch size 16?
- What's the allocation pattern? (Realloc on each token or pre-allocate?)

**Expected findings:**
- KV cache memory dominates for long prompts
- Current implementation: Concatenate on each token (inefficient for prefill)
- Opportunity: Pre-allocate blocks, avoid reallocation

#### Day 3: Design Batched Quantization Algorithm

**Challenge:** KVQuant quantizes one token at a time (generation). For prefill, we need to quantize N tokens simultaneously in a batch.

**Design considerations:**
1. **Variable-length prompts:** Each sequence in batch has different length
2. **Padding:** Need to handle padding tokens correctly
3. **Per-channel scaling:** Compute scaling factors across all tokens in sequence
4. **Memory efficiency:** Avoid materializing full fp16 cache before quantization

**Pseudocode:**
```python
def batch_quantize_prefill(keys, values, batch_size, seq_lengths, 
                          scaling_factors, nuq_datatype):
    """
    Quantize Keys and Values during prefill phase
    
    Args:
        keys: [batch_size, num_heads, max_seq_len, head_dim] - fp16
        values: [batch_size, num_heads, max_seq_len, head_dim] - fp16
        batch_size: Number of sequences in batch
        seq_lengths: Actual length of each sequence (variable)
        scaling_factors: Pre-computed per-channel scaling factors
        nuq_datatype: Non-uniform quantization lookup table
    
    Returns:
        quantized_keys: [batch_size, num_heads, max_seq_len, head_dim] - 3-bit indices
        quantized_values: [batch_size, num_heads, max_seq_len, head_dim] - 3-bit indices
        sparse_keys: CSC format sparse matrix for outliers
        sparse_values: CSR format sparse matrix for outliers
    """
    
    quantized_keys = []
    quantized_values = []
    sparse_keys = []
    sparse_values = []
    
    for b in range(batch_size):
        seq_len = seq_lengths[b]
        
        # Extract Keys/Values for this sequence (ignore padding)
        seq_keys = keys[b, :, :seq_len, :]  # [num_heads, seq_len, head_dim]
        seq_values = values[b, :, :seq_len, :]
        
        # Per-channel quantization for Keys
        # Channels = seq_len dimension
        for head in range(num_heads):
            head_keys = seq_keys[head]  # [seq_len, head_dim]
            
            # Normalize per channel (per head_dim)
            for ch in range(head_dim):
                channel_keys = head_keys[:, ch]  # [seq_len]
                
                # Apply scaling factor (pre-computed offline)
                scale = scaling_factors[layer_id][head][ch]
                normalized = channel_keys / scale
                
                # Detect outliers (1% threshold)
                outlier_threshold = compute_outlier_threshold(normalized, pct=0.01)
                outlier_mask = torch.abs(normalized) > outlier_threshold
                
                # Quantize non-outliers using nuq datatype
                non_outliers = normalized[~outlier_mask]
                quantized_indices = quantize_to_nuq(non_outliers, nuq_datatype)
                
                # Store outliers in sparse format
                outliers = channel_keys[outlier_mask]
                outlier_positions = torch.where(outlier_mask)[0]
                sparse_keys.append((outlier_positions, outliers))
                
                # Store quantized indices
                quantized_keys.append(quantized_indices)
        
        # Per-token quantization for Values (similar logic)
        # ...
    
    return quantized_keys, quantized_values, sparse_keys, sparse_values
```

**Key design decisions:**
- Use **pre-computed** scaling factors (from calibration) - no online recomputation
- Apply **per-vector outlier detection** (1% threshold per channel/token)
- Store quantized cache in **blocked format** (design in Week 4)

#### Day 4: Prototype Naive Batched Quantization

**Implement simplified version:**
```python
# File: kvquant/batched_quant.py

import torch
import numpy as np

class BatchedKVQuantizer:
    """Quantize KV cache during prefill phase"""
    
    def __init__(self, config, nuq_datatype, scaling_factors):
        self.config = config
        self.nuq_datatype = nuq_datatype  # Pre-computed lookup table
        self.scaling_factors = scaling_factors  # Pre-computed per-channel
        
    def quantize_keys_batch(self, keys, seq_lengths):
        """
        Quantize Keys for batched prefill
        
        Args:
            keys: [batch_size, num_layers, num_heads, max_seq_len, head_dim]
            seq_lengths: [batch_size] - actual lengths (excluding padding)
        
        Returns:
            quantized_keys: Quantized indices
            sparse_keys: Outliers in sparse format
        """
        batch_size, num_layers, num_heads, max_seq_len, head_dim = keys.shape
        
        quantized_keys = []
        sparse_keys = []
        
        for layer_id in range(num_layers):
            layer_quantized = []
            layer_sparse = []
            
            for b in range(batch_size):
                seq_len = seq_lengths[b]
                
                # Extract sequence (ignore padding)
                seq_keys = keys[b, layer_id, :, :seq_len, :]  # [num_heads, seq_len, head_dim]
                
                # Quantize per-channel (pre-RoPE)
                for head in range(num_heads):
                    for ch in range(head_dim):
                        channel = seq_keys[head, :, ch]  # [seq_len]
                        
                        # Normalize using pre-computed scaling factor
                        scale = self.scaling_factors[layer_id][head][ch]
                        normalized = channel / scale
                        
                        # Detect outliers (top 1%)
                        threshold = torch.quantile(torch.abs(normalized), 0.99)
                        outlier_mask = torch.abs(normalized) > threshold
                        
                        # Quantize non-outliers
                        non_outliers = normalized[~outlier_mask]
                        indices = self._quantize_nuq(non_outliers)
                        
                        # Store
                        layer_quantized.append(indices)
                        layer_sparse.append({
                            'values': channel[outlier_mask],
                            'positions': torch.where(outlier_mask)[0]
                        })
            
            quantized_keys.append(layer_quantized)
            sparse_keys.append(layer_sparse)
        
        return quantized_keys, sparse_keys
    
    def _quantize_nuq(self, values):
        """Quantize values using non-uniform datatype"""
        # Find closest nuq datatype entry for each value
        # nuq_datatype is [2^k entries] lookup table
        
        # Broadcasting: values [N] vs nuq_datatype [2^k]
        distances = torch.abs(values.unsqueeze(1) - self.nuq_datatype.unsqueeze(0))
        indices = torch.argmin(distances, dim=1)
        
        return indices  # [N] - each is k-bit index into nuq_datatype
```

**Test on toy example:**
```python
# Test script
batch_size = 4
num_layers = 2
num_heads = 8
max_seq_len = 128
head_dim = 64

# Dummy data
keys = torch.randn(batch_size, num_layers, num_heads, max_seq_len, head_dim)
seq_lengths = torch.tensor([100, 120, 80, 128])

# Dummy nuq datatype (8 entries for 3-bit)
nuq_datatype = torch.tensor([-1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5])

# Dummy scaling factors
scaling_factors = torch.ones(num_layers, num_heads, head_dim)

# Quantize
quantizer = BatchedKVQuantizer(None, nuq_datatype, scaling_factors)
quantized, sparse = quantizer.quantize_keys_batch(keys, seq_lengths)

print(f"Quantized keys shape: {len(quantized)}")
print(f"Sparse keys count: {len(sparse)}")
```

#### Day 5: Design Blocked Memory Allocation

**Problem:** Current KVQuant appends tokens one-by-one to CSC/CSR sparse matrices. This requires reallocation and copying.

**Solution:** Blocked allocation - pre-allocate memory in blocks, append without reallocation.

**Design:**
```python
class BlockedSparseMatrix:
    """
    Sparse matrix with blocked allocation to avoid reallocation overhead
    
    Structure:
        - Allocate memory in blocks of B tokens (e.g., B=256)
        - When block is full, allocate next block
        - Blocks are linked (like linked list)
        - No copying needed when appending
    
    Format: CSC (Compressed Sparse Column) for Keys
        - columns[block_id]: Column indices for block
        - rows[block_id]: Row indices for block  
        - values[block_id]: Values for block
        - Each block stores up to B tokens
    """
    
    def __init__(self, block_size=256, max_blocks=128):
        self.block_size = block_size  # Tokens per block
        self.max_blocks = max_blocks  # Maximum number of blocks
        
        # Pre-allocate blocks
        self.columns = []  # List of column index arrays
        self.rows = []     # List of row index arrays
        self.values = []   # List of value arrays
        
        self.current_block = 0
        self.tokens_in_current_block = 0
        
        # Allocate first block
        self._allocate_block()
    
    def _allocate_block(self):
        """Allocate a new block of memory"""
        self.columns.append(torch.zeros(self.block_size, dtype=torch.int32))
        self.rows.append([])  # Variable-length per column
        self.values.append([])
        
    def append_token(self, outlier_positions, outlier_values):
        """
        Append outliers for a new token to the cache
        
        Args:
            outlier_positions: [num_outliers] - row indices (within token)
            outlier_values: [num_outliers] - outlier values
        """
        
        # Check if current block is full
        if self.tokens_in_current_block >= self.block_size:
            self.current_block += 1
            self.tokens_in_current_block = 0
            self._allocate_block()
        
        # Append to current block
        col_idx = self.tokens_in_current_block
        self.columns[self.current_block][col_idx] = col_idx
        self.rows[self.current_block].append(outlier_positions)
        self.values[self.current_block].append(outlier_values)
        
        self.tokens_in_current_block += 1
    
    def to_csc(self):
        """Convert to standard CSC format for matrix operations"""
        # Concatenate all blocks
        all_columns = torch.cat([self.columns[i][:self.tokens_in_current_block] 
                                for i in range(self.current_block + 1)])
        all_rows = [r for block_rows in self.rows for r in block_rows]
        all_values = [v for block_values in self.values for v in block_values]
        
        return {
            'columns': all_columns,
            'rows': all_rows,
            'values': all_values
        }
```

**Memory savings analysis:**
- Without blocking: O(n²) copies when appending to size-n array
- With blocking: O(1) append, O(n) only when converting to CSC for operations

**Deliverables Week 2:**
- ✅ Deep understanding of KVQuant architecture
- ✅ Memory profiling identifying allocation bottlenecks
- ✅ Batched quantization algorithm designed
- ✅ Prototype batched quantization implemented
- ✅ Blocked sparse matrix allocation designed

---

## WEEK 3: Implement Batched Quantization

### Objectives
- [ ] Integrate batched quantization into KVQuant pipeline
- [ ] Handle variable-length sequences correctly
- [ ] Validate accuracy on simple examples
- [ ] Debug and optimize

### Tasks

#### Day 1-2: Integration with KVQuant

**Modify KVQuant's forward pass to use batched quantization during prefill:**

```python
# File: kvquant/prefill_quant.py

from kvquant.batched_quant import BatchedKVQuantizer
from kvquant.nuq import load_nuq_datatype
import torch

class PrefillQuantizedAttention:
    """
    Drop-in replacement for standard attention that quantizes KV cache during prefill
    """
    
    def __init__(self, config, layer_id):
        self.config = config
        self.layer_id = layer_id
        
        # Load pre-computed quantization parameters
        self.nuq_datatype = load_nuq_datatype(layer_id)
        self.scaling_factors = load_scaling_factors(layer_id)
        
        # Initialize quantizer
        self.quantizer = BatchedKVQuantizer(
            config=config,
            nuq_datatype=self.nuq_datatype,
            scaling_factors=self.scaling_factors
        )
        
        # Blocked sparse storage
        self.key_cache = BlockedSparseMatrix(block_size=256)
        self.value_cache = BlockedSparseMatrix(block_size=256)
        
    def forward(self, query, key, value, attention_mask=None, is_prefill=False):
        """
        Forward pass with quantized KV cache
        
        Args:
            query: [batch_size, num_heads, seq_len, head_dim]
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            attention_mask: [batch_size, seq_len]
            is_prefill: Whether this is prefill phase (True) or generation (False)
        
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
        """
        
        if is_prefill:
            # PREFILL PHASE: Quantize entire sequence at once
            seq_lengths = get_sequence_lengths(attention_mask)  # [batch_size]
            
            # Quantize Keys and Values in batched manner
            quantized_keys, sparse_keys = self.quantizer.quantize_keys_batch(
                key, seq_lengths
            )
            quantized_values, sparse_values = self.quantizer.quantize_values_batch(
                value, seq_lengths
            )
            
            # Store in blocked cache
            for b in range(len(seq_lengths)):
                for t in range(seq_lengths[b]):
                    self.key_cache.append_token(
                        sparse_keys[b][t]['positions'],
                        sparse_keys[b][t]['values']
                    )
                    self.value_cache.append_token(
                        sparse_values[b][t]['positions'],
                        sparse_values[b][t]['values']
                    )
            
            # Dequantize for attention computation
            dequantized_keys = self._dequantize_keys(quantized_keys, sparse_keys)
            dequantized_values = self._dequantize_values(quantized_values, sparse_values)
            
        else:
            # GENERATION PHASE: Use existing KVQuant logic (quantize one token)
            # ... (existing code)
            pass
        
        # Standard attention computation
        attention_output = self._compute_attention(
            query, dequantized_keys, dequantized_values, attention_mask
        )
        
        return attention_output
    
    def _dequantize_keys(self, quantized_keys, sparse_keys):
        """Dequantize Keys using nuq datatype lookup"""
        # For each quantized index, lookup value in nuq_datatype
        dequantized = []
        
        for batch_keys, batch_sparse in zip(quantized_keys, sparse_keys):
            # Lookup quantized indices in nuq_datatype
            dense_values = self.nuq_datatype[batch_keys]  # [seq_len, head_dim]
            
            # Add back outliers from sparse matrix
            for pos, val in zip(batch_sparse['positions'], batch_sparse['values']):
                dense_values[pos] = val
            
            # Rescale using pre-computed scaling factors
            dense_values = dense_values * self.scaling_factors
            
            dequantized.append(dense_values)
        
        return torch.stack(dequantized)
```

#### Day 3: Handle Variable-Length Sequences

**Challenge:** Different sequences in a batch have different lengths due to padding.

**Solution:** Use attention mask to identify actual sequence lengths, ignore padding.

```python
def get_sequence_lengths(attention_mask):
    """
    Extract actual sequence length for each sequence in batch
    
    Args:
        attention_mask: [batch_size, max_seq_len]
                       1 = real token, 0 = padding
    
    Returns:
        seq_lengths: [batch_size] - actual length of each sequence
    """
    # Sum across sequence dimension to get length
    seq_lengths = attention_mask.sum(dim=1)  # [batch_size]
    
    return seq_lengths.long()


def quantize_with_padding(keys, attention_mask, quantizer):
    """
    Quantize Keys while correctly handling padding
    
    Strategy:
        1. Extract actual sequence length from attention mask
        2. Only quantize real tokens (ignore padding)
        3. Store padding tokens as zeros (they're masked out in attention anyway)
    """
    batch_size, num_heads, max_seq_len, head_dim = keys.shape
    seq_lengths = get_sequence_lengths(attention_mask)
    
    quantized_keys = []
    sparse_keys = []
    
    for b in range(batch_size):
        seq_len = seq_lengths[b].item()
        
        # Extract real tokens only
        real_keys = keys[b, :, :seq_len, :]  # [num_heads, seq_len, head_dim]
        
        # Quantize
        quant, sparse = quantizer.quantize_keys_single(real_keys)
        
        # Pad back to max_seq_len (with zeros for quantized indices)
        padded_quant = torch.zeros(num_heads, max_seq_len, head_dim, dtype=quant.dtype)
        padded_quant[:, :seq_len, :] = quant
        
        quantized_keys.append(padded_quant)
        sparse_keys.append(sparse)
    
    return torch.stack(quantized_keys), sparse_keys
```

#### Day 4: Validate Accuracy

**Test 1: Quantization round-trip**
```python
def test_quantization_round_trip():
    """Verify quantization->dequantization recovers original values (with small error)"""
    
    # Generate random Keys
    keys = torch.randn(4, 8, 128, 64)  # batch=4, heads=8, seq=128, dim=64
    
    # Quantize
    quantized, sparse = quantizer.quantize_keys_batch(keys, seq_lengths=[128]*4)
    
    # Dequantize
    dequantized = dequantizer.dequantize_keys(quantized, sparse)
    
    # Compute error
    error = torch.abs(keys - dequantized).mean()
    
    print(f"Quantization error: {error:.6f}")
    assert error < 0.1, "Quantization error too large!"
```

**Test 2: Perplexity on single sample**
```python
def test_single_sample_perplexity():
    """Verify quantized model achieves similar perplexity on a single sample"""
    
    model_fp16 = load_model_fp16()
    model_quantized = load_model_quantized(prefill_quant=True)
    
    sample = load_wikitext2_sample(index=0)  # Single 2K token sample
    
    # Compute perplexity
    ppl_fp16 = compute_perplexity(model_fp16, sample)
    ppl_quant = compute_perplexity(model_quantized, sample)
    
    print(f"fp16 PPL: {ppl_fp16:.2f}")
    print(f"Quantized PPL: {ppl_quant:.2f}")
    print(f"Degradation: {ppl_quant - ppl_fp16:.2f}")
    
    assert abs(ppl_quant - ppl_fp16) < 0.5, "Perplexity degradation too large!"
```

**Run tests:**
```bash
python tests/test_batched_quantization.py
```

**Expected results:**
- Quantization error: <0.05 (3-bit quantization)
- Perplexity degradation: <0.2 PPL on single sample

#### Day 5: Debug and Optimize

**Common issues:**
1. **NaN in dequantization:** Check scaling factor computation
2. **High quantization error:** Verify nuq datatype is loaded correctly
3. **Memory leak:** Ensure sparse matrices are freed after use
4. **Slow performance:** Profile with `torch.profiler` to identify bottlenecks

**Optimization checklist:**
- [ ] Use `torch.jit.script` for hot paths
- [ ] Ensure all tensors are on GPU (no CPU-GPU transfers)
- [ ] Use in-place operations where possible
- [ ] Vectorize loops (replace Python loops with torch operations)

**Deliverables Week 3:**
- ✅ Batched quantization integrated into KVQuant
- ✅ Variable-length sequences handled correctly
- ✅ Accuracy validated on toy examples
- ✅ Quantization error <0.05, perplexity degradation <0.2 PPL

---

## WEEK 4-5: Implement Blocked Memory Allocation

### Objectives
- [ ] Implement blocked CSC/CSR sparse matrix structure
- [ ] Zero-copy append operations
- [ ] Benchmark memory allocation overhead
- [ ] Validate memory savings vs. baseline

### Tasks

#### Week 4, Day 1-2: Implement Blocked Sparse Matrix

**Complete implementation:**

```python
# File: kvquant/blocked_sparse.py

import torch
from typing import List, Tuple

class BlockedCSCMatrix:
    """
    Blocked Compressed Sparse Column (CSC) matrix for Keys
    
    Design:
        - Memory is pre-allocated in blocks of B tokens (e.g., B=256)
        - Each block stores CSC format: (column_ptr, row_indices, values)
        - Blocks are stored in a list (no reallocation when appending)
        - Conversion to standard CSC only when needed for computation
    
    Memory Layout:
        Block 0: Tokens 0-255
        Block 1: Tokens 256-511
        ...
        
    Advantages:
        - O(1) append (no reallocation)
        - Memory pre-allocated, no fragmentation
        - Can be converted to CSC for efficient matrix operations
    """
    
    def __init__(self, block_size=256, max_tokens=32768, head_dim=64):
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.head_dim = head_dim
        
        # Calculate number of blocks needed
        self.num_blocks = (max_tokens + block_size - 1) // block_size
        
        # Pre-allocate all blocks
        self.blocks = []
        for _ in range(self.num_blocks):
            self.blocks.append({
                'column_ptr': torch.zeros(block_size + 1, dtype=torch.int32),
                'row_indices': [],  # Will grow dynamically within block
                'values': []
            })
        
        self.current_block_idx = 0
        self.tokens_in_current_block = 0
        self.total_tokens = 0
    
    def append(self, outlier_row_indices: torch.Tensor, 
               outlier_values: torch.Tensor):
        """
        Append outliers for a single token (column in CSC format)
        
        Args:
            outlier_row_indices: [num_outliers] - row indices (0 to head_dim-1)
            outlier_values: [num_outliers] - fp16 outlier values
        """
        
        # Check if we need to move to next block
        if self.tokens_in_current_block >= self.block_size:
            self.current_block_idx += 1
            self.tokens_in_current_block = 0
            
            if self.current_block_idx >= self.num_blocks:
                raise ValueError(f"Exceeded max tokens {self.max_tokens}")
        
        # Get current block
        block = self.blocks[self.current_block_idx]
        col_idx = self.tokens_in_current_block
        
        # Update column pointer (marks start of this column's data)
        if col_idx == 0:
            block['column_ptr'][0] = 0
        else:
            block['column_ptr'][col_idx] = block['column_ptr'][col_idx - 1] + len(block['row_indices'][-1])
        
        # Append row indices and values for this token
        block['row_indices'].append(outlier_row_indices.cpu())
        block['values'].append(outlier_values.cpu())
        
        self.tokens_in_current_block += 1
        self.total_tokens += 1
    
    def batch_append(self, batch_outliers: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Append multiple tokens at once (for prefill)
        
        Args:
            batch_outliers: List of (row_indices, values) tuples, one per token
        """
        for row_indices, values in batch_outliers:
            self.append(row_indices, values)
    
    def to_standard_csc(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert to standard CSC format for matrix operations
        
        Returns:
            column_ptr: [total_tokens + 1]
            row_indices: [total_nonzeros]
            values: [total_nonzeros]
        """
        
        all_column_ptrs = []
        all_row_indices = []
        all_values = []
        
        current_offset = 0
        
        # Iterate through blocks that have data
        for block_idx in range(self.current_block_idx + 1):
            block = self.blocks[block_idx]
            
            # Determine how many tokens in this block
            if block_idx < self.current_block_idx:
                num_tokens_in_block = self.block_size
            else:
                num_tokens_in_block = self.tokens_in_current_block
            
            # Concatenate row indices and values
            block_rows = torch.cat(block['row_indices'][:num_tokens_in_block])
            block_vals = torch.cat(block['values'][:num_tokens_in_block])
            
            all_row_indices.append(block_rows)
            all_values.append(block_vals)
            
            # Adjust column pointers
            block_col_ptr = block['column_ptr'][:num_tokens_in_block + 1].clone()
            block_col_ptr += current_offset
            all_column_ptrs.append(block_col_ptr)
            
            current_offset = block_col_ptr[-1]
        
        # Concatenate all blocks
        column_ptr = torch.cat(all_column_ptrs)
        row_indices = torch.cat(all_row_indices)
        values = torch.cat(all_values)
        
        return column_ptr, row_indices, values
    
    def memory_footprint(self) -> dict:
        """Calculate memory usage in bytes"""
        
        # Column pointers: 4 bytes × (num_blocks × block_size)
        column_ptr_mem = self.num_blocks * (self.block_size + 1) * 4
        
        # Row indices + values: Count actual non-zeros
        total_nonzeros = 0
        for block_idx in range(self.current_block_idx + 1):
            block = self.blocks[block_idx]
            num_tokens = self.block_size if block_idx < self.current_block_idx else self.tokens_in_current_block
            for i in range(num_tokens):
                total_nonzeros += len(block['row_indices'][i])
        
        # Row indices: 4 bytes each, values: 2 bytes each (fp16)
        data_mem = total_nonzeros * (4 + 2)
        
        total_mem = column_ptr_mem + data_mem
        
        return {
            'column_ptr_bytes': column_ptr_mem,
            'data_bytes': data_mem,
            'total_bytes': total_mem,
            'total_mb': total_mem / (1024 ** 2),
            'total_nonzeros': total_nonzeros
        }


# Similarly implement BlockedCSRMatrix for Values
class BlockedCSRMatrix:
    """Blocked Compressed Sparse Row (CSR) matrix for Values"""
    # Similar structure, but row-major instead of column-major
    pass
```

#### Week 4, Day 3: Benchmark Memory Allocation

**Create benchmark comparing blocked vs. naive allocation:**

```python
# File: experiments/benchmark_memory_allocation.py

import torch
import time
import numpy as np
from kvquant.blocked_sparse import BlockedCSCMatrix

def naive_csc_append(column_ptr, row_indices, values, new_rows, new_values):
    """
    Naive approach: Reallocate and copy entire array on each append
    (This is what KVQuant currently does implicitly)
    """
    # This requires copying all existing data
    column_ptr = torch.cat([column_ptr, torch.tensor([column_ptr[-1] + len(new_rows)])])
    row_indices = torch.cat([row_indices, new_rows])
    values = torch.cat([values, new_values])
    
    return column_ptr, row_indices, values


def benchmark_memory_allocation(num_tokens=32768, sparsity=0.01, head_dim=64):
    """
    Compare naive vs. blocked allocation
    
    Args:
        num_tokens: Number of tokens to append
        sparsity: Percentage of outliers (e.g., 0.01 = 1%)
        head_dim: Dimension of head (64 for Llama-7B)
    """
    
    outliers_per_token = int(head_dim * sparsity)
    
    # Generate random outliers for each token
    all_outliers = []
    for _ in range(num_tokens):
        row_idx = torch.randperm(head_dim)[:outliers_per_token]
        vals = torch.randn(outliers_per_token, dtype=torch.float16)
        all_outliers.append((row_idx, vals))
    
    # Benchmark 1: Naive allocation (current KVQuant)
    print("Benchmarking naive allocation...")
    column_ptr = torch.zeros(1, dtype=torch.int32)
    row_indices = torch.tensor([], dtype=torch.int32)
    values = torch.tensor([], dtype=torch.float16)
    
    start = time.time()
    for row_idx, vals in all_outliers:
        column_ptr, row_indices, values = naive_csc_append(
            column_ptr, row_indices, values, row_idx, vals
        )
    naive_time = time.time() - start
    
    print(f"Naive allocation time: {naive_time:.3f} seconds")
    
    # Benchmark 2: Blocked allocation (our approach)
    print("Benchmarking blocked allocation...")
    blocked_csc = BlockedCSCMatrix(block_size=256, max_tokens=num_tokens, head_dim=head_dim)
    
    start = time.time()
    for row_idx, vals in all_outliers:
        blocked_csc.append(row_idx, vals)
    blocked_time = time.time() - start
    
    print(f"Blocked allocation time: {blocked_time:.3f} seconds")
    
    # Speedup
    speedup = naive_time / blocked_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Memory footprint
    mem_stats = blocked_csc.memory_footprint()
    print(f"\nMemory footprint: {mem_stats['total_mb']:.2f} MB")
    print(f"Total nonzeros: {mem_stats['total_nonzeros']}")
    
    # Verify correctness: Convert to standard CSC and compare
    print("\nVerifying correctness...")
    column_ptr_blocked, row_indices_blocked, values_blocked = blocked_csc.to_standard_csc()
    
    assert len(column_ptr) == len(column_ptr_blocked), "Column pointer mismatch!"
    assert len(row_indices) == len(row_indices_blocked), "Row indices mismatch!"
    print("✓ Correctness verified")
    
    return {
        'naive_time': naive_time,
        'blocked_time': blocked_time,
        'speedup': speedup,
        'memory_mb': mem_stats['total_mb']
    }


if __name__ == "__main__":
    # Benchmark for different sequence lengths
    results = {}
    for num_tokens in [1024, 2048, 8192, 32768]:
        print(f"\n{'='*60}")
        print(f"Benchmarking {num_tokens} tokens")
        print(f"{'='*60}")
        
        results[num_tokens] = benchmark_memory_allocation(
            num_tokens=num_tokens,
            sparsity=0.01,
            head_dim=64
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for num_tokens, stats in results.items():
        print(f"{num_tokens:5d} tokens: {stats['speedup']:.2f}x speedup, {stats['memory_mb']:.1f} MB")
```

**Run benchmark:**
```bash
python experiments/benchmark_memory_allocation.py
```

**Expected results:**
```
1024  tokens: 1.2x speedup, 0.8 MB
2048  tokens: 2.5x speedup, 1.6 MB
8192  tokens: 8.1x speedup, 6.4 MB
32768 tokens: 25.3x speedup, 25.6 MB
```

Speedup increases with sequence length because naive reallocation has O(n²) complexity.

#### Week 4, Day 4-5: Integrate Blocked Allocation into Prefill Pipeline

**Modify PrefillQuantizedAttention to use blocked storage:**

```python
# File: kvquant/prefill_quant.py (updated)

class PrefillQuantizedAttention:
    """Updated to use blocked allocation"""
    
    def __init__(self, config, layer_id):
        # ... (previous init code)
        
        # Replace naive storage with blocked storage
        self.key_cache_blocked = BlockedCSCMatrix(
            block_size=256,
            max_tokens=config.max_seq_len,
            head_dim=config.head_dim
        )
        self.value_cache_blocked = BlockedCSRMatrix(
            block_size=256,
            max_tokens=config.max_seq_len,
            head_dim=config.head_dim
        )
    
    def forward(self, query, key, value, attention_mask=None, is_prefill=False):
        if is_prefill:
            # Quantize batch
            quantized_keys, sparse_keys = self.quantizer.quantize_keys_batch(
                key, get_sequence_lengths(attention_mask)
            )
            quantized_values, sparse_values = self.quantizer.quantize_values_batch(
                value, get_sequence_lengths(attention_mask)
            )
            
            # Append to blocked cache (zero-copy)
            for batch_idx in range(len(sparse_keys)):
                for token_outliers in sparse_keys[batch_idx]:
                    self.key_cache_blocked.append(
                        token_outliers['positions'],
                        token_outliers['values']
                    )
            
            # Similarly for values
            for batch_idx in range(len(sparse_values)):
                for token_outliers in sparse_values[batch_idx]:
                    self.value_cache_blocked.append(
                        token_outliers['positions'],
                        token_outliers['values']
                    )
            
            # Dequantize for attention (convert blocked -> standard CSC/CSR)
            column_ptr, row_indices, key_outliers = self.key_cache_blocked.to_standard_csc()
            dequantized_keys = self._dequantize_with_sparse(
                quantized_keys, column_ptr, row_indices, key_outliers
            )
            
            # ... (attention computation)
```

#### Week 5: Validation and Memory Profiling

**Tasks:**
- Run full prefill with blocked allocation
- Profile memory usage with PyTorch profiler
- Validate: No memory leak, no performance regression
- Compare memory footprint: Blocked vs. naive

**Deliverables Week 4-5:**
- ✅ Blocked CSC/CSR implementation complete
- ✅ Zero-copy append operations validated
- ✅ Memory allocation speedup: 10-25× for long sequences
- ✅ Integration with prefill pipeline complete
- ✅ Memory savings: 40-60% reduction in allocation overhead

---

## WEEK 6: Fused Kernels

### Objectives
- [ ] Implement fused attention-quantization kernel using Triton
- [ ] Optimize memory bandwidth utilization
- [ ] Validate numerical correctness
- [ ] Benchmark performance

### Tasks

#### Day 1-2: Design Fused Kernel

**Challenge:** Current pipeline has multiple passes:
1. Forward pass → compute Keys/Values
2. Quantize Keys/Values (separate kernel)
3. Append to sparse cache (another kernel)
4. Dequantize for attention (yet another kernel)

**Solution:** Fuse steps 1-3 into single kernel to reduce memory bandwidth.

**Triton kernel pseudocode:**
```python
# File: kvquant/kernels/fused_prefill_quant.py

import triton
import triton.language as tl

@triton.jit
def fused_prefill_quantize_kernel(
    # Inputs
    keys_ptr, values_ptr,  # [batch, heads, seq_len, head_dim]
    attention_mask_ptr,    # [batch, seq_len]
    nuq_datatype_ptr,      # [2^k] lookup table
    scaling_factors_ptr,   # [heads, head_dim]
    
    # Outputs
    quantized_keys_ptr,    # [batch, heads, seq_len, head_dim] - 3-bit indices
    sparse_keys_col_ptr,   # CSC format
    sparse_keys_row_ptr,
    sparse_keys_val_ptr,
    
    # Sizes
    batch_size, num_heads, seq_len, head_dim,
    
    # Strides
    stride_kb, stride_kh, stride_ks, stride_kd,
    
    # Config
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr
):
    """
    Fused kernel: Load Keys → Quantize → Store quantized + outliers
    
    Strategy:
        - Load tile of Keys (e.g., 32×64)
        - Normalize using scaling factors
        - Detect outliers (top 1%)
        - Quantize non-outliers using nuq lookup
        - Store quantized indices
        - Store outliers in sparse format
    
    Memory bandwidth saved:
        - No intermediate fp16 cache materialized
        - Direct load → quantize → store
    """
    
    # Program ID (which tile to process)
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    # Load attention mask to get actual sequence length
    mask_offset = pid_batch * seq_len + tl.arange(0, BLOCK_SIZE)
    mask = tl.load(attention_mask_ptr + mask_offset)
    
    # Load Keys for this tile
    key_offset = (pid_batch * stride_kb + 
                 pid_head * stride_kh + 
                 pid_seq * stride_ks * BLOCK_SIZE)
    
    key_ptrs = keys_ptr + key_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_ks + tl.arange(0, head_dim)[None, :]
    keys = tl.load(key_ptrs, mask=mask[:, None])
    
    # Load scaling factors for this head
    scale_ptrs = scaling_factors_ptr + pid_head * head_dim + tl.arange(0, head_dim)
    scales = tl.load(scale_ptrs)
    
    # Normalize
    normalized = keys / scales[None, :]
    
    # Detect outliers (per-channel)
    for ch in range(head_dim):
        channel_vals = normalized[:, ch]
        
        # Compute 99th percentile threshold
        # (Triton doesn't have percentile, use approximation)
        threshold = tl.max(tl.abs(channel_vals)) * 0.99
        
        outlier_mask = tl.abs(channel_vals) > threshold
        
        # Quantize non-outliers
        non_outliers = tl.where(outlier_mask, 0.0, channel_vals)  # Zero out outliers
        
        # Find closest nuq datatype entry (broadcast and argmin)
        # ... (lookup in nuq_datatype)
        quantized_indices = nuq_lookup(non_outliers, nuq_datatype_ptr)
        
        # Store quantized indices
        quant_offset = (pid_batch * num_heads * seq_len * head_dim +
                       pid_head * seq_len * head_dim +
                       pid_seq * BLOCK_SIZE * head_dim +
                       ch)
        quant_ptrs = quantized_keys_ptr + quant_offset + tl.arange(0, BLOCK_SIZE) * head_dim
        tl.store(quant_ptrs, quantized_indices)
        
        # Store outliers in sparse format (CSC)
        # Count outliers
        num_outliers = tl.sum(outlier_mask.to(tl.int32))
        
        if num_outliers > 0:
            # Extract outlier positions and values
            outlier_positions = tl.where(outlier_mask)  # Indices
            outlier_values = keys[:, ch][outlier_mask]  # Original fp16 values
            
            # Atomic append to sparse matrix
            # ... (complex, may need host-side coordination)


# Wrapper function
def fused_prefill_quantize(keys, values, attention_mask, 
                          nuq_datatype, scaling_factors):
    """
    Fuse prefill quantization into single kernel call
    
    Returns:
        quantized_keys, quantized_values, sparse_keys, sparse_values
    """
    
    batch_size, num_heads, seq_len, head_dim = keys.shape
    
    # Allocate outputs
    quantized_keys = torch.zeros(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=torch.uint8,  # 3-bit packed into uint8
        device='cuda'
    )
    
    # Sparse storage (allocated on host, filled by kernel)
    # ... (may need multiple kernel launches)
    
    # Launch kernel
    grid = (batch_size, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    fused_prefill_quantize_kernel[grid](
        keys, values, attention_mask,
        nuq_datatype, scaling_factors,
        quantized_keys, sparse_keys_col, sparse_keys_row, sparse_keys_val,
        batch_size, num_heads, seq_len, head_dim,
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3),
        BLOCK_SIZE=32, NUM_WARPS=4
    )
    
    return quantized_keys, sparse_keys
```

**Note:** Full Triton kernel implementation is complex. For this project, we can:
1. **Option A:** Implement simplified version (no fusion, just Triton attention)
2. **Option B:** Use PyTorch's built-in optimizations + manual loop fusion
3. **Option C:** Focus on blocked allocation gains, defer kernel fusion to future work

**Recommendation for Week 6:** Implement Option B (manual loop fusion in PyTorch) to avoid over-spending time on low-level CUDA optimization.

#### Day 3-4: Implement PyTorch Loop Fusion

**Simpler approach: Fuse in PyTorch without custom Triton kernel:**

```python
# File: kvquant/prefill_quant_fused.py

def fused_prefill_quantize_pytorch(keys, values, attention_mask,
                                   quantizer, key_cache, value_cache):
    """
    Fused prefill quantization in PyTorch (not custom CUDA)
    
    Strategy:
        - Single pass through Keys/Values
        - Quantize and append to cache in one loop
        - Minimize temporary allocations
    """
    
    batch_size, num_heads, seq_len, head_dim = keys.shape
    
    # Extract sequence lengths
    seq_lengths = attention_mask.sum(dim=1)  # [batch]
    
    # Process in batches to reduce memory
    for b in range(batch_size):
        actual_len = seq_lengths[b].item()
        
        # Extract this sequence's Keys/Values
        seq_keys = keys[b, :, :actual_len, :]  # [heads, actual_len, head_dim]
        seq_values = values[b, :, :actual_len, :]
        
        # Quantize (vectorized over all heads and tokens at once)
        with torch.no_grad():  # No gradients needed
            # Normalize
            normalized_keys = seq_keys / quantizer.scaling_factors.unsqueeze(1)
            
            # Outlier detection (vectorized)
            outlier_threshold = torch.quantile(
                torch.abs(normalized_keys), 0.99, dim=1, keepdim=True
            )  # [heads, 1, head_dim]
            
            outlier_mask = torch.abs(normalized_keys) > outlier_threshold
            
            # Quantize non-outliers
            non_outliers = normalized_keys.clone()
            non_outliers[outlier_mask] = 0.0  # Mask out outliers
            
            # Lookup in nuq datatype (vectorized)
            quantized_indices = quantizer.quantize_vectorized(non_outliers)
            
            # Extract outliers for sparse storage
            outlier_positions = torch.nonzero(outlier_mask, as_tuple=True)
            outlier_values = seq_keys[outlier_positions]
            
            # Append to cache (blocked, zero-copy)
            for token_idx in range(actual_len):
                # Get outliers for this token
                token_outlier_mask = outlier_mask[:, token_idx, :]  # [heads, head_dim]
                token_outliers = seq_keys[:, token_idx, :][token_outlier_mask]
                token_outlier_pos = torch.nonzero(token_outlier_mask, as_tuple=True)[1]
                
                # Append to blocked cache
                key_cache.append(token_outlier_pos, token_outliers)
        
        # Similarly for Values
        # ...
    
    return quantized_indices
```

**This approach:**
- ✅ Reduces memory allocations (single pass)
- ✅ Leverages PyTorch's optimized vectorization
- ✅ Much simpler to implement and debug than custom CUDA
- ⚠️ Not as fast as hand-written Triton kernel, but still 1.5-2× faster than naive

#### Day 5: Benchmark Fused Implementation

**Compare:**
1. Naive (multiple passes): Load keys → quantize → append
2. Fused (single pass): Load+quantize+append in one loop

**Expected improvement:** 1.5-2× reduction in memory bandwidth, 1.2-1.5× latency improvement

**Deliverables Week 6:**
- ✅ Fused prefill quantization implemented in PyTorch
- ✅ Memory bandwidth reduced by 40-50%
- ✅ Prefill latency improved by 20-30%

---

## WEEK 7: Accuracy Validation

### Objectives
- [ ] Evaluate perplexity on LongBench (6 tasks)
- [ ] Evaluate on RULER (13 tasks)
- [ ] Evaluate on Passkey Retrieval
- [ ] Statistical significance testing

### Tasks

#### Day 1-2: LongBench Evaluation

**Download and setup LongBench:**
```bash
cd $PLG_GROUPS_STORAGE/kvquant-prefill/datasets
git clone https://github.com/THUDM/LongBench.git
cd LongBench

# Install requirements
pip install --break-system-packages -r requirements.txt
```

**LongBench tasks to evaluate:**
1. **NarrativeQA:** Long narrative comprehension
2. **Qasper:** Question answering on scientific papers
3. **MultiFieldQA-en:** Multi-hop reasoning
4. **HotpotQA:** Wikipedia-based QA
5. **2WikiMultihopQA:** Multi-hop reasoning
6. **Musique:** Complex reasoning

**Create evaluation script:**
```python
# File: experiments/evaluate_longbench.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from longbench import LongBenchDataset, evaluate_model
from kvquant.prefill_quant import load_quantized_model
import json

def evaluate_longbench(model, tokenizer, tasks, max_seq_len=32768):
    """Evaluate model on LongBench tasks"""
    
    results = {}
    
    for task in tasks:
        print(f"\nEvaluating task: {task}")
        
        # Load dataset
        dataset = LongBenchDataset(task, max_length=max_seq_len)
        
        # Evaluate
        task_results = evaluate_model(
            model, tokenizer, dataset,
            batch_size=4,  # Small batch due to long context
            max_new_tokens=512
        )
        
        results[task] = task_results
        print(f"  Accuracy: {task_results['accuracy']:.2f}%")
        print(f"  F1: {task_results['f1']:.2f}")
    
    return results


# Main evaluation
if __name__ == "__main__":
    # Load models
    print("Loading fp16 baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        "../models/Llama-2-7B-32K",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-7B-32K")
    
    print("Loading quantized model with prefill quantization...")
    model_quant = load_quantized_model(
        "../models/Llama-2-7B-32K",
        quantization="nuq3",
        sparsity=0.01,
        prefill_quant=True  # Enable prefill quantization
    )
    
    # Tasks to evaluate
    tasks = [
        "narrativeqa", "qasper", "multifieldqa_en",
        "hotpotqa", "2wikimqa", "musique"
    ]
    
    # Evaluate both models
    print("\n" + "="*60)
    print("Evaluating fp16 baseline")
    print("="*60)
    results_fp16 = evaluate_longbench(model_fp16, tokenizer, tasks)
    
    print("\n" + "="*60)
    print("Evaluating quantized model (prefill quant)")
    print("="*60)
    results_quant = evaluate_longbench(model_quant, tokenizer, tasks)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    for task in tasks:
        acc_fp16 = results_fp16[task]['accuracy']
        acc_quant = results_quant[task]['accuracy']
        degradation = acc_quant - acc_fp16
        
        print(f"{task:20s}: fp16={acc_fp16:.2f}%, quant={acc_quant:.2f}%, "
              f"Δ={degradation:+.2f}%")
    
    # Save results
    with open('longbench_results.json', 'w') as f:
        json.dump({
            'fp16': results_fp16,
            'quantized': results_quant
        }, f, indent=2)
```

**Submit LongBench evaluation job:**
```bash
# File: slurm_jobs/eval_longbench.sh

#!/bin/bash
#SBATCH --job-name=longbench_eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=12:00:00
#SBATCH --mem=1024GB

module load cuda/12.4.0
module load python/3.10

cd $PLG_GROUPS_STORAGE/kvquant-prefill
source venv/bin/activate

python experiments/evaluate_longbench.py
```

**Expected results (target):**
```
Task                 : fp16      quant    Degradation
narrativeqa          : 23.5%     23.3%    -0.2%
qasper               : 31.2%     31.0%    -0.2%
multifieldqa_en      : 45.1%     44.9%    -0.2%
hotpotqa             : 38.7%     38.5%    -0.2%
2wikimqa             : 35.4%     35.2%    -0.2%
musique              : 22.8%     22.6%    -0.2%

Average degradation: -0.2% (well within <0.5% target)
```

#### Day 3: RULER Evaluation

**RULER:** Synthetic benchmark for testing long-context understanding (needle-in-haystack, multi-hop reasoning, etc.)

**Download:**
```bash
cd $PLG_GROUPS_STORAGE/kvquant-prefill/datasets
git clone https://github.com/hsiehjackson/RULER.git
```

**13 RULER tasks:**
1-3. Needle-in-haystack (1, 2, 3 needles)
4-6. Multi-key lookup (1, 2, 3 keys)
7. Multi-value retrieval
8. Multi-query reasoning
9-10. Variable tracking (forward, backward)
11-12. Common words extraction (frequency, QA)
13. QA (multiple hops)

**Evaluation script:**
```python
# File: experiments/evaluate_ruler.py

from ruler import RULERBenchmark, evaluate_model

# Similar structure to LongBench evaluation
# ...
```

**Expected results:**
- Accuracy degradation <1% across all 13 tasks
- Some tasks (needle-in-haystack) should show <0.1% degradation

#### Day 4: Passkey Retrieval

**Passkey Retrieval:** Test model's ability to retrieve specific information from very long contexts.

**Create evaluation script:**
```python
# File: experiments/evaluate_passkey.py

import torch
import random

def generate_passkey_sample(context_length=32000, passkey_length=5):
    """
    Generate synthetic passkey retrieval sample
    
    Format:
        "The grass is green. The sky is blue. ... [PASSKEY: 12345] ... The sun is hot."
        "What is the passkey? Answer: "
    """
    
    # Random position for passkey (not too early, not too late)
    passkey_pos = random.randint(context_length // 4, 3 * context_length // 4)
    
    # Generate random passkey
    passkey = ''.join([str(random.randint(0, 9)) for _ in range(passkey_length)])
    
    # Generate filler text
    filler_sentences = [
        "The grass is green.",
        "The sky is blue.",
        "The sun is hot.",
        "The moon is bright.",
        "Water is wet."
    ]
    
    # Build context
    context_tokens = []
    for i in range(context_length):
        if i == passkey_pos:
            context_tokens.append(f" [PASSKEY: {passkey}] ")
        else:
            context_tokens.append(random.choice(filler_sentences))
    
    context = ' '.join(context_tokens)
    query = "What is the passkey? Answer: "
    
    return context + query, passkey


def evaluate_passkey_retrieval(model, tokenizer, 
                               context_lengths=[2000, 4000, 8000, 16000, 32000],
                               num_samples=50):
    """Evaluate passkey retrieval accuracy"""
    
    results = {}
    
    for context_len in context_lengths:
        print(f"\nEvaluating context length: {context_len}")
        
        correct = 0
        total = 0
        
        for i in range(num_samples):
            # Generate sample
            text, true_passkey = generate_passkey_sample(context_len)
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            # Decode
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
            
            # Check if passkey is in generated text
            if true_passkey in generated:
                correct += 1
            
            total += 1
        
        accuracy = 100.0 * correct / total
        results[context_len] = accuracy
        
        print(f"  Accuracy: {accuracy:.1f}%")
    
    return results


# Run evaluation
if __name__ == "__main__":
    # Load models
    model_fp16 = load_model_fp16()
    model_quant = load_model_quantized(prefill_quant=True)
    
    print("Evaluating fp16 baseline...")
    results_fp16 = evaluate_passkey_retrieval(model_fp16, tokenizer)
    
    print("\nEvaluating quantized model...")
    results_quant = evaluate_passkey_retrieval(model_quant, tokenizer)
    
    # Compare
    print("\nCOMPARISON")
    print("="*60)
    for context_len in [2000, 4000, 8000, 16000, 32000]:
        acc_fp16 = results_fp16[context_len]
        acc_quant = results_quant[context_len]
        print(f"{context_len:5d} tokens: fp16={acc_fp16:.1f}%, "
              f"quant={acc_quant:.1f}%, Δ={acc_quant - acc_fp16:+.1f}%")
```

**Expected results:**
```
Context Length  : fp16     quant    Degradation
2000 tokens     : 100.0%   100.0%   0.0%
4000 tokens     : 100.0%   99.8%    -0.2%
8000 tokens     : 100.0%   99.6%    -0.4%
16000 tokens    : 99.6%    99.2%    -0.4%
32000 tokens    : 98.8%    98.4%    -0.4%
```

#### Day 5: Statistical Significance Testing

**Run multiple seeds for each evaluation:**
```python
# File: experiments/statistical_tests.py

from scipy import stats
import numpy as np

def paired_t_test(results_baseline, results_method, alpha=0.05):
    """
    Perform paired t-test to check if degradation is statistically significant
    
    H0: Mean degradation = 0
    H1: Mean degradation ≠ 0
    """
    
    degradations = []
    for task in results_baseline.keys():
        baseline_acc = results_baseline[task]['accuracy']
        method_acc = results_method[task]['accuracy']
        degradations.append(method_acc - baseline_acc)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(degradations, 0.0)
    
    print(f"Mean degradation: {np.mean(degradations):.3f}%")
    print(f"Std degradation: {np.std(degradations):.3f}%")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"Result: Degradation is statistically significant (p < {alpha})")
    else:
        print(f"Result: Degradation is NOT statistically significant (p >= {alpha})")
    
    return {
        'mean_degradation': np.mean(degradations),
        'std_degradation': np.std(degradations),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }


# Run with 5 different random seeds
results_baseline_seeds = []
results_quant_seeds = []

for seed in [42, 123, 456, 789, 101112]:
    set_seed(seed)
    results_baseline_seeds.append(evaluate_longbench(model_fp16, tokenizer, tasks))
    results_quant_seeds.append(evaluate_longbench(model_quant, tokenizer, tasks))

# Statistical test
stats_results = paired_t_test(
    aggregate_results(results_baseline_seeds),
    aggregate_results(results_quant_seeds)
)
```

**Expected outcome:**
- Mean degradation: -0.2% ± 0.1%
- p-value: ~0.15 (NOT statistically significant)
- **Conclusion:** Prefill quantization maintains accuracy within statistical noise

**Deliverables Week 7:**
- ✅ LongBench evaluation complete (6 tasks, <0.5% degradation)
- ✅ RULER evaluation complete (13 tasks, <1% degradation)
- ✅ Passkey retrieval complete (<0.5% degradation)
- ✅ Statistical tests show no significant degradation (p > 0.05)

---

## WEEK 8: Throughput Benchmarking

### Objectives
- [ ] Systematic throughput measurement across batch sizes and prompt lengths
- [ ] Compare fp16 vs. quantized (prefill quant)
- [ ] Analyze memory usage and batch size scaling
- [ ] Validate 2-3× throughput improvement claim

### Tasks

#### Day 1-2: Design Throughput Benchmark

**Variables to sweep:**
- **Prompt lengths:** 512, 2048, 8192, 16384, 32768
- **Batch sizes:** 1, 4, 16, 32, 64 (if memory permits)
- **Quantization:** fp16, nuq3-1%, nuq4-1%

**Metrics:**
- **Throughput:** Tokens/second (prompt length × batch size / latency)
- **Latency:** Seconds to complete prefill
- **Memory:** Peak GPU memory (GB)

**Benchmark script:**
```python
# File: experiments/benchmark_throughput.py

import torch
import time
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

pynvml.nvmlInit()

def measure_throughput(model, tokenizer, prompt_length, batch_size):
    """
    Measure prefill throughput
    
    Returns:
        - latency_sec: Time to complete prefill
        - throughput_tokens_per_sec: Tokens/second
        - peak_memory_gb: Peak GPU memory usage
    """
    
    # Generate dummy prompt
    prompt = "Hello world " * (prompt_length // 2)
    prompts = [prompt] * batch_size
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=prompt_length,
        truncation=True
    ).to("cuda")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Measure
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize()
    
    end = time.time()
    
    latency = end - start
    total_tokens = prompt_length * batch_size
    throughput = total_tokens / latency
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        'latency_sec': latency,
        'throughput_tokens_per_sec': throughput,
        'peak_memory_gb': peak_memory
    }


def run_throughput_benchmark(model_name, quantization=None):
    """
    Run full throughput benchmark across all configurations
    """
    
    # Load model
    if quantization is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = load_quantized_model(
            model_name,
            quantization=quantization,
            prefill_quant=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configurations
    prompt_lengths = [512, 2048, 8192, 16384, 32768]
    batch_sizes = [1, 4, 16, 32]
    
    results = {}
    
    for prompt_len in prompt_lengths:
        for batch_size in batch_sizes:
            config_key = f"len{prompt_len}_batch{batch_size}"
            
            print(f"Benchmarking {config_key}...")
            
            try:
                result = measure_throughput(
                    model, tokenizer, prompt_len, batch_size
                )
                
                results[config_key] = result
                
                print(f"  Throughput: {result['throughput_tokens_per_sec']:.1f} tok/s")
                print(f"  Latency: {result['latency_sec']:.3f} sec")
                print(f"  Peak memory: {result['peak_memory_gb']:.2f} GB")
                
            except RuntimeError as e:
                print(f"  FAILED (OOM): {str(e)}")
                results[config_key] = {'error': 'OOM'}
    
    return results


if __name__ == "__main__":
    # Benchmark fp16 baseline
    print("="*60)
    print("Benchmarking fp16 baseline")
    print("="*60)
    
    results_fp16 = run_throughput_benchmark(
        "../models/Llama-2-7B-32K",
        quantization=None
    )
    
    # Save intermediate results
    with open('throughput_fp16.json', 'w') as f:
        json.dump(results_fp16, f, indent=2)
    
    # Benchmark quantized (nuq3-1%)
    print("\n" + "="*60)
    print("Benchmarking quantized (nuq3-1%, prefill quant)")
    print("="*60)
    
    results_quant = run_throughput_benchmark(
        "../models/Llama-2-7B-32K",
        quantization="nuq3-1%"
    )
    
    # Save results
    with open('throughput_quant.json', 'w') as f:
        json.dump(results_quant, f, indent=2)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    for config_key in results_fp16.keys():
        if 'error' in results_fp16[config_key] or 'error' in results_quant[config_key]:
            continue
        
        throughput_fp16 = results_fp16[config_key]['throughput_tokens_per_sec']
        throughput_quant = results_quant[config_key]['throughput_tokens_per_sec']
        speedup = throughput_quant / throughput_fp16
        
        mem_fp16 = results_fp16[config_key]['peak_memory_gb']
        mem_quant = results_quant[config_key]['peak_memory_gb']
        mem_reduction = (mem_fp16 - mem_quant) / mem_fp16 * 100
        
        print(f"{config_key:25s}: {speedup:.2f}x speedup, {mem_reduction:.1f}% mem reduction")
```

#### Day 3: Run Throughput Benchmarks

**Submit Slurm job:**
```bash
#!/bin/bash
#SBATCH --job-name=throughput_benchmark
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=24:00:00
#SBATCH --mem=1024GB

cd $PLG_GROUPS_STORAGE/kvquant-prefill
source venv/bin/activate

python experiments/benchmark_throughput.py
```

**Expected results (target):**
```
Configuration             Speedup    Memory Reduction
len512_batch1             1.1x       10%
len512_batch4             1.3x       15%
len512_batch16            1.5x       25%
len2048_batch1            1.2x       15%
len2048_batch4            1.6x       30%
len2048_batch16           2.1x       45%
len8192_batch1            1.3x       20%
len8192_batch4            1.9x       40%
len8192_batch16           2.5x       60%   ← Key result
len16384_batch4           2.2x       50%
len16384_batch16          2.8x       70%
len32768_batch4           2.4x       60%
len32768_batch16          OOM (fp16), 2.9x (quant)  ← Key enablement

Key findings:
- Speedup increases with prompt length (more KV cache to compress)
- Memory reduction enables larger batch sizes (fp16 OOMs, quant succeeds)
- 2-3× throughput improvement for long prompts (16K+) with large batches
```

#### Day 4: Analyze Batch Size Scaling

**Create scaling analysis:**
```python
# File: experiments/analyze_scaling.py

import json
import matplotlib.pyplot as plt

def plot_throughput_scaling():
    """Plot how throughput scales with batch size"""
    
    # Load results
    with open('throughput_fp16.json') as f:
        results_fp16 = json.load(f)
    with open('throughput_quant.json') as f:
        results_quant = json.load(f)
    
    # Extract data for 16K prompt length
    batch_sizes = [1, 4, 16, 32]
    throughput_fp16_16k = []
    throughput_quant_16k = []
    
    for bs in batch_sizes:
        key = f"len16384_batch{bs}"
        
        if key in results_fp16 and 'error' not in results_fp16[key]:
            throughput_fp16_16k.append(results_fp16[key]['throughput_tokens_per_sec'])
        else:
            throughput_fp16_16k.append(0)  # OOM
        
        if key in results_quant and 'error' not in results_quant[key]:
            throughput_quant_16k.append(results_quant[key]['throughput_tokens_per_sec'])
        else:
            throughput_quant_16k.append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(batch_sizes, throughput_fp16_16k, 
           marker='o', label='fp16 baseline', linewidth=2)
    ax.plot(batch_sizes, throughput_quant_16k, 
           marker='s', label='Quantized (nuq3-1%)', linewidth=2)
    
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=14)
    ax.set_title('Throughput Scaling (16K prompt length)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('throughput_scaling.png', dpi=300)
    print("Saved: throughput_scaling.png")


def plot_memory_usage():
    """Plot memory usage vs. batch size"""
    # Similar analysis for peak memory
    pass


if __name__ == "__main__":
    plot_throughput_scaling()
    plot_memory_usage()
```

#### Day 5: Generate Figures for Paper

**Create publication-quality figures:**
1. **Figure 1:** Throughput vs. batch size (multiple prompt lengths)
2. **Figure 2:** Memory usage vs. batch size
3. **Figure 3:** Speedup heatmap (prompt length × batch size)

**Deliverables Week 8:**
- ✅ Systematic throughput benchmark complete
- ✅ 2-3× throughput improvement for long prompts + large batches
- ✅ Memory reduction enables 2× larger batch sizes
- ✅ Publication-quality figures generated

---

## WEEK 9: Energy Profiling

### Objectives
- [ ] Measure energy per token with NVML
- [ ] Compare fp16 vs. quantized
- [ ] Validate: Energy savings proportional to throughput improvement
- [ ] Calculate carbon footprint reduction

### Tasks

#### Day 1-2: Energy Measurement Infrastructure

**Extend energy monitor from Week 1:**
```python
# File: utils/energy_profiler.py

import pynvml
import time
import csv
from codecarbon import EmissionsTracker

class DetailedEnergyProfiler:
    """
    Detailed energy profiling with per-operation breakdown
    """
    
    def __init__(self, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]):
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids]
        self.gpu_ids = gpu_ids
        
        # CodeCarbon for carbon emissions
        self.carbon_tracker = EmissionsTracker()
        
        # Log file
        self.log_file = open('energy_log.csv', 'w')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp', 'operation', 'duration_sec', 
            'energy_joules', 'avg_power_watts', 'carbon_kg'
        ])
    
    def start_operation(self, operation_name):
        """Start measuring energy for an operation"""
        self.operation_name = operation_name
        self.start_time = time.time()
        
        # Read initial energy from all GPUs
        self.start_energy = sum([
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) 
            for handle in self.handles
        ]) / 1000.0  # Convert mJ to J
        
        self.carbon_tracker.start()
    
    def end_operation(self):
        """End measuring and log results"""
        self.end_time = time.time()
        
        # Read final energy
        self.end_energy = sum([
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            for handle in self.handles
        ]) / 1000.0
        
        self.carbon_tracker.stop()
        
        # Calculate metrics
        duration = self.end_time - self.start_time
        energy = self.end_energy - self.start_energy
        avg_power = energy / duration if duration > 0 else 0
        carbon = self.carbon_tracker.final_emissions
        
        # Log
        self.csv_writer.writerow([
            self.start_time, self.operation_name, duration,
            energy, avg_power, carbon
        ])
        self.log_file.flush()
        
        return {
            'operation': self.operation_name,
            'duration_sec': duration,
            'energy_joules': energy,
            'avg_power_watts': avg_power,
            'carbon_emissions_kg': carbon
        }
    
    def __del__(self):
        self.log_file.close()


def profile_energy_per_token(model, tokenizer, prompt_lengths, batch_sizes):
    """
    Measure energy per token for different configurations
    """
    
    profiler = DetailedEnergyProfiler()
    results = {}
    
    for prompt_len in prompt_lengths:
        for batch_size in batch_sizes:
            config_key = f"len{prompt_len}_batch{batch_size}"
            
            print(f"Profiling energy: {config_key}")
            
            # Generate prompts
            prompt = "Hello " * (prompt_len // 2)
            inputs = tokenizer(
                [prompt] * batch_size,
                return_tensors="pt",
                padding="max_length",
                max_length=prompt_len,
                truncation=True
            ).to("cuda")
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(**inputs)
            
            # Measure
            profiler.start_operation(config_key)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            torch.cuda.synchronize()
            stats = profiler.end_operation()
            
            # Calculate per-token energy
            total_tokens = prompt_len * batch_size
            energy_per_token = stats['energy_joules'] / total_tokens
            
            results[config_key] = {
                **stats,
                'total_tokens': total_tokens,
                'energy_per_token_joules': energy_per_token
            }
            
            print(f"  Energy: {stats['energy_joules']:.2f} J")
            print(f"  Energy/token: {energy_per_token:.4f} J/token")
    
    return results
```

#### Day 3: Run Energy Profiling

**Profile both fp16 and quantized:**
```python
# File: experiments/profile_energy.py

import torch
from utils.energy_profiler import profile_energy_per_token

# Load models
model_fp16 = load_model_fp16("../models/Llama-2-7B-32K")
model_quant = load_quantized_model("../models/Llama-2-7B-32K", "nuq3-1%", prefill_quant=True)
tokenizer = load_tokenizer()

# Configuration
prompt_lengths = [2048, 8192, 16384, 32768]
batch_sizes = [1, 4, 16]

# Profile fp16
print("Profiling fp16 baseline...")
results_fp16 = profile_energy_per_token(
    model_fp16, tokenizer, prompt_lengths, batch_sizes
)

# Profile quantized
print("Profiling quantized (nuq3-1%)...")
results_quant = profile_energy_per_token(
    model_quant, tokenizer, prompt_lengths, batch_sizes
)

# Compare
print("\nENERGY COMPARISON")
print("="*80)
print(f"{'Config':25s} {'fp16 (J/tok)':15s} {'Quant (J/tok)':15s} {'Reduction'}")
print("="*80)

for config in results_fp16.keys():
    energy_fp16 = results_fp16[config]['energy_per_token_joules']
    energy_quant = results_quant[config]['energy_per_token_joules']
    reduction = (energy_fp16 - energy_quant) / energy_fp16 * 100
    
    print(f"{config:25s} {energy_fp16:.5f}       {energy_quant:.5f}       {reduction:+.1f}%")

# Save results
save_results(results_fp16, results_quant, 'energy_profiling_results.json')
```

**Expected results:**
```
Config                    fp16 (J/tok)    Quant (J/tok)   Reduction
len2048_batch1            0.00120         0.00108         +10.0%
len2048_batch4            0.00095         0.00072         +24.2%
len2048_batch16           0.00081         0.00048         +40.7%
len8192_batch1            0.00145         0.00125         +13.8%
len8192_batch4            0.00118         0.00078         +33.9%
len8192_batch16           0.00098         0.00051         +48.0%
len16384_batch4           0.00135         0.00085         +37.0%
len16384_batch16          0.00108         0.00052         +51.9%
len32768_batch4           0.00152         0.00095         +37.5%
len32768_batch16          OOM             0.00058         N/A

Key findings:
- Energy per token reduced by 40-50% for long prompts with large batches
- Energy savings roughly proportional to throughput improvement
- Quantization enables configurations that OOM with fp16 (energy savings = ∞)
```

#### Day 4: Carbon Footprint Analysis

**Calculate carbon impact:**
```python
# File: experiments/carbon_footprint.py

def calculate_carbon_impact():
    """
    Calculate carbon footprint reduction if deployed at scale
    
    Assumptions:
        - 1 billion tokens/day (typical large service)
        - Average prompt length: 8K tokens
        - Average batch size: 16
        - Energy per token: from profiling results
        - Carbon intensity: 0.5 kg CO2/kWh (US average)
    """
    
    # Parameters
    tokens_per_day = 1e9
    avg_prompt_len = 8192
    avg_batch_size = 16
    carbon_intensity = 0.5  # kg CO2/kWh
    
    # Energy per token (from profiling)
    energy_per_token_fp16 = 0.00098  # J/token (len8192_batch16)
    energy_per_token_quant = 0.00051  # J/token
    
    # Daily energy
    daily_energy_fp16 = tokens_per_day * energy_per_token_fp16  # Joules
    daily_energy_quant = tokens_per_day * energy_per_token_quant
    
    # Convert to kWh
    daily_energy_fp16_kwh = daily_energy_fp16 / (3.6e6)  # 1 kWh = 3.6e6 J
    daily_energy_quant_kwh = daily_energy_quant / (3.6e6)
    
    # Carbon emissions
    daily_carbon_fp16 = daily_energy_fp16_kwh * carbon_intensity  # kg CO2
    daily_carbon_quant = daily_energy_quant_kwh * carbon_intensity
    
    # Annual
    annual_carbon_fp16 = daily_carbon_fp16 * 365
    annual_carbon_quant = daily_carbon_quant * 365
    
    # Savings
    daily_savings = daily_carbon_fp16 - daily_carbon_quant
    annual_savings = annual_carbon_fp16 - annual_carbon_quant
    
    # Cost savings (assuming $0.10/kWh)
    electricity_cost_per_kwh = 0.10
    annual_cost_savings = (daily_energy_fp16_kwh - daily_energy_quant_kwh) * 365 * electricity_cost_per_kwh
    
    print("CARBON FOOTPRINT ANALYSIS")
    print("="*60)
    print(f"Scenario: 1B tokens/day, 8K prompts, batch size 16")
    print()
    print(f"Daily energy consumption:")
    print(f"  fp16:      {daily_energy_fp16_kwh:.1f} kWh/day")
    print(f"  Quantized: {daily_energy_quant_kwh:.1f} kWh/day")
    print(f"  Savings:   {daily_energy_fp16_kwh - daily_energy_quant_kwh:.1f} kWh/day ({(daily_energy_fp16_kwh - daily_energy_quant_kwh)/daily_energy_fp16_kwh*100:.1f}%)")
    print()
    print(f"Annual carbon emissions:")
    print(f"  fp16:      {annual_carbon_fp16:,.0f} kg CO2/year")
    print(f"  Quantized: {annual_carbon_quant:,.0f} kg CO2/year")
    print(f"  Reduction: {annual_savings:,.0f} kg CO2/year")
    print()
    print(f"Equivalent to:")
    print(f"  {annual_savings / 4000:.1f} cars removed from roads (4 tons CO2/car/year)")
    print(f"  {annual_savings / 20:.1f} trees planted (20 kg CO2/tree/year)")
    print()
    print(f"Annual cost savings: ${annual_cost_savings:,.0f}")


if __name__ == "__main__":
    calculate_carbon_impact()
```

**Expected output:**
```
CARBON FOOTPRINT ANALYSIS
============================================================
Scenario: 1B tokens/day, 8K prompts, batch size 16

Daily energy consumption:
  fp16:      272.2 kWh/day
  Quantized: 141.7 kWh/day
  Savings:   130.6 kWh/day (48.0%)

Annual carbon emissions:
  fp16:      49,669 kg CO2/year
  Quantized: 25,859 kg CO2/year
  Reduction: 23,810 kg CO2/year

Equivalent to:
  6.0 cars removed from roads (4 tons CO2/car/year)
  1,191 trees planted (20 kg CO2/tree/year)

Annual cost savings: $4,766
```

#### Day 5: Energy Visualizations

**Create energy figures for paper:**
```python
# File: experiments/plot_energy.py

import matplotlib.pyplot as plt
import json

def plot_energy_per_token():
    """Energy per token vs. prompt length"""
    
    with open('energy_profiling_results.json') as f:
        results = json.load(f)
    
    # Extract data for batch size 16
    prompt_lengths = [2048, 8192, 16384, 32768]
    energy_fp16 = []
    energy_quant = []
    
    for plen in prompt_lengths:
        key = f"len{plen}_batch16"
        if key in results['fp16']:
            energy_fp16.append(results['fp16'][key]['energy_per_token_joules'] * 1000)  # Convert to mJ
        else:
            energy_fp16.append(None)  # OOM
        
        energy_quant.append(results['quantized'][key]['energy_per_token_joules'] * 1000)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(prompt_lengths, energy_fp16, marker='o', label='fp16', linewidth=2, markersize=8)
    ax.plot(prompt_lengths, energy_quant, marker='s', label='Quantized (nuq3-1%)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Prompt Length (tokens)', fontsize=14)
    ax.set_ylabel('Energy per Token (mJ)', fontsize=14)
    ax.set_title('Energy Efficiency vs. Prompt Length (Batch Size 16)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('energy_per_token.png', dpi=300)


if __name__ == "__main__":
    plot_energy_per_token()
    # Additional plots...
```

**Deliverables Week 9:**
- ✅ Energy measurements complete for all configurations
- ✅ 40-50% energy reduction for long prompts + large batches
- ✅ Carbon footprint analysis showing significant impact at scale
- ✅ Publication-quality energy figures

---

## WEEK 10: Ablations & Analysis

### Objectives
- [ ] Ablation studies: Effect of block size, sparsity, bit-width
- [ ] Sensitivity analysis: How do design choices affect performance?
- [ ] Pareto frontier analysis: Accuracy vs. throughput vs. energy
- [ ] Identify optimal configurations

### Tasks

#### Day 1: Block Size Ablation

**Question:** How does block size affect performance?

**Test block sizes:** 64, 128, 256, 512, 1024

```python
# File: experiments/ablation_block_size.py

def ablate_block_size():
    """Test effect of block size on throughput and memory"""
    
    block_sizes = [64, 128, 256, 512, 1024]
    results = {}
    
    for block_size in block_sizes:
        print(f"\nTesting block size: {block_size}")
        
        # Create model with this block size
        model = load_quantized_model(
            "../models/Llama-2-7B-32K",
            quantization="nuq3-1%",
            prefill_quant=True,
            block_size=block_size
        )
        
        # Measure throughput on 16K prompt, batch 16
        throughput_result = measure_throughput(
            model, tokenizer,
            prompt_length=16384,
            batch_size=16
        )
        
        # Measure memory overhead
        memory_overhead = measure_memory_overhead(model, block_size)
        
        results[block_size] = {
            'throughput': throughput_result['throughput_tokens_per_sec'],
            'memory_overhead_mb': memory_overhead
        }
        
        print(f"  Throughput: {results[block_size]['throughput']:.1f} tok/s")
        print(f"  Memory overhead: {results[block_size]['memory_overhead_mb']:.1f} MB")
    
    # Find optimal
    optimal_block_size = max(results.items(), 
                            key=lambda x: x[1]['throughput'])[0]
    
    print(f"\nOptimal block size: {optimal_block_size}")
    
    return results
```

**Expected findings:**
- Block size 256-512 offers best trade-off
- Too small (64): High overhead from block management
- Too large (1024): Wasted memory pre-allocation
- Optimal: 256 (used in main experiments)

#### Day 2: Sparsity Ablation

**Question:** How does outlier sparsity affect accuracy and efficiency?

**Test sparsities:** 0% (no outliers), 0.1%, 0.5%, 1%, 2%

```python
# File: experiments/ablation_sparsity.py

def ablate_sparsity():
    """Test effect of outlier sparsity"""
    
    sparsities = [0.0, 0.001, 0.005, 0.01, 0.02]
    results = {}
    
    for sparsity in sparsities:
        print(f"\nTesting sparsity: {sparsity*100:.1f}%")
        
        # Create model with this sparsity
        model = load_quantized_model(
            "../models/Llama-2-7B-32K",
            quantization="nuq3",
            sparsity=sparsity,
            prefill_quant=True
        )
        
        # Measure accuracy
        ppl = evaluate_perplexity(model, tokenizer, dataset="wikitext2")
        
        # Measure throughput
        throughput = measure_throughput(
            model, tokenizer,
            prompt_length=16384,
            batch_size=16
        )['throughput_tokens_per_sec']
        
        # Memory footprint
        memory = measure_memory_footprint(model)
        
        results[sparsity] = {
            'perplexity': ppl,
            'throughput': throughput,
            'memory_gb': memory
        }
    
    # Plot Pareto frontier
    plot_pareto_accuracy_throughput(results)
    
    return results
```

**Expected findings:**
- 0% sparsity (no outliers): Higher accuracy degradation (~0.3 PPL)
- 1% sparsity: Best trade-off (used in main experiments)
- 2% sparsity: Diminishing returns (marginal accuracy improvement, higher memory)

#### Day 3: Bit-Width Comparison

**Question:** How do 2-bit, 3-bit, 4-bit quantization compare?

```python
# File: experiments/ablation_bitwidth.py

def compare_bitwidths():
    """Compare 2-bit, 3-bit, 4-bit quantization"""
    
    configs = [
        {'bits': 2, 'quant': 'nuq2-1%'},
        {'bits': 3, 'quant': 'nuq3-1%'},
        {'bits': 4, 'quant': 'nuq4-1%'}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['bits']}-bit quantization")
        
        model = load_quantized_model(
            "../models/Llama-2-7B-32K",
            quantization=config['quant'],
            prefill_quant=True
        )
        
        # Accuracy
        ppl = evaluate_perplexity(model, tokenizer, dataset="wikitext2")
        
        # Throughput
        throughput = measure_throughput(
            model, tokenizer,
            prompt_length=16384,
            batch_size=16
        )['throughput_tokens_per_sec']
        
        # Memory
        memory = measure_memory_footprint(model)
        
        # Energy
        energy = measure_energy_per_token(model, tokenizer, 
                                         prompt_length=16384,
                                         batch_size=16)
        
        results[config['bits']] = {
            'perplexity': ppl,
            'throughput': throughput,
            'memory_gb': memory,
            'energy_per_token_mj': energy * 1000
        }
    
    # Print comparison table
    print_comparison_table(results)
    
    return results
```

**Expected results:**
```
Bits  Perplexity  Throughput  Memory  Energy/tok
----  ----------  ----------  ------  ----------
2     5.41        45000       18 GB   0.55 mJ    (aggressive, higher degradation)
3     5.17        38000       22 GB   0.51 mJ    (best balance) ✓
4     5.13        32000       28 GB   0.62 mJ    (conservative, minimal degradation)

Conclusion: 3-bit offers best accuracy-efficiency trade-off
```

#### Day 4: Pareto Frontier Analysis

**Generate Pareto frontier: Accuracy vs. Throughput vs. Energy**

```python
# File: experiments/pareto_analysis.py

def generate_pareto_frontier():
    """
    Generate 3D Pareto frontier: Perplexity vs. Throughput vs. Energy
    
    Configurations to test:
        - fp16 (baseline)
        - nuq4 (no sparsity)
        - nuq4-1%
        - nuq3 (no sparsity)
        - nuq3-1%
        - nuq2-1%
    """
    
    configs = [
        {'name': 'fp16', 'quant': None},
        {'name': 'nuq4', 'quant': 'nuq4', 'sparsity': 0.0},
        {'name': 'nuq4-1%', 'quant': 'nuq4-1%'},
        {'name': 'nuq3', 'quant': 'nuq3', 'sparsity': 0.0},
        {'name': 'nuq3-1%', 'quant': 'nuq3-1%'},
        {'name': 'nuq2-1%', 'quant': 'nuq2-1%'}
    ]
    
    results = []
    
    for config in configs:
        print(f"Evaluating {config['name']}...")
        
        # Load model
        if config['quant'] is None:
            model = load_model_fp16()
        else:
            model = load_quantized_model(
                quant=config['quant'],
                prefill_quant=True
            )
        
        # Measure all metrics
        ppl = evaluate_perplexity(model, tokenizer)
        throughput = measure_throughput(model, tokenizer, 16384, 16)['throughput_tokens_per_sec']
        energy = measure_energy_per_token(model, tokenizer, 16384, 16)
        
        results.append({
            'name': config['name'],
            'perplexity': ppl,
            'throughput': throughput,
            'energy_per_token_mj': energy * 1000
        })
    
    # Plot 3D Pareto frontier
    plot_3d_pareto(results)
    
    # Find Pareto-optimal points
    pareto_optimal = find_pareto_optimal(results)
    
    print("\nPareto-optimal configurations:")
    for config in pareto_optimal:
        print(f"  {config['name']}: "
              f"PPL={config['perplexity']:.2f}, "
              f"Throughput={config['throughput']:.0f} tok/s, "
              f"Energy={config['energy_per_token_mj']:.2f} mJ/tok")
    
    return results
```

#### Day 5: Sensitivity Analysis

**Test robustness to:**
1. Different calibration data
2. Different model sizes (7B vs. 13B)
3. Different prompt distributions

**Deliverables Week 10:**
- ✅ Block size ablation (optimal: 256)
- ✅ Sparsity ablation (optimal: 1%)
- ✅ Bit-width comparison (3-bit best trade-off)
- ✅ Pareto frontier analysis identifying optimal configs
- ✅ Sensitivity analysis showing robustness

---

## WEEK 11-12: Paper Writing & Submission

### Objectives
- [ ] Write complete draft paper
- [ ] Create all figures and tables
- [ ] Internal review and revision
- [ ] ArXiv submission
- [ ] ICML submission (January 31 deadline)

### Paper Structure

**Title:** "Efficient Prefill: Blocked Quantization for Batched LLM Inference"

**Abstract (150 words):**
```
Long-context LLM inference is bottlenecked by KV cache memory during the prefill phase,
limiting batch sizes and throughput. While recent work addresses generation-phase quantization,
prefill remains unexplored. We introduce Blocked KV Quantization, combining (1) batched per-channel
quantization for variable-length prompts, (2) blocked CSR/CSC allocation eliminating reallocation
overhead, and (3) fused attention-quantization kernels. On Llama-2-7B-32K, our method achieves
2.8× throughput improvement for 16K prompts with batch size 16, enabling 2× larger batches while
maintaining <0.1 perplexity degradation on LongBench. Energy profiling shows 48% reduction in
energy per token, equivalent to removing 6 cars from roads annually if deployed at billion-token
scale. We provide the first hardware-validated analysis of prefill quantization and release
open-source implementations extending KVQuant.
```

**Section Outline:**

1. **Introduction** (1 page)
   - Motivation: Long-context inference bottleneck
   - Gap: KVQuant addresses generation, prefill unexplored
   - Contributions: Batched quantization, blocked allocation, energy analysis

2. **Background** (0.5 pages)
   - LLM inference phases (prefill vs. generation)
   - KV cache quantization basics
   - Memory allocation challenges

3. **Method** (2 pages)
   - **3.1 Batched Quantization for Variable-Length Prompts**
     - Per-channel quantization with padding handling
     - Non-uniform datatype (nuqX)
     - Outlier detection and sparse storage
   
   - **3.2 Blocked Memory Allocation**
     - CSC/CSR blocked structure
     - Zero-copy append operations
     - Conversion to standard format for attention
   
   - **3.3 Fused Kernels (if implemented)**
     - Load-quantize-store pipeline
     - Memory bandwidth optimization

4. **Experiments** (3 pages)
   - **4.1 Experimental Setup**
     - Models: Llama-2-7B/13B/30B-32K
     - Datasets: LongBench, RULER, Passkey
     - Baselines: fp16, KVQuant (generation-only)
   
   - **4.2 Accuracy Validation**
     - Table 1: LongBench results
     - Table 2: RULER results
     - Figure 1: Passkey retrieval accuracy vs. context length
   
   - **4.3 Throughput Benchmarking**
     - Figure 2: Throughput vs. batch size
     - Figure 3: Memory usage enabling larger batches
     - Table 3: Speedup across configurations
   
   - **4.4 Energy Profiling**
     - Figure 4: Energy per token vs. prompt length
     - Table 4: Carbon footprint at scale
   
   - **4.5 Ablation Studies**
     - Figure 5: Effect of block size, sparsity, bit-width
     - Figure 6: Pareto frontier (accuracy vs. throughput vs. energy)

5. **Related Work** (0.5 pages)
   - KV cache quantization (KVQuant, KIVI, Atom, FlexGen)
   - Prefill optimization (MInference, Flash Attention)
   - Energy-efficient inference

6. **Discussion** (0.5 pages)
   - When does prefill quantization help most? (long prompts + large batches)
   - Limitations: Requires offline calibration
   - Future work: Dynamic bit allocation, on-device inference

7. **Conclusion** (0.25 pages)

8. **Appendix** (2-3 pages)
   - Implementation details
   - Additional ablations
   - Full numerical results

### Writing Timeline

#### Day 1-2 (Week 11): Draft Sections 1-3
- Introduction: Clearly motivate problem
- Background: Concise but complete
- Method: Detailed technical description with pseudocode

#### Day 3-4 (Week 11): Create All Figures & Tables
- Use matplotlib for publication-quality figures (300 dpi)
- Tables in LaTeX format
- Color scheme: Colorblind-friendly
- All figures must have captions and be referenced in text

#### Day 5-7 (Week 11): Draft Sections 4-6
- Experiments: Present results clearly, avoid redundancy
- Related work: Position clearly vs. prior work
- Discussion: Insightful analysis, not just repetition

#### Day 8-10 (Week 12): Revision & Polishing
- Read entire paper out loud (catch awkward phrasing)
- Check math notation consistency
- Verify all claims are supported by results
- Spell check and grammar check

#### Day 11 (Week 12): Internal Review
- Send to colleagues for feedback
- Address reviewer comments

#### Day 12 (Week 12): Final Submission Prep
- Convert to ICML format (icml2026.sty)
- Check page limit (8 pages main + unlimited appendix)
- Verify all figures/tables are cited
- Supplementary materials: Code link, dataset details

#### Day 13 (Week 12): Submit!
- ArXiv submission (morning)
- ICML submission (afternoon, deadline is Jan 31, 11:59 PM Pacific)

### LaTeX Template

```latex
\documentclass{article}
\usepackage{icml2026}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{hyperref}

\title{Efficient Prefill: Blocked Quantization for Batched LLM Inference}

\author{
  Your Name \\
  Your Institution \\
  \texttt{your.email@institution.edu}
}

\begin{document}

\maketitle

\begin{abstract}
Long-context LLM inference is bottlenecked by KV cache memory during the prefill phase...
\end{abstract}

\section{Introduction}
Large language models (LLMs) have revolutionized...

% ... (rest of paper)

\end{document}
```

**Deliverables Week 11-12:**
- ✅ Complete draft paper (8 pages + appendix)
- ✅ All figures and tables publication-ready
- ✅ Internal review complete
- ✅ ArXiv preprint submitted
- ✅ ICML 2026 submission complete

---

## SLURM JOB TEMPLATES

### General Template

```bash
#!/bin/bash
#SBATCH --job-name=kvquant_prefill
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=48:00:00
#SBATCH --mem=1024GB
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Environment setup
module load cuda/12.4.0
module load python/3.10
module load pytorch/2.4.0

# Navigate to project directory
cd $PLG_GROUPS_STORAGE/kvquant-prefill
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16

# Run experiment
python experiments/your_experiment.py

# Cleanup
echo "Job completed at $(date)"
```

### Parallel Job Array Template (for sweeps)

```bash
#!/bin/bash
#SBATCH --job-name=kvquant_sweep
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-19  # 20 parallel jobs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=12:00:00
#SBATCH --mem=1024GB

# Load environment
cd $PLG_GROUPS_STORAGE/kvquant-prefill
source venv/bin/activate

# Map array task ID to configuration
CONFIGS=(
  "len512_batch1"
  "len512_batch4"
  "len512_batch16"
  "len2048_batch1"
  "len2048_batch4"
  "len2048_batch16"
  # ... (all 20 configurations)
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Run experiment for this configuration
python experiments/benchmark_throughput.py --config $CONFIG
```

---

## CODE REPOSITORY STRUCTURE

```
kvquant-prefill/
├── README.md
├── requirements.txt
├── setup.py
│
├── kvquant/  # Core library
│   ├── __init__.py
│   ├── batched_quant.py       # Batched quantization algorithm
│   ├── blocked_sparse.py      # Blocked CSC/CSR matrices
│   ├── prefill_quant.py       # Main prefill quantization module
│   ├── nuq.py                 # Non-uniform quantization (from KVQuant)
│   └── kernels/
│       ├── triton_kernels.py  # Triton CUDA kernels
│       └── pytorch_fused.py   # PyTorch loop fusion
│
├── experiments/  # Evaluation scripts
│   ├── evaluate_longbench.py
│   ├── evaluate_ruler.py
│   ├── evaluate_passkey.py
│   ├── benchmark_throughput.py
│   ├── profile_energy.py
│   ├── ablation_block_size.py
│   ├── ablation_sparsity.py
│   ├── pareto_analysis.py
│   └── statistical_tests.py
│
├── utils/  # Utilities
│   ├── energy_monitor.py
│   ├── energy_profiler.py
│   └── plotting.py
│
├── scripts/  # Helper scripts
│   ├── download_models.sh
│   ├── download_datasets.sh
│   ├── setup_wandb.py
│   └── run_baseline_measurement.py
│
├── slurm_jobs/  # SLURM job scripts
│   ├── eval_longbench.sh
│   ├── benchmark_throughput.sh
│   ├── profile_energy.sh
│   └── ablations.sh
│
├── paper/  # LaTeX source
│   ├── main.tex
│   ├── figures/
│   └── tables/
│
├── results/  # Experiment results
│   ├── longbench_results.json
│   ├── throughput_results.json
│   ├── energy_results.json
│   └── ablation_results.json
│
└── logs/  # Slurm job logs
```

---

## RISK MITIGATION & CONTINGENCY PLANS

### Risk 1: Accuracy Degradation >0.5 PPL
**Mitigation:**
- Use 4-bit instead of 3-bit (more conservative)
- Increase sparsity to 2%
- Per-token scaling factors for Keys (instead of per-channel)

**Fallback:**
- Even with 1 PPL degradation, still publishable if throughput gains are large
- Reframe as "accuracy-efficiency trade-off exploration"

### Risk 2: Throughput Improvement <2×
**Mitigation:**
- Focus on memory reduction enabling larger batches (key contribution)
- Emphasize energy savings (unique angle)

**Fallback:**
- 1.5× speedup is still significant and publishable
- Highlight that this is **first work** on prefill quantization

### Risk 3: Implementation Complexity / Bugs
**Mitigation:**
- Extensive unit tests for each component
- Validate against KVQuant baseline frequently
- Start simple, add complexity incrementally

**Fallback:**
- Simplify: Drop fused kernels, focus on blocked allocation
- Use naive PyTorch implementation if Triton is problematic

### Risk 4: ICML Rejection
**Mitigation:**
- Strong baselines, rigorous evaluation
- Clear positioning vs. KVQuant
- Novelty: First prefill quantization + energy analysis

**Backup Venues:**
- NeurIPS 2026 (Sep deadline)
- MLSys 2026 (Feb deadline)
- EMNLP 2026 (May deadline)

### Risk 5: GPU Allocation Issues
**Mitigation:**
- Run smaller experiments on fewer GPUs
- Use job arrays to parallelize
- Submit jobs during off-peak hours

**Fallback:**
- Scale down to 7B model only (skip 13B/30B)
- Reduce number of ablations
- Use shorter prompt lengths (8K max instead of 32K)

---

## SUCCESS METRICS & CHECKPOINTS

### Week 2 Checkpoint
- ✅ KVQuant baseline reproduced
- ✅ Batched quantization algorithm designed
- **Decision point:** If can't reproduce KVQuant, reassess project scope

### Week 4 Checkpoint
- ✅ Blocked allocation implemented
- ✅ 10× allocation speedup demonstrated
- **Decision point:** If allocation doesn't speed up, focus on other contributions

### Week 7 Checkpoint
- ✅ Accuracy <0.5 PPL degradation validated
- **Decision point:** If degradation >0.5 PPL, switch to 4-bit or increase sparsity

### Week 9 Checkpoint
- ✅ 2× throughput improvement demonstrated
- ✅ Energy savings measured
- **Decision point:** If throughput <1.5×, reframe paper focus on energy

### Week 12 Checkpoint
- ✅ Paper submitted to ICML
- **Decision point:** If paper not ready, submit to NeurIPS instead

---

## FINAL CHECKLIST

### Before Starting
- [ ] Athena account activated and quota checked
- [ ] GitHub repository created
- [ ] WandB account set up
- [ ] Models and datasets downloaded
- [ ] Environment tested with KVQuant reproduction

### During Project
- [ ] Weekly sync meetings with advisor
- [ ] Daily log of progress and issues
- [ ] Code committed to GitHub daily
- [ ] Results logged to WandB

### Before Submission
- [ ] All experiments complete
- [ ] All figures publication-quality
- [ ] Paper proofread by colleagues
- [ ] Code released on GitHub
- [ ] Supplementary materials prepared
- [ ] ArXiv submission complete
- [ ] ICML submission complete

---

## APPENDIX: USEFUL COMMANDS

### Environment Setup
```bash
# SSH with port forwarding for Jupyter
ssh -L 8888:localhost:8888 username@athena.cyfronet.pl

# Check GPU availability
sinfo -p plgrid-gpu-a100

# Monitor job
squeue -u username
watch -n 1 'squeue -u username'

# Cancel job
scancel <job_id>

# Check storage usage
du -sh $PLG_GROUPS_STORAGE/kvquant-prefill
```

### Git Workflow
```bash
# Daily commit
git add .
git commit -m "Week X Day Y: <brief description>"
git push origin main

# Create feature branch
git checkout -b feature/blocked-allocation
# ... make changes ...
git commit -m "Implement blocked CSC matrix"
git push origin feature/blocked-allocation
```

### Python Debugging
```python
# Interactive debugger
import pdb; pdb.set_trace()

# Memory profiling
from memory_profiler import profile
@profile
def my_function():
    pass

# GPU memory tracking
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

**END OF EXECUTION PLAN**

*This is a living document. Update as you progress and encounter new challenges. Good luck with your research!*
