| Model | Exact match | Field F1 | Valid JSON (schema) | p50 latency | Throughput | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct, zero-shot | 12.1% | 84.7% | 52.8% | 4.38 s | 1.39/s | 3.3 GB |
| Qwen2.5-1.5B-Instruct, 3-shot | 5.5% | 82.7% | 44.7% | 5.43 s | 0.75/s | 3.4 GB |
| **Sieve: 1.5B + LoRA (bf16)** | 92.0% | 99.5% | 99.5% | 4.49 s | 1.72/s | 3.3 GB |

| Field accuracy | base-zero | base-few | finetuned |
|---|---:|---:|---:|
| `title` | 97.5% | 97.0% | 98.5% |
| `company` | 83.4% | 87.4% | 100.0% |
| `location` | 97.5% | 97.0% | 100.0% |
| `work_mode` | 82.4% | 80.4% | 99.0% |
| `employment_type` | 78.9% | 69.8% | 99.5% |
| `seniority` | 55.3% | 55.3% | 99.0% |
| `salary` | 59.3% | 66.3% | 100.0% |
| `experience_years_min` | 98.5% | 98.0% | 100.0% |
| `skills` | 47.2% | 40.2% | 95.0% |

| Run | LoRA rank | LR | Epochs | Trainable params | Val loss | Val exact match | Val field F1 | Val valid JSON | Peak train VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r16-lr1e-4 | 16 | 1e-04 | 1 | 18.5M | 0.0184 | 53.5% | 96.3% | 94.9% | 2.53 GB |
| r8-lr2e-4 | 8 | 2e-04 | 1 | 9.2M | 0.0165 | 60.6% | 97.1% | 94.9% | 2.47 GB |
| r16-lr2e-4 | 16 | 2e-04 | 1 | 18.5M | 0.0126 | 74.7% | 98.3% | 97.0% | 2.53 GB |
| r32-lr2e-4 | 32 | 2e-04 | 1 | 36.9M | 0.0090 | 76.8% | 98.6% | 100.0% | 2.71 GB |
| r32-lr2e-4-ep2 | 32 | 2e-04 | 2 | 36.9M | 0.0065 | 91.9% | 99.5% | 99.0% | 2.71 GB |
