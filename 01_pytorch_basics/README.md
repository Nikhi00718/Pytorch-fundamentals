# PyTorch Basics

This folder contains a beginner-friendly notebook that introduces the core concepts of PyTorch.
It focuses on tensors and the essential operations required to build a strong foundation for deep learning.

---

## Topics Covered

* Creating tensors
* Random, zero, and one tensors
* Tensor shapes and data types
* Basic tensor operations
* Matrix multiplication
* Tensor reshaping, stacking, and squeezing

---

## Notebook

* `00_pytorch_fundamentals.ipynb`

---

## Notebook Structure with Key Syntax

### Step 1: Introduction to Tensors

Explains tensors as the main data structure in PyTorch.

**Syntax**

```
import torch
```

---

### Step 2: Creating Tensors

Creates tensors with specific values.

**Syntax**

```
torch.tensor(data)
```

---

### Step 3: Random Tensors

Generates tensors with random values.

**Syntax**

```
torch.rand(size)
torch.randn(size)
```

---

### Step 4: Zero and One Tensors

Creates tensors filled with zeros or ones.

**Syntax**

```
torch.zeros(size)
torch.ones(size)
```

---

### Step 5: Tensor Data Types

Checks and changes data types.

**Syntax**

```
tensor.dtype
tensor.type(dtype)
```

---

### Step 6: Tensor Shape and Size

Checks tensor dimensions.

**Syntax**

```
tensor.shape
tensor.size()
```

---

### Step 7: Basic Tensor Operations

Performs arithmetic operations.

**Syntax**

```
tensor1 + tensor2
tensor1 - tensor2
tensor1 * tensor2
tensor1 / tensor2
torch.add(tensor1, tensor2)
```

---

### Step 8: Matrix Multiplication

Performs matrix multiplication.

**Syntax**

```
torch.matmul(tensor1, tensor2)
tensor1 @ tensor2
```

---

### Step 9: Tensor Statistics

Finds min, max, mean, and sum.

**Syntax**

```
torch.min(tensor)
torch.max(tensor)
torch.mean(tensor)
torch.sum(tensor)
```

---

### Step 10: Reshaping Tensors

Changes tensor shape.

**Syntax**

```
tensor.reshape(new_shape)
tensor.view(new_shape)
```

---

### Step 11: Stacking Tensors

Combines tensors together.

**Syntax**

```
torch.stack([tensor1, tensor2])
```

---

### Step 12: Squeeze and Unsqueeze

Removes or adds dimensions.

**Syntax**

```
tensor.squeeze()
tensor.unsqueeze(dim)
```

---

### Step 13: Tensor Indexing

Accesses specific elements.

**Syntax**

```
tensor[index]
tensor[row, column]
tensor[:, 0]
```

---

### Step 14: PyTorch and NumPy Conversion

Converts between tensors and arrays.

**Syntax**

```
tensor.numpy()
torch.from_numpy(array)
```

---

## How to Run

1. Open the notebook in Google Colab, Jupyter, or VS Code.
2. Run the cells from top to bottom.

---

## Requirements

* Python 3.x
* PyTorch

Install PyTorch:

```
pip install torch
```
