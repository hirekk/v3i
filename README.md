# Octonion Perceptron

This project implements a classic perceptron from the ground up, with a unique twist: it uses octonions for weights, inputs, and outputs instead of traditional real numbers.

## What are Octonions?

Octonions are an 8-dimensional hypercomplex number system that extends the quaternions. They can be written in the form:

```
a + bi + cj + dk + el + fm + gn + ho
```

where a, b, c, d, e, f, g, h are real numbers, and i, j, k, l, m, n, o are the imaginary basis elements.

Octonions have some interesting properties:
- They are non-commutative (a×b ≠ b×a)
- They are non-associative ((a×b)×c ≠ a×(b×c))
- They form the largest normed division algebra over the real numbers

## Project Structure

- `src/v3i/models/octonion_perceptron.py`: Core implementation of the Octonion class and OctonionPerceptron
- `src/v3i/examples/octonion_perceptron_mnist.py`: Example of using the octonion perceptron with MNIST data
- `src/v3i/data/extract.py`: Utilities for downloading and processing MNIST data

## Implementation Details

The implementation includes:

1. **Octonion Class**: A custom implementation of octonions with basic operations like addition and multiplication
2. **OctonionPerceptron**: A classic perceptron model that uses octonions for weights and computations
3. **Training Loop**: A simple training procedure for the octonion perceptron

## Usage

To run the MNIST example:

```bash
python -m src.v3i.examples.octonion_perceptron_mnist
```

This will:
1. Download the MNIST dataset
2. Create a binary classification problem (distinguishing between digits 0 and 1)
3. Train an octonion perceptron on this data
4. Plot the training accuracy over epochs

## Notes on the Implementation

- The octonion multiplication implemented is a simplified version and doesn't fully capture the complete octonion algebra
- For practical purposes, the activation function only uses the real part (first component) of the octonion
- This is primarily an educational implementation to explore the concept of using hypercomplex numbers in neural networks

## Future Directions

- Implement proper octonion multiplication using the Fano plane or multiplication tables
- Explore different activation functions that better utilize the 8-dimensional nature of octonions
- Compare performance with traditional perceptrons using real numbers
- Extend to multi-layer networks with octonion weights
