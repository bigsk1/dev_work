import torch

def test_pytorch_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Testing with a simple operation.")

        # Create a random tensor
        x = torch.rand(5, 3)
        print("Original Tensor:\n", x)

        # Move the tensor to GPU
        x = x.cuda()
        print("Tensor on CUDA:\n", x)

        # Perform a simple addition operation
        y = x + 1
        print("Tensor after addition:\n", y)

        print("PyTorch and CUDA are working correctly!")
    else:
        print("CUDA is not available. Please check your installation.")

if __name__ == "__main__":
    test_pytorch_cuda()
