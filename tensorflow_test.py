import tensorflow as tf

def test_tensorflow():
    # Print TensorFlow version
    print("TensorFlow Version:", tf.__version__)

    # Check for GPU availability
    gpu_available = tf.config.list_physical_devices('GPU')
    print("GPU Available:", bool(gpu_available))

    # Perform a basic TensorFlow operation
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    c = tf.matmul(a, b)

    print("Result of a matrix multiplication operation:")
    print(c)

if __name__ == "__main__":
    test_tensorflow()
