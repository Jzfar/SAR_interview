import tensorflow as tf

# 定义输入数据
batch_size = 1
input_height, input_width, input_channels = 32, 32, 4
output_channels = 7
kernel_size = 3

# 随机生成输入数据和卷积核权重
inputs = tf.random.normal([batch_size, input_height, input_width, input_channels])
kernels = tf.random.normal([kernel_size, kernel_size, input_channels, output_channels])
bias = tf.random.normal([output_channels])

# 手动实现卷积操作
def manual_conv2d(inputs, kernels, bias, stride=1, padding='valid'):
    batch_size, input_height, input_width, input_channels = inputs.shape
    kernel_size, _, _, output_channels = kernels.shape

    # 计算输出特征图的尺寸
    if padding == 'valid':
        output_height = (input_height - kernel_size) // stride + 1
        output_width = (input_width - kernel_size) // stride + 1
        pad_height, pad_width = 0, 0
    elif padding == 'same':
        output_height = input_height
        output_width = input_width
        # 计算填充大小
        pad_height = ((output_height - 1) * stride + kernel_size - input_height) // 2
        pad_width = ((output_width - 1) * stride + kernel_size - input_width) // 2
        # 对输入数据进行填充
        inputs = tf.pad(inputs, [[0, 0], [pad_height, pad_height], [pad_width, pad_width], [0, 0]])
    else:
        raise ValueError("padding 必须是 'valid' 或 'same'")

    # 初始化输出特征图
    outputs = tf.zeros([batch_size, output_height, output_width, output_channels])

    # 执行卷积操作
    for b in range(batch_size):  # 遍历批次
        for i in range(0, output_height):  # 遍历输出高度
            for j in range(0, output_width):  # 遍历输出宽度
                for k in range(output_channels):  # 遍历输出通道
                    # 获取当前卷积区域
                    input_slice = inputs[b, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size, :]
                    # 计算卷积结果
                    outputs = tf.tensor_scatter_nd_add(
                        outputs,
                        [[b, i, j, k]],
                        [tf.reduce_sum(input_slice * kernels[:, :, :, k]) + bias[k]]
                    )
    return outputs

# 调用手动实现的卷积函数
outputs = manual_conv2d(inputs, kernels, bias, stride=1, padding='same')

# 应用 ReLU 激活函数
outputs = tf.nn.relu(outputs)

# 打印输出结果
print("输入形状:", inputs.shape)
print("卷积核形状:", kernels.shape)
print("输出形状:", outputs.shape)
print("输出值:\n", outputs.numpy())