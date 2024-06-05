import torch
import numpy as np


##################################################
#  Tensor
##################################################

#  初始化
def tensor_initialization(from_type):
    if from_type == "data":
        # 从数据初始化
        data = [[1, 2], [3, 4]]
        x_data = torch.tensor(data)
        print(f"从数据初始化:\n{x_data}\n")


    elif from_type == "numpy":
        data = np.random.rand(2, 3)
        np_array = np.array(data)
        x_np = torch.from_numpy(np_array)
        print(f"从numpy初始化:\n{x_np}\n")

    elif from_type == "tensor":
        data = [[1, 2], [3, 4]]
        x_data = torch.tensor(data)
        print("从另一个张量初始化：")
        x_ones = torch.ones_like(x_data)  # retains the properties of x_data
        print(f"Ones Tensor: \n {x_ones} \n")

        x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
        print(f"Random Tensor: \n {x_rand} \n")

    elif from_type == "random":
        # 创建了一个形状为(2, 3)的张量
        shape = (2, 3,)
        # 元素是随机的浮点数，范围从0到1
        rand_tensor = torch.rand(shape)
        # 元素全部为1
        ones_tensor = torch.ones(shape)
        # 元素全部为0
        zeros_tensor = torch.zeros(shape)

        print(f"Random Tensor: \n {rand_tensor} \n")
        print(f"Ones Tensor: \n {ones_tensor} \n")
        print(f"Zeros Tensor: \n {zeros_tensor}")

# 张量属性
def tensor_attributes():
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

# 张量操作
def tensor_operation():
    tensor = torch.rand(3, 4)

    # 在GPu上运行tensor
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        print(f"Device tensor is stored on: {tensor.device}")
    print("*" * 50)
    # tensor切片
    tensor = torch.ones(4, 4)
    # 4*4的形状，第2列，索引1的位置为0
    tensor[:, 1] = 0
    print(tensor)
    print("*" * 50)
    # dim=1 表示拼接操作是沿着列进行的
    t1 = torch.cat([tensor, tensor, tensor], dim=-1)
    print(t1)



if __name__ == '__main__':
    from_type_dict = {
        "data": "data",
        "numpy": "numpy",
        "tensor": "tensor",
        "random": "random",
    }
    # tensor_initialization(from_type_dict["data"])
    print("*" * 100)
    tensor_attributes()
    print("*"*100)
    tensor_operation()
