# This is a sample Python script.
import torch

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    a = [0,1,2,3,4,5,6,7,8,9]
    print(a[-7:-2])
    a_tensor = torch.tensor(a)
    tensor_norm = torch.norm(a_tensor[-7:-2], p=1)
    print(tensor_norm)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
