from enum import Enum

class TestEnum(Enum):
    ONE = 1
    TWO = 2
    THREE = 3

a = TestEnum.ONE
print(1 == TestEnum.ONE.value)