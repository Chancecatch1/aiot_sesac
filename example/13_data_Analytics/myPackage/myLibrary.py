def hi():
    
    print('Hello Package')


class myClass:
    def func_1(self):
        print('Hello Class')



class Calculator:
    def inputData(self, m, n):
        self.m = m
        self.n = n
    def addtion(self):
        result = self.m + self.n
        return result


class Calculator_2:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        print('메모리에 인스턴스가 %d, %d 값과 함께 생성되었습니다.' % (m, n))
    def inputData_2(self, m, n):
        self.m = m
        self.n = n
    def subtraction(self):
        result = self.m - self.n
        return result


class myComputer(Calculator, Calculator_2):
    def multiplication(self):
        result = self.m * self.n
        return result
    def division(self):
        result = self.m / self.n
        return result
