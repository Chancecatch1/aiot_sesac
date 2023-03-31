def hi():
    
    print('사용자 정의 함수 실행')

    print('Hello World')

    print('사용자 정의 함수 종료')



def addition(m, n):
  k = m + n
  return k


subtraction = lambda m, n : m - n


def add_all(*inputs): 
    result = 0 
    
    for i in inputs: 
        result = result + i 
    return result
