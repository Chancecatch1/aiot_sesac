# open
f = open('./test.txt', 'w', encoding='utf-8')
data = '안녕하세요'
f.write(data)
f.close()

f = open('./test.txt', 'r', encoding='utf-8')
data = f.readline()
print(data)
f.close()

# pickle
import pickle
a_obj = {'test':'테스트'}
file = open('./test.pkl', 'wb')
pickle.dump(a_obj, file)
file.close()

file = open('./test.pkl', 'rb')
data = pickle.load(file)
print(data)
file.close()

# json
import json
file = open('./test.json', 'w')
json.dump(['foo', {'bar': ('baz', None, 1.0, 2)}], file)
file.close()

file = open('./test.json', 'r')
data = json.load(file)
print(data)
file.close()

print('='*50)

# 실습
from SESAC_AIoT.example.bank1 import Customer

f = open('./customers.txt', 'r', encoding='utf-8')
# data_r = f.read()
# print(data_r)
# print('='*50)
# data_rline = f.readline()
# print(data_rline)
# print('='*50)
data_rlines = f.readlines()
print(data_rlines)
print('='*50)

customer_list = []
for line in data_rlines:
    c_name, c_id = line.split()
    customer = Customer(c_name, c_id)
    customer_list.append([customer.get_id(), customer.get_name()])

print(customer)
print(customer_list)
f.close()

