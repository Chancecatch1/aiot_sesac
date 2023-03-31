s ='나는 빨리 집에 가고 싶다'

ss = s.split()
print(ss)
print('-'.join(ss))
print('='*50)

s1 ='나는 빨리 {} 가고 싶다. {} 공부를 해야 집에 {} 있다'
s2 ='하지만, 집에, 갈수'

s2s = s2.split(', ')
print(s2s)
s1s = s1.format(s2s[1], s2s[0], s2s[2])
print(s1s)