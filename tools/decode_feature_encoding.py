from base64 import encode
import sys
'''
Input: Encoded Feature Value, corresponding to number of row in q_table 
'''
encoded_feature_number = int(sys.argv[1])

feature_bases = [5, 5, 5, 5, 16, 2]
feature_values = []

for base in feature_bases:
    encoded_feature_number, feature_value = divmod(encoded_feature_number, base)
    feature_values.append(feature_value)
    
print(feature_values)

for idx, feature_value in enumerate(feature_values):
    if idx == 0:
        print('shortest coin distance:')
    if idx == 1:
        print('shortest crate distance:')
    if idx == 2:
        print('shortest opponent distance:')
    if idx == 3:
        print('shortest safety distance:')
    if idx < 4:
        if feature_value == 0:
            print('north')
        if feature_value == 1:
            print('south')
        if feature_value == 2:
            print('east')
        if feature_value == 3:
            print('west')
        if feature_value == 4:
            print('n/a')
    if idx == 4:
        print('move to danger:')
        if feature_value == 0: #0000
            print('n/a')
        if feature_value == 1: #0001
            print('west')
        if feature_value == 2: #0010
            print('east')
        if feature_value == 3: #0011
            print('west; east')
        if feature_value == 4: #0100
            print('south')
        if feature_value == 5: #0101
            print('south; east')
        if feature_value == 6: #0110
            print('south; west')
        if feature_value == 7: #0111
            print('south; west; east')
        if feature_value == 8: #1000
            print('north')
        if feature_value == 9: #1001
            print('north; west')
        if feature_value == 10: #1010
            print('north; east')
        if feature_value == 11: #1011
            print('north; east; west')
        if feature_value == 12: #1100
            print('north; south')
        if feature_value == 13: #1101
            print('north; south, west')
        if feature_value == 14: #1110
            print('north; south; east')
        if feature_value == 15: #1111
            print('north; south; east; west')
    if idx == 5:
        if feature_value == 0:
            print('bad bomb')
        else:
            print('good bomb')