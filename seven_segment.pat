# BasicProp Pattern File - 7-Segment Display Character Recognition
#
# Segment layout:
#   +--1--+
#   5     2
#   +--7--+
#   6     3
#   +--4--+
#
# Inputs:  7 values [seg1 seg2 seg3 seg4 seg5 seg6 seg7]
#          1 = segment lit, 0 = segment off
#
# Outputs: 7 values = ASCII code of character, 7-bit binary MSB first
#          e.g. '3' -> ASCII 51 -> 0110011
#
Number of patterns = 17
Number of inputs = 7
Number of outputs = 7
[Patterns]
1 1 1 1 1 1 0    0 1 1 0 0 0 0
0 1 1 0 0 0 0    0 1 1 0 0 0 1
1 1 0 1 0 1 1    0 1 1 0 0 1 0
1 1 1 1 0 0 1    0 1 1 0 0 1 1
0 1 1 0 1 0 1    0 1 1 0 1 0 0
1 0 1 1 1 0 1    0 1 1 0 1 0 1
1 0 1 1 1 1 1    0 1 1 0 1 1 0
1 1 1 0 0 0 0    0 1 1 0 1 1 1
1 1 1 1 1 1 1    0 1 1 1 0 0 0
1 1 1 1 1 0 1    0 1 1 1 0 0 1
1 1 1 0 1 1 1    1 0 0 0 0 0 1
0 0 1 1 1 1 1    1 0 0 0 0 1 0
1 0 0 1 1 1 0    1 0 0 0 0 1 1
0 1 1 1 0 1 1    1 0 0 0 1 0 0
1 0 0 1 1 1 1    1 0 0 0 1 0 1
1 0 0 0 1 1 1    1 0 0 0 1 1 0
0 1 1 0 1 1 1    1 0 0 1 0 0 0
