rf_aero: initially ran as a regression, changing to a classification since value of stability index are mostly 100 (we're looking for the ones that aren't)


info on aero set: count    150000.000000
mean         91.333445
std          24.773129
min           0.000000
25%         100.000000
50%         100.000000
75%         100.000000
max         100.000000
Name: stability_index, dtype: float64
stability_index
100.00    127871
0.00        6031
98.39         10
99.19          9
90.68          8
87.11          8
65.27          8
76.11          8
97.87          8
81.38          7

rf_aero error currently .999998 (too high need to modify model)
  variable importance in aero set is downforce_n (60%), drag_n (23%), speed (9%), and wing angle (7%), drs apparently doesn't matter
